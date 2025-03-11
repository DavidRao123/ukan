import os, glob, copy
import numpy as np
from torchvision import transforms
from models.gan_networks import UnetGenerator, get_norm_layer
import tifffile
import random
from utils.util import ForwardOTF, Cal_Otf
from skimage.transform import resize
import torch
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset
from models.trcan import ESRT_noET

class SelmaDataset( Dataset ):
    """
    The SelmaDataset reads 1C hig-res simulation images
    Generate low-res and his-res pair using OTF or GAN
    """
    def __init__(self, data_dir:str="./hr", patch_size:int=256, npatch_per_image:int=200, genMethod:str='OTF'):
        self.hr_dir = data_dir
        self.ids = [folder for folder in os.listdir(self.hr_dir)
                    if os.path.isdir(os.path.join(data_dir,folder))]
        self.img_size = patch_size
        self.npatch_per_image = npatch_per_image
        self.channels = ["A"]
        if not genMethod.upper() in ['GAN', 'OTF', 'MIX', 'COMB']:
            print( 'Error: the simulation method is not supported!' )

        self.genMethod = genMethod

        self.GT_files = glob.glob("{}/*".format(self.hr_dir))
        hr_imgfile = random.choice(self.GT_files)
        hr_ch = sitk.GetArrayFromImage( sitk.ReadImage(hr_imgfile) )
        self.Size = hr_ch.shape

        if genMethod != 'OTF':
            self.generator = self.get_generator()

            transform_list = [transforms.Grayscale(1),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))]
            self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len( self.GT_files )*self.npatch_per_image

    def get_ch(self, file_full_path):
        filename = os.path.split(file_full_path)[-1]
        for ch in self.channels:
            if '.'+ch+'.' in filename:
                return ch

    def data_simulation(self, gt, ch='A'):
        """ from gt generate wf image and sim image
        input:
            gt: the hr image
            ch: channel name
        ouput:
            lr_ch: low resolution image with lr_sim and lr_wf
            hr_ch: the normatlized hr image
        """

        # generate Wide field (WF) image
        otf = Cal_Otf(dx=gt.shape[-1], rand=True, ch=ch)
        gt = np.float32(gt)
        # gt_min, gt_max = gt.min(), gt.max()
        low_wf = ForwardOTF(gt, otf)


        ## normalize
        low_wf = (low_wf - low_wf.min()) / (low_wf.max() - low_wf.min())
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        std_base = {'A': 3000 }
        noise_level = np.array([(np.random.rand() * 4000) + 10000]) / 65535.
        noise_std = ((np.random.rand() * 500) + std_base[ch]) / 65535.  # 250./65535.

        #
        bg = np.random.randn(low_wf.shape[1],low_wf.shape[2])*noise_std+noise_level
        low_wf += bg
        low_wf = (low_wf - low_wf.min()) / (low_wf.max() - low_wf.min())    ## normalize make low_wf to [0, 1]

        # hr_ch = np.expand_dims(gt, axis=0)
        return low_wf[0,:], gt

    def get_generator(self):
        cpt_file = './model_weights/shannelCell_net_G.pth'

        state_dict = torch.load(cpt_file, weights_only=True)
        generator = UnetGenerator(1, 1, 8, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=True)
        generator.load_state_dict(state_dict)
        generator.eval()  # set generator to evaluation mode

        ## turn on dropout layer
        generator.model.model[1].model[3].model[3].model[3].model[7].train()
        generator.model.model[1].model[3].model[3].model[3].model[3].model[7].train()
        # generator.model.model[1].model[3].model[3].model[3].model[3].model[3].model[7].train()

        return generator

    def __getitem__(self, i) -> (torch.Tensor, torch.Tensor):
        GT_imgfile = self.GT_files[i % len(self.GT_files)]
        # lr_imgfile = GT_imgfile.replace('_gt.mrc', f'_level_{self.Level:02}.mrc')

        slice_k = np.random.randint(self.Size[-1])
        GT_img = sitk.GetArrayFromImage( sitk.ReadImage( GT_imgfile ) )

        ## get a slice
        GT_img = GT_img[slice_k, :, :]
        ## normalize to [0, 1]
        hr = (GT_img - GT_img.min()) / (GT_img.max() - GT_img.min())
        if self.img_size != self.Size[-1]:
            hr = resize( hr, (self.img_size, self.img_size) )
        lr, hr = self.data_simulation( hr )

        if self.genMethod == 'OTF':
            ## none process
            lr = lr

        elif self.genMethod.upper() == 'GAN':
            lr_img = generator_image(self.generator, hr, self.transform)
            ## normalize to [0, 1]
            lr = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min())

        elif self.genMethod.upper() == 'MIX':
            lr_img = generator_image(self.generator, hr, self.transform)
            ## normalize to [0, 1]
            lr_gan = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min())

            weight = np.random.uniform(0.7, 0.9)
            lr = weight * lr + (1 - weight) * lr_gan

        elif self.genMethod.upper() == 'COMB':
            probability = np.random.uniform(0, 1)
            if probability > 0.2:
                lr = lr
            else:
                lr_img = generator_image(self.generator, hr, self.transform)
                ## normalize to [0, 1]
                lr = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min())

        return np.expand_dims(lr, axis=0), np.expand_dims(hr, axis=0)



def generator_image(net, img, transform, device='cpu'):
    ## normalize the inputs
    img = np.float32(img)
    MIN, MAX = img.min(), img.max()
    input = (img - MIN) / (MAX - MIN) * 255
    input_tensor = transform(Image.fromarray(input))
    input_tensor = torch.unsqueeze(input_tensor, 0)

    with torch.no_grad():
        pred = net(input_tensor.to(device))

    pred = pred.cpu().detach().float().numpy()  # convert it into a numpy array
    pred = np.squeeze(pred)
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    lr_img = pred * (MAX - MIN) + MIN
    return lr_img
     


if __name__ == '__main__':
    hr_dir = r'D:\data\SELMA3DwithAnno\SELMA3D_training_annotated\shannel_cells\raw'
    data_sim_wf = SelmaDataset(data_dir=hr_dir, genMethod='MIX')

    # for i in range(100):
    #     lr, hr = data_sim_wf.__getitem__(10)
    #     print(lr.min(), lr.max(), hr.min(),hr.max())

    lr, hr = data_sim_wf.__getitem__(200)
    print(lr.min(), lr.max(), hr.min(), hr.max())

    from matplotlib import  pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow( lr[0,:100,:100], cmap='gray' )
    axs[1].imshow(hr[0,:100,:100], cmap='gray')
    plt.show()

    # noise_level = np.array([(np.random.rand() * 4000) + 2000]) / 65535.
    # noise_std = ((np.random.rand() * 400) + 200) / 65535.  # 250./65535.
    # print( noise_level, noise_std )

