import os, glob, copy
import numpy as np
from utils.read_mrc import read_mrc
import random
import torch
from torch.utils.data import Dataset
# from models.gan_networks import UnetGenerator, get_norm_layer
from torchvision import transforms
from PIL import Image
class BioSRDataset(Dataset):
    """
    The SimuDataset reads 1C hig-res simulation images
    Generate low-res and his-res pair using OTF or GAN
    """
    def __init__(self, data_dir:str="./hr", patch_size:int=256, npatch_per_image:int=32, Level:int=1, genMethod:str='None'):
        self.data_dir = data_dir
        self.Level = Level
        self.img_size = patch_size
        self.npatch_per_image = npatch_per_image

        self.GT_files = glob.glob( f"{data_dir}/*/*Data_gt.mrc" )
        GT_imgfile = random.choice( self.GT_files )
        header, GT_image = read_mrc( GT_imgfile )
        self.Size = GT_image.shape
        self.genMethod = genMethod
        # if genMethod != 'None':
        #     self.generator = self.get_generator()
        #
        #     transform_list = [transforms.Grayscale(1),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5), (0.5))]
        #     self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len( self.GT_files )*self.npatch_per_image

    # def get_generator(self):
    #     if 'CCPs' in self.data_dir:
    #         cpt_file = '../model_weights/CCPs_net_G.pth'
    #     elif 'F-actin' in self.data_dir:
    #         cpt_file = '../model_weights/Factin_net_G.pth'
    #     elif 'Microtubules' in self.data_dir:
    #         cpt_file = '../model_weights/Microtubules_net_G.pth'
    #     else:
    #         print('Error')
    #         exit(code="")
    #     state_dict = torch.load(cpt_file)
    #     generator = UnetGenerator(1, 1, 8, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=True)
    #     generator.load_state_dict(state_dict)
    #     generator.eval()    #set generator to evaluation mode
    #
    #     # turn on dropout layer
    #     generator.model.model[1].model[3].model[3].model[3].model[7].train()
    #     generator.model.model[1].model[3].model[3].model[3].model[3].model[7].train()
    #     generator.model.model[1].model[3].model[3].model[3].model[3].model[3].model[7].train()
    #
    #     return generator

    def __getitem__(self, i) -> (torch.Tensor,torch.Tensor):
        GT_imgfile = self.GT_files[ i%len(self.GT_files) ]
        lr_imgfile = GT_imgfile.replace( '_gt.mrc', f'_level_{self.Level:02}.mrc' )


        slice_k = np.random.randint( self.Size[-1] )
        y,x = np.random.randint(0,self.Size[1]-self.img_size-1),np.random.randint(0,self.Size[0]-self.img_size-1)

        header, GT_img = read_mrc( GT_imgfile )
        header, lr_img = read_mrc( lr_imgfile )

        ## get a slice
        GT_img = GT_img[:, :, slice_k]
        ## normalize to [0, 1]
        GT_img = (GT_img - GT_img.min()) / (GT_img.max() - GT_img.min())
        # extract patch
        hr = GT_img[y:y + self.img_size, x:x + self.img_size]

        ## get a slice
        lr_img = lr_img[:, :, slice_k]
        ## normalize to [0, 1]
        lr_img = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min())
        # extract patch
        lr = lr_img[y:y + self.img_size, x:x + self.img_size]

        if self.genMethod == 'None':
            ## none process
            lr = lr

            # return np.expand_dims(lr, axis=0), np.expand_dims(hr, axis=0)
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

        return np.expand_dims(lr,axis=0), np.expand_dims(hr,axis=0)

    # @staticmethod
    # def transforms_predict(img):
    #     img = img*2200.
    #     img = np.expand_dims(img,axis=0)
    #     return torch.from_numpy(img).to(torch.float32)
def generator_image(net, img, transform, device='cpu' ):
    ## normalize the inputs
    img = np.float32(img)
    MIN, MAX = img.min(), img.max()
    input = (img - MIN) / (MAX - MIN) * 255
    input_tensor = transform( Image.fromarray(input) )
    input_tensor = torch.unsqueeze(input_tensor, 0)

    with torch.no_grad():
        pred = net(input_tensor.to(device))

    pred = pred.cpu().detach().float().numpy()  # convert it into a numpy array
    pred = np.squeeze(pred)
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    lr_img = pred * (MAX - MIN) + MIN
    return lr_img

if __name__ == '__main__':
    hr_dir = r'D:\zzhai\data\BioSR\Microtubules\train'
    dataset_biosr = BioSRDataset(data_dir=hr_dir, Level=1, genMethod='MIX')

    lr, hr = dataset_biosr.__getitem__(10)
    print(lr.shape, hr.shape)
    print(lr.min(), np.percentile(lr, 0.5), lr.max(), np.percentile(lr, 99.5))
    print( hr.min(), np.percentile(hr, 0.5), hr.max(), np.percentile(hr, 99.5),)

    from matplotlib import  pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow( lr[0,:100,:100], cmap='gray' )
    axs[1].imshow(hr[0,:100,:100], cmap='gray')

    # axs[0].hist(lr.reshape(-1))
    # axs[1].hist(hr.reshape(-1))
    plt.show()



