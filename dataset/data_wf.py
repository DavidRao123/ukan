import os, glob, copy
import numpy as np
from torchvision import transforms
import tifffile
import random
from utils.util import ForwardOTF, Cal_Otf
import torch
from torch.utils.data import Dataset
from models.trcan import ESRT_noET
class SimuDataset(Dataset):
    """
    The SimuDataset reads 1C hig-res simulation images
    Generate low-res and his-res pair using OTF or GAN
    """
    def __init__(self, hr_dir:str="./hr", patch_size:int=256, npatch_per_image:int=32, simulationMethod:str='OTF'):
        self.hr_dir = hr_dir
        self.ids = [folder for folder in os.listdir(self.hr_dir) 
                    if os.path.isdir(os.path.join(hr_dir,folder))]
        self.img_size = patch_size
        self.npatch_per_image = npatch_per_image
        self.channels = ["A","C","G","T"]
        if not simulationMethod.upper() in ['GAN', 'OTF', 'MIX']:
            print( 'Error: the simulation method is not supported!' )

        self.simulationMethod = simulationMethod

        hr_imgfiles = glob.glob("{}/*/*.tif".format(hr_dir))
        hr_imgfile = random.choice(hr_imgfiles)
        hr_ch = tifffile.imread(hr_imgfile)
        self.Size = hr_ch.shape

        # if simulationMethod.upper() == 'GAN' or simulationMethod.upper() == 'MIX':
        #     self.net = ESRT_noET(n_channels=1, n_classes=1, n_feats=64, num_encoder=1)
        #     cpt_file = '../model_weights/ESRT_noET.pth'
        #     self.net.load_state_dict( torch.load(cpt_file) )
        #     self.device = torch.device("cpu")
        #     #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.net.to(self.device)
        #     self.net.eval()
        #
        #     transform_list = [transforms.ToTensor()]
        #     transform_list += [transforms.Normalize((0.5), (0.5))]
        #     self.transform = transforms.Compose(transform_list)



    def __len__(self):
        return len( self.ids )*self.npatch_per_image

    def get_ch(self, file_full_path):
        filename = os.path.split(file_full_path)[-1]
        for ch in self.channels:
            if '.'+ch+'.' in filename:
                return ch

    def __getitem__(self, i) -> (torch.Tensor,torch.Tensor):
        hr_path = os.path.join(self.hr_dir, self.ids[i%len(self.ids)])

        y,x = np.random.randint(0,self.Size[-1]-self.img_size-1),np.random.randint(0,self.Size[-2]-self.img_size-1)

        hr_imgfiles = glob.glob(os.path.join(hr_path,"*.tif"))
        hr_imgfile = random.choice( hr_imgfiles )
        hr_ch = tifffile.imread( hr_imgfile )



        ch = self.get_ch(hr_imgfile)
        if self.simulationMethod.upper() == 'OTF':
            lr, hr = self.data_simulation(gt=hr_ch, ch=ch)
            hr = hr[:, y:y + self.img_size, x:x + self.img_size]
            lr = lr[:, y:y + self.img_size, x:x + self.img_size]
        elif self.simulationMethod.upper() == 'GAN':
            ## write a function for data_simulation with GAN
            # hr_ch = hr_ch[ y:y + self.img_size, x:x + self.img_size ]
            lr, hr = self.data_simulation_gan( hr_ch )
            hr = hr[:, y:y + self.img_size, x:x + self.img_size]
            lr = lr[:, y:y + self.img_size, x:x + self.img_size]
        elif self.simulationMethod.upper() == 'MIX':

            ## using both otf_simulation and gan simulation; mixture image of
            lr_otf, hr_otf = self.data_simulation(gt=hr_ch[1,:,:], ch=ch)
            lr_gan, hr = self.data_simulation_gan( hr_ch )
            weight = np.random.uniform( 0.7, 0.9 )
            lr = weight*lr_otf + (1-weight)*lr_gan        ## merge lr_otf and lr gan with a weight

            hr = hr[:, y:y + self.img_size, x:x + self.img_size]
            lr = lr[:, y:y + self.img_size, x:x + self.img_size]
        elif self.simulationMethod.upper() == 'COMB':
            probability = np.random.uniform(0,1)
            if probability > 0.2:
                lr, hr = self.data_simulation(gt=hr_ch[1, :, :], ch=ch)
            else:
                lr, hr = self.data_simulation_gan(hr_ch)

            hr = hr[:, y:y + self.img_size, x:x + self.img_size]
            lr = lr[:, y:y + self.img_size, x:x + self.img_size]
        # ## cut with window
        # # lr_p1, lr_p99 = np.percentile(lr, 1),
        # ## standization
        # lr = (lr - lr.mean()) / lr.std()
        # hr = (hr - hr.mean()) / hr.std()
        return lr, hr
        # return torch.from_numpy(lr), torch.from_numpy(hr)
      
    def data_simulation(self, gt, ch):
        """ from gt generate wf image and sim image
        input:
            gt: the hr image
            ch: channel name
        ouput:
            lr_ch: low resolution image with lr_sim and lr_wf
            hr_ch: the normatlized hr image
        """
        ## cut hr_ch with p1 and p99
        range1 = np.random.uniform(0.9, 1.1)
        range2 = np.random.uniform(0.9, 1.1)
        p1, p99 = np.percentile(gt, 1)*range1, np.percentile(gt, 99)*range2
        gt[gt<p1] = p1
        gt[gt>p99] = p99

        # generate Wide field (WF) image
        otf = Cal_Otf(dx=self.Size[-1], rand=True, ch=ch)
        gt = np.float32(gt)
        low_wf = ForwardOTF(gt, otf)

        low_wf = (low_wf - low_wf.min()) / (low_wf.max() - low_wf.min())
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        ## add noise to WF image
        # noise_level = np.array([(np.random.rand()*4000)+2000])/65535.
        # noise_std = ((np.random.rand()*400)+200)/65535. #250./65535.

        std_base = {'A': 180,'C':190, 'G': 280, 'T': 190 }
        noise_level = np.array([(np.random.rand() * 4000) + 2000]) / 65535.
        noise_std = ((np.random.rand() * 200) + std_base[ch]) / 65535.  # 250./65535.
        #
        bg = np.random.randn(low_wf.shape[1],low_wf.shape[2])*noise_std+noise_level
        low_wf += bg
        low_wf = (low_wf - low_wf.min()) / (low_wf.max() - low_wf.min())    ## normalize make low_wf to [0, 1]

        # range = np.random.uniform( 0.9,1.1 )
        # low_wf = low_wf / (range*np.percentile(low_wf, 99))
        # gt = gt / (range*np.percentile(gt, 99))

        hr_ch = np.expand_dims(gt, axis=0)
        return low_wf, hr_ch

    def data_simulation_gan(self, low_gt):
        ## using GAN to simulate low resolution image
        low_wf = low_gt[:1,:,:]
        gt = low_gt[1:,:,:]

        ## cut hr_ch with p1 and p99
        # range1 = np.random.uniform(0.9, 1.1)
        # range2 = np.random.uniform(0.9, 1.1)
        p1_lr, p99_lr = np.percentile(low_wf, 1) * np.random.uniform(0.9, 1.1), np.percentile(low_wf, 99)*np.random.uniform(0.9, 1.1)
        gt[gt < p1_lr] = p1_lr
        gt[gt > p99_lr] = p99_lr

        p1_gt, p99_gt = np.percentile(gt, 1)*np.random.uniform(0.9, 1.1), np.percentile(gt, 99)*np.random.uniform(0.9, 1.1)
        gt[gt<p1_gt] = p1_gt
        gt[gt>p99_gt] = p99_gt

        # normalize
        low_wf = (low_wf - low_wf.min()) / (low_wf.max() - low_wf.min())
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        # range = np.random.uniform(0.9, 1.1)
        # low_wf = low_wf / (range * np.percentile(low_wf, 99))
        # gt = gt / (range * np.percentile(gt, 99))

        # hr_ch = np.expand_dims(gt, axis=0)
        return low_wf, gt

    @staticmethod
    def transforms_post4C(hr, lr):
        #Generate 1C dataset: 4C in, 4C out; default outputs
        return hr.to(torch.float32), lr.to(torch.float32)
    
    @staticmethod
    def transforms_predict(img):
        img = img*2200.
        img = np.expand_dims(img,axis=0)
        return torch.from_numpy(img).to(torch.float32)
     


if __name__ == '__main__':
    hr_dir = r'D:/zzhai/data/train_data_single/1Frame_260nm_cubic_gan'
    data_sim_wf = SimuDataset(hr_dir=hr_dir, simulationMethod='MIX')
    for i in range(100):
        lr, hr = data_sim_wf.__getitem__(10)
        print(lr.min(), lr.max(), hr.min(),hr.max())


    from matplotlib import  pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow( lr[0,:100,:100], cmap='gray' )
    axs[1].imshow(hr[0,:100,:100], cmap='gray')
    plt.show()

    # noise_level = np.array([(np.random.rand() * 4000) + 2000]) / 65535.
    # noise_std = ((np.random.rand() * 400) + 200) / 65535.  # 250./65535.
    # print( noise_level, noise_std )

