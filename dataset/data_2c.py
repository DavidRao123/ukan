import os, glob
import numpy as np

import tifffile
import random
from utils.util import ForwardOTF, Cal_Otf, Add_SIM, Cal_SIM
import torch
from torch.utils.data import Dataset

class SimuDataset(Dataset):
    """
    The ApoC4Dataset reads 4C Seq hig-res simulation images
    Generate low-res and his-res pair using Apo filtering
    Simulation type 1, bg noise not regarded
    """
    def __init__(self, hr_dir:str="./hr", patch_size:int=256, npatch_per_image:int=32):
        self.hr_dir = hr_dir
        self.ids = [folder for folder in os.listdir(self.hr_dir) 
                    if os.path.isdir(os.path.join(hr_dir,folder))]
        self.img_size = patch_size
        self.npatch_per_image = npatch_per_image
        # self.channels = ["A","C","G","T"]
        self.channels = ["H", "L"]

    def __len__(self):
        return len( self.ids )*self.npatch_per_image

    def get_ch(self, file_full_path):
        filename = os.path.split(file_full_path)[-1]
        for ch in self.channels:
            if '.'+ch+'.' in filename:
                return ch

    def __getitem__(self, i) -> (torch.Tensor,torch.Tensor):
        hr_path = os.path.join(self.hr_dir, self.ids[i%len(self.ids)])

        y,x = np.random.randint(0,2200.-self.img_size-1),np.random.randint(0,2200.-self.img_size-1)

        hr_imgfiles = glob.glob(os.path.join(hr_path,"*.tif"))
        hr_imgfile = random.choice( hr_imgfiles )
        hr_ch = tifffile.imread( hr_imgfile )
        hr_ch = hr_ch[y:y+self.img_size,x:x+self.img_size]

        ch = self.get_ch( hr_imgfile )
        lr, hr = self.data_simulation(gt=hr_ch, ch=ch)
        return  lr, hr
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
        # generate Wide field (WF) image
        otf = Cal_Otf(dx=self.img_size, rand=True, ch=ch)
        low_wf = ForwardOTF(gt, otf)

        ##normalized
        low_wf = low_wf/low_wf.max()

        ## add noise to WF image
        noise_level = np.array([(np.random.rand()*4000)+2000])/65535.
        noise_std = ((np.random.rand()*400)+200)/65535. #250./65535.
        bg = np.random.randn(low_wf.shape[1],low_wf.shape[2])*noise_std+noise_level
        low_wf += bg

        # generate Structure Illumination to gt
        sim = Cal_SIM(dx=self.img_size, ch=ch )
        otf = Cal_Otf(dx=self.img_size, rand=True, ch=ch)
        low_sim = Add_SIM(gt, sim, otf)
        low_sim = low_sim / low_sim.max()

        ## add noise to SIM image
        noise_level = np.array([(np.random.rand() * 4000) + 2000]) / 65535.
        noise_std = ((np.random.rand() * 400) + 200) / 65535.  # 250./65535.
        bg = np.random.randn(low_wf.shape[1], low_wf.shape[2]) * noise_std + noise_level
        low_sim += bg

        gt = gt / gt.max()
        hr_ch = np.expand_dims(gt, axis=0)
        lr_ch = np.concatenate( (low_sim, low_wf), axis=0)
        return lr_ch, hr_ch  
        
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
    hr_dir = r'D:\project\data\Simulated_data\1.39NA_GT'
    data_sim_wf = SimuDataset(hr_dir=hr_dir)
    lr, hr = data_sim_wf.__getitem__(10)
    print(lr.shape, hr.shape)

