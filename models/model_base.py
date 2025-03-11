import os, logging, time
from abc import abstractmethod

import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

class BaseModel():
    '''Model dir, device'''
    def __init__(self, exp_name="test", exp_dir = './exp/', device="cuda"):
        self.exp_name = exp_name
        # self.exp_dir = "../exp/"+exp_name+"/"
        self.exp_dir = exp_dir

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.logger = self.get_logger()
        #
        self.device = torch.device(device)
        if device == "cuda":
            self.logger.info("Using GPU: {}".format(torch.cuda.get_device_name(0)))
            torch.cuda.empty_cache()
    
    def get_logger(self):
        logger = logging.getLogger()     
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_dir, "log.txt"))
        sh = logging.StreamHandler()
        fa = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(fa)
        sh.setFormatter(fa)
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger
    
    def load(self, model_path=None, type='Undefined'):
        if model_path is None:
            checkpoint = torch.load(os.path.join(self.exp_dir, "bestloss.pth"), map_location=self.device)
        else:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'type' in checkpoint:
            if checkpoint['type'] != type:
                raise Exception('incorret model type')  
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("model loaded")
        
    def save_model(self, save_dir=None, type=None, epoch=None):
        checkpoint = {
            'type':type,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }
        if save_dir is None:
            save_dir = self.exp_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch is None:
            savepath = os.path.join(save_dir,'bestloss.pth')
        else:
            savepath = os.path.join(save_dir, 'model_{:03d}.pth'.format(epoch))
        torch.save(checkpoint,savepath)
        
    @staticmethod
    def pred_crop(low, model, device):
        # low: the input image
        # cal crop params
        inp=low.shape[-1]
        np_crop = 1024
        dp_crop0 = 128
        dp_crop1 = np.int32(np.round(dp_crop0/2))
        N = np.int32(np.ceil((inp-np_crop)/(np_crop-dp_crop0))+1)
        dp_crop = np.int32(np_crop-np.round((inp-np_crop)/(N-1)))
        cp1 = np.arange(0, (np_crop-dp_crop)*N-1, np_crop-dp_crop)
        cp1[-1]=inp-np_crop-1
        cp2 = cp1+np_crop
        dp_crop=cp2[0]-cp1[1]
        #
        pred = torch.zeros_like(low)
        with torch.no_grad():
            for crow in range(N):
                for ccol in range(N):  
                    for ch in range(low.shape[0]):                 
                        lowcrop = low[ch:ch+1,cp1[crow]:cp2[crow],cp1[ccol]:cp2[ccol]]
                        predcrop = model(lowcrop.to(device))
                        # modified montage
                        cpr1=cp1[crow]+dp_crop1 if crow>0 else cp1[crow]
                        cpr2=dp_crop1 if crow>0 else 0
                        cpc1=cp1[ccol]+dp_crop1 if ccol>0 else cp1[ccol]
                        cpc2=dp_crop1 if ccol>0 else 0 
                        pred[ch,cpr1:cp2[crow],cpc1:cp2[ccol]]=predcrop[0,cpr2:,cpc2:]
                        #pred[ch,cp1[crow]:cp2[crow],cp1[ccol]:cp2[ccol]]=predcrop
                        
        return pred.cpu()
    
    """general transform wrapper"""
    @staticmethod
    def transforms_post(hr, lr, is1C=True):
        if is1C:
            return BaseModel.transforms_post1C(hr, lr)
        else:
            return BaseModel.transforms_post4C(hr, lr)

    @staticmethod
    def transforms_post4C(hr, lr):
        #Generate 1C dataset: 4C in, 4C out; default outputs
        return hr.to(torch.float32), lr.to(torch.float32)

    @staticmethod
    def transforms_post1C(hr, lr):
        #Generate 1C dataset: 1C in, 1C out 
        hr=hr.reshape((hr.shape[0]*hr.shape[1],1,hr.shape[2],hr.shape[3]))
        lr=lr.reshape((lr.shape[0]*lr.shape[1],1,lr.shape[2],lr.shape[3]))
        return hr.to(torch.float32), lr.to(torch.float32)
    