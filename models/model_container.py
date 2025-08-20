import glob, tqdm, tifffile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM
# from models.trcan import LARN_noET, LARN
from models.rcan import RCAN
# from models.DNBSRN import DNBSRN
from models.model_base import BaseModel
from models.dfcan import DFCAN
from models.srcnn import SRCNN
from models.srkan import SRKAN
from models.rkan import RKAN
from models.ukan import U_KAN
from models.unet import SR_Unet
from models.RCKAN import RCKAN
from models.rckanp import RCKANP
from models.msaukan import MSA_UKAN
from models.attenukan import AttenUKAN
from models.attenmsaukan import AttenMSA_UKAN
from models.attenmbaukan3layer import AttenMSA_UKAN3
# import intel_extension_for_pytorch as ipex

"""RCAN model with 1C input"""
class modelContainer(BaseModel):
    """RCAN Model"""
    def __init__(self, exp_name="test", exp_dir = '../exp/', modelname = 'RCAN', device="cuda", n_groups=5, n_blocks=3, n_feats=48, loss='SmoothL1',
                 in_channel = 1, n_class=1, num_ARBs=5, lr = 1e-4):
        # initialize the model Container when the model Container is called then
        # the __init__ would be called for initialize the attributes of the instance.
        super().__init__(exp_name=exp_name,exp_dir=exp_dir, device=device)
        # initialize the the attributes of the parent class
        # model
        if modelname == 'RCAN' or modelname=='rcan':
            self.model = RCAN(n_channels=in_channel, n_classes=n_class, n_resblocks=n_blocks, n_resgroups=n_groups, n_feats=n_feats)
        # elif modelname == 'DNBSRN':
        #     self.model = DNBSRN( n_channels=in_channel, num_branches=3, n_feat=48, kernel_size=7 )
        elif modelname == 'DFCAN':
            self.model = DFCAN( n_channels=in_channel, n_classes= n_class )
        elif modelname == 'SRCNN':
            self.model = SRCNN( n_channels=in_channel, n_classes= n_class )
        elif modelname == 'SRKAN':
            self.model = SRKAN( n_channels=in_channel, n_classes= n_class )
        elif modelname == 'RKAN':
            self.model = RKAN( n_channels=in_channel, n_classes= n_class )
        elif modelname == 'UNET':
            self.model = SR_Unet( n_channels=in_channel, n_classes= n_class )
        elif modelname == 'UKAN':
            self.model = U_KAN(in_channels=in_channel).to(self.device)
        elif modelname == 'RCKAN':
            self.model = RCKAN(n_channels=in_channel, n_classes=n_class).to(self.device)
        elif modelname == 'RCKANP':
            self.model = RCKANP(n_channels=in_channel, n_classes=n_class).to(self.device)
        elif modelname == 'MSA_UKAN':
            self.model = MSA_UKAN(in_channels=in_channel).to(self.device)
        elif modelname == 'AttenUKAN':
            self.model = AttenUKAN(in_channels=in_channel).to(self.device)
        elif modelname == 'AttenMSA_UKAN':
            self.model = AttenMSA_UKAN(in_channels=in_channel).to(self.device)
        elif modelname == 'AttenMSA_UKAN3':
            self.model = AttenMSA_UKAN3(in_channels=in_channel).to(self.device)
        # elif modelname == 'LARN1en':
        #     self.model = LARN_noET(n_channels=in_channel, n_classes=n_class, num_encoder=1, num_ARBs=num_ARBs)
        # elif modelname == 'LARN2en':
        #     self.model = LARN_noET(n_channels=in_channel, n_classes=n_class, num_encoder=2)
        # elif modelname == 'LARN3en':
        #     self.model = LARN_noET(n_channels=in_channel, n_classes=n_class, num_encoder=3)
        # elif modelname == 'LARN_ET':
        #     self.model = LARN( n_channels=in_channel, n_classes=n_class, n_feats = 32, num_encoder=1 )
        # default loss functions
        self.loss = loss
        # Define the loss according to the passed argumentation
        if loss=="SmoothL1":
            self.loss_function = torch.nn.SmoothL1Loss()
            self.loss_function.to(self.device)
        elif loss=="L2":
            self.loss_function = torch.nn.MSELoss()
            self.loss_function.to(self.device)
        elif loss == 'MIX':
            self.weight = 0.84
            self.L1_loss = torch.nn.SmoothL1Loss()
            self.MSSSIM = MS_SSIM(data_range=1.75, size_average=True, channel=1)

            self.L1_loss.to(self.device)
            self.MSSSIM.to(self.device)
        else:
            Exception("Unknown loss functions!")
        # default optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.lr)
        # device        
        self.model.to(self.device) 
        # self.loss_function.to(self.device)
        # is 1C
        self.is1C = True

    # def train_ds(self, trainset, validationset, batch_size=16, nepoch=300, lr=1e-4, is_eval=False):
    #     # default maximum epochs
    #     valid_loss_min = 1e6
    #     # default scheduler
    #     for param_group in self.optimizer.param_groups:
    #         param_group["lr"] = lr
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    #     # dataloader
    #     # train_sampler, val_sampler,_,_ = BaseModel.split_trainset(trainset)
    #     train_loader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, num_workers=batch_size, shuffle=True)
    #     val_loader = DataLoader(validationset, batch_size=batch_size, pin_memory=True, num_workers=batch_size, shuffle=True)
    #     # epoch loop
    #     for epoch in range(0, nepoch):
    #         # training
    #         epoch_loss = self.train_epoch(train_loader)
    #         # validation
    #         valid_loss = self.valid_epoch(val_loader)
    #         scheduler.step(valid_loss)
    #         # eval
    #         if is_eval:
    #             mapping = self.eval_dataset(trainset)
    #         else:
    #             mapping = 0
    #         # logging and saving
    #         self.logger.info('Epochs: [%d/%d]; Train loss: %0.6f; Valid loss: %0.6f; mapping: %.6f; lr: %.6f \n'% (epoch, nepoch, epoch_loss, valid_loss, mapping, self.optimizer.param_groups[0]['lr']))
    #         if valid_loss < valid_loss_min:
    #             valid_loss_min = valid_loss
    #             self.save_model(type='rcan5g3b')
    #         if self.optimizer.param_groups[0]['lr']<1e-6:
    #             break
    #     return self.logger

    def train_epoch(self, train_loader, SWriter = None):
        self.model.train()
        epoch_loss = 0



        # training
        batch_tqdm = tqdm.tqdm(train_loader, total=len(train_loader))
        for batch in batch_tqdm:
            low, gt = batch[0], batch[1]

            low = low.to(device=self.device, dtype=torch.float32, non_blocking=True)
            gt = gt.to(device=self.device, dtype=torch.float32, non_blocking=True)



            pred = self.model(low)
            if self.loss != 'MIX':
                loss = self.loss_function(pred, gt)
            else:
                loss = self.weight * (1-self.MSSSIM(pred, gt)) + (1 - self.weight) * self.L1_loss(pred, gt)

            epoch_loss += loss.item()
            # backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            # tqdm
            batch_tqdm.set_description(f"Train: ")
            batch_tqdm.set_postfix(loss=loss.item())
        epoch_loss = epoch_loss/len(train_loader)
        if SWriter is not None:
            return epoch_loss, [low, gt, pred]
        return epoch_loss
    
    def valid_epoch(self, val_loader):
        self.model.eval()
        valid_loss = 0
        batch_tqdm = tqdm.tqdm(val_loader, total=len(val_loader))
        with torch.no_grad():
            for batch in batch_tqdm:
                low, gt = batch[0], batch[1]
                # low, gt = self.transforms_post(low, gt, self.is1C)
                low = low.to(device=self.device, dtype=torch.float32, non_blocking=True)
                gt = gt.to(device=self.device, dtype=torch.float32, non_blocking=True)
                with torch.no_grad():
                    pred = self.model(low)
                    if self.loss != 'MIX':
                        loss = self.loss_function(pred, gt)
                    else:
                        loss = self.weight * (1-self.MSSSIM(pred, gt)) + (1 - self.weight) * self.L1_loss(pred, gt)
                valid_loss += loss.data.item()
                # tqdm
                batch_tqdm.set_description(f"Valid: ")                
                batch_tqdm.set_postfix(loss=loss.item())
            valid_loss = valid_loss/len(val_loader)
        return valid_loss
        
    # def eval_dataset(self, predset):
    #     from . import data_wf as mydata2
    #     predset = mydata2.WFDataset(lr_dir="C:/Data/92BR_Reg")
    #     save_dir = "C:/Data/92BR_%s"%self.exp_name
    #     #
    #     pred_loader = DataLoader(predset, batch_size=1, pin_memory=True, num_workers=0)
    #     batch_tqdm = tqdm.tqdm(pred_loader, total=len(pred_loader))
    #     self.model.eval()
    #     for batch in batch_tqdm:
    #         batch_tqdm.set_description(f"Eval: ")
    #         low, gt = predset.transforms_post4C(batch[0], batch[1])
    #         with torch.no_grad():
    #             pred = self.pred_crop(low, self.model, self.device)
    #             imgsave = pred.numpy()*10000.
    #         imgsave[imgsave<0]=0
    #         imgsave[imgsave>65535]=65535
    #         #
    #         cyc = batch[2]
    #         wfpath = os.path.join(predset.lr_dir,cyc[0])
    #         srpath = os.path.join(save_dir,cyc[0])
    #         if not os.path.exists(srpath):
    #             os.makedirs(srpath)
    #         channels = ["A","C","G","T"]
    #         for ch in range(len(channels)):
    #             ch_files = glob.glob(os.path.join(wfpath,"*"+channels[ch]+".*.tif"))
    #             tifffile.imwrite(os.path.join(srpath,os.path.basename(ch_files[0])),np.uint16(imgsave[0,ch,:,:]).squeeze())
    #     client_path = 'C:/BGIwork/20220510_Basecall/client.exe'
    #     img_path = "C:/Data/92BR_%s"%self.exp_name
    #     cmd_basecall='start /wait cmd /c %s %s 30 52 53 -S -N DP88_%s'%(client_path,img_path, self.exp_name)
    #     os.system(cmd_basecall)
    #     mapping_file = 'C:/BGIwork/20220600_SimuData/20221000_DeepSeq/Exp_20221000/OutputFq/DP88_%s/L01/summaryTable.csv'%self.exp_name
    #     if os.path.exists(mapping_file):
    #         scores = pd.read_csv(mapping_file)
    #         scores = scores[scores['Category']=='MappingRate(%)']
    #         if  scores.size >0:
    #             mapping = float(scores[scores['Category']=='MappingRate(%)'].iloc[0,1])
    #         else:
    #             mapping = 0
    #     else:
    #         mapping = 0
    #     return mapping
        
    def pred_folder(self, data_dir, target_dir):
        cycs = [folder for folder in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir,folder))]
        for cyc in cycs:
            img_path = os.path.join(data_dir, cyc)
            imgfiles = glob.glob(os.path.join(img_path,"*.tif"))        
            for imgfile in imgfiles:
                img = tifffile.imread(imgfile)               
                img = self.transforms_predict(img) 
                with torch.no_grad():
                    pred = self.model(img.to(self.device))
                savepath = os.path.join(target_dir,cyc)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                tifffile.imwrite(os.path.join(savepath,os.path.basename(imgfile)),np.int16(pred.cpu().numpy()*5000))
            
    def pred_simfolder(self, data_dir, target_dir, wf_dir=None):
        cycs = [folder for folder in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir,folder))]
        channels = ["A.","C.","G.","T."]   
        for cyc in tqdm.tqdm(cycs):
            img_path = os.path.join(data_dir, cyc)
            for ch in range(len(channels)):
                ch_files = glob.glob(os.path.join(img_path,"*."+channels[ch]+"*.tif"))
                for cc in range(len(ch_files)):
                    img = tifffile.imread(ch_files[cc])
                    img = np.expand_dims(img,axis=0)
                    if cc==0:
                        imgs = img
                    else:
                        imgs = np.concatenate((imgs,img),axis=0)
                # loop channel files
                imgave = imgs.mean(axis=0)
                imgave = np.expand_dims(imgave,axis=0)
                if ch == 0:
                    imgch = imgave
                    files_ch = [ch_files[0]]
                else:
                    imgch = np.concatenate((imgch,imgave),axis=0)
                    files_ch = files_ch + [ch_files[0]]
            # predict
            imgin = self.transforms_predict(imgch)  #normalize the imgch with the 1th and 99th percentil
            pred = self.pred_crop(imgin, self.model, self.device)            
            srimg = pred.cpu().numpy()
            srimg = self.img_norm_maxmin(srimg)
            # srimg = img_norm_percentile(img)
            srimg = srimg*65535.        
            # save images
            for ch in range(len(channels)):
                # save wide field images
                if wf_dir is not None:
                    wfpath = os.path.join(wf_dir,cyc)
                    if not os.path.exists(wfpath):
                        os.makedirs(wfpath)
                    wfimg = imgin
                    wfimg[wfimg>65535]=65535
                    tifffile.imwrite(os.path.join(wfpath,os.path.basename(files_ch[ch])),np.uint16(wfimg[ch,:,:]))
                # save predicted super resolution images                
                savepath = os.path.join(target_dir,cyc)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                tifffile.imwrite(os.path.join(savepath,os.path.basename(files_ch[ch])),np.uint16(srimg[ch,:,:]))
  
    def transforms_predict(self,img):
        self.p99 = np.percentile(img,99)

        # self.MIN = np.min(img)
        self.MIN = np.percentile(img, 1)
        img[img<self.MIN] = self.MIN
        img[img>self.p99] = self.p99  ## test if this improves

        img_norm =(img-self.MIN)/(self.p99-self.MIN)

        # img_norm = img / self.p99
        #img_norm = np.minimum(np.maximum(img_norm,0),1)
        #img_norm = np.expand_dims(img_norm,axis=0)
        return torch.from_numpy(img_norm).to(torch.float32)

    def transforms_std_predict(self,img):
        ##cut the window
        self.p99 = np.percentile(img, 99)
        self.MIN = np.percentile(img, 1)
        img[img < self.MIN] = self.MIN
        img[img > self.p99] = self.p99
        img = img -self.MIN

        self.mean = img.mean()
        self.std = img.std()

        img_norm =(img-self.mean)/(self.std)

        # img_norm = img / self.p99
        #img_norm = np.minimum(np.maximum(img_norm,0),1)
        #img_norm = np.expand_dims(img_norm,axis=0)
        return torch.from_numpy(img_norm).to(torch.float32)
    
    # image normalization with maximum and minimum
    def img_norm_maxmin(self, img):
        img = img.astype(np.float32)
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        return img
    
if __name__ == "__main__":
    import os
    data_path = '/home/share/huadjyin/home/zhaizhiwei/project/SingleFrameModel/'
    model_name = "RCAN1C1F_MPT1s2NA12"
    # model_file = "../"+model_name+"/bestloss.pth"
    model_file = data_path + model_name + "/bestloss.pth"
    model = RCAN(exp_name=model_name, device="cuda:0")
    # model = RCAN1Cmodel(exp_name=model_name, device="cpu")
    model.load(model_file, type='rcan5g3b')
    datapre = 'P480_133BLT1s2crop'
    # sr_dir = os.path.join('../', datapre+'_'+model_name+'_Generated' )
    # model.pred_simfolder(data_dir=os.path.join('../', datapre), target_dir=sr_dir, wf_dir=None)
    sr_dir = os.path.join(data_path, datapre + '_' + model_name + '_Generated')
    model.pred_simfolder(data_dir=os.path.join(data_path,datapre), target_dir=sr_dir, wf_dir=None)
