import glob, os, time
import tifffile
from argparse import ArgumentParser
import torch, tqdm, logging
import numpy as np
from models.model_container import modelContainer
import pickle
from monai.inferers import SlidingWindowInferer
from utils.util import evaluation_metrics
from utils.read_mrc import write_mrc, read_mrc

def pred_evaluate(model, gt_file, lr_file, args):
    ## predict the lr image and evaluate against gt image

    header, gt_img = read_mrc(gt_file)
    header, lr_img = read_mrc(lr_file)

    Size = gt_img.shape
    MAEs, PSNRs, SSIMs = [], [], []

    model.model.to(model.device)
    model.model.eval()

    pred_imgs = []
    for k in range( Size[-1] ):
        img = lr_img[:,:,k]
        print("lr_img shape", img.shape)
        # img = model.transforms_predict(img)
        MIN, MAX = img.min(), img.max()
        img = (img - MIN) / (MAX - MIN)
        img = torch.from_numpy(img).to(torch.float32)

        img = torch.unsqueeze(img, dim=0)  # add channel dim
        img = torch.unsqueeze(img, dim=0)  # add batch dim

        with torch.no_grad():
           # stime = time.time()
            # pred = model.model(img.to(model.device)).cpu().numpy()
            pred = monai_sliding_window_inference(
                image_np=img.squeeze().cpu().numpy(),
                model=model.model,
                device=model.device,
                patch_size=(256, 256),
                overlap=0.25
            )
            pred = np.squeeze(pred)
            pred = pred * (MAX - MIN) + MIN

            pred_imgs.append(pred)
            mae, psnr, ssim = evaluation_metrics( gt_img[:,:,k], pred )

            MAEs.append(mae)
            PSNRs.append(psnr)
            SSIMs.append(ssim)
    if args.save_pred_image:
        ##the save predicted image
        pred_imgs = np.stack(pred_imgs, axis=-1).astype(np.uint16)
        path_list = os.path.split( lr_file )
        filename = path_list[-1]
        cell_type = os.path.split(path_list[0])[-1]
        save_dir = f'{args.save_dir}/{cell_type}'
        os.makedirs( save_dir, exist_ok=True )
        write_mrc( f"{save_dir}/pred_{filename}", pred_imgs, header )
        # print("to do")
    return MAEs, PSNRs, SSIMs



def monai_sliding_window_inference(image_np, model, device, patch_size=(256, 256), overlap=0.25):
    """
    image_np: 单张 numpy 2D 图像 (H, W)，如 502x502
    model: PyTorch 模型，已经加载到 device 上
    device: torch.device
    """
    model.eval()
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=1,
        overlap=overlap,
        mode='gaussian',
        padding_mode='reflect',
        progress=False,
        device=device
    )

    # 准备输入张量
    input_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = inferer(input_tensor, model)
    
    pred = pred.squeeze().cpu().numpy()
    return pred


def test_main( args):

    model = modelContainer(modelname=args.model_name, device=device, in_channel = 1, n_class=1, num_ARBs=args.num_ARBs)

    # load weights
    model_file = args.model_weights
    model.load(model_file, type='rcan5g3b')

    GT_files = glob.glob(f"{args.data_dir}/*/*Data_gt.mrc")
    MAEs_list, PSNRs_list, SSIMs_list = [], [],[]
    for gt_file in tqdm.tqdm(GT_files):
        lr_file = gt_file.replace( '_gt.mrc', f'_level_{ args.Level:02d}.mrc' )
        MAEs, PSNRs, SSIMs = pred_evaluate(model, gt_file, lr_file, args)
        MAEs_list.append(MAEs)
        PSNRs_list.append(PSNRs)
        SSIMs_list.append(SSIMs)

    # print('End inference:  ', args.save_dir)
    MAE_np = np.array(MAEs_list)
    PSNR_np = np.array(PSNRs_list)
    SSIM_np = np.array(SSIMs_list)
    result_dict = {'MAE': MAE_np,
                   'PSNR': PSNR_np,
                   'SSIM': SSIM_np
                   }
    print( f'MAE: {MAE_np.mean():.2f} {MAE_np.std():.2f}; PSNR: {PSNR_np.mean():.2f} {PSNR_np.std():.2f}; '
           f'SSIM: {SSIM_np.mean():.2f} {SSIM_np.std():.2f}' )
    save_file = f'{args.save_dir}_level{args.Level:02d}.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(result_dict, f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# python test_biosr.py --data_dir D:\zzhai\data\BioSR\CCPs/test --model_name RCAN --Level 1 --model_weights ./logs/biosr/RCAN_CCPs_Lmix_cutSigmoid_model-RCAN_240426_lr0.0001_bs16_Ps256_Npatch32_N50/bestloss.pth
if __name__ == '__main__':
    parser = ArgumentParser(description='Training model rcan for image SR')
    # parser.add_argument('--exp_name', default='RCAN', type=str, help="the name of experiment")
    parser.add_argument('--data_dir', default='./data/test_dir', type=str, help="the input directory")
    parser.add_argument('--model_name', default='RCAN', choices=['RCAN', 'DNBSRN', 'DFCAN', 'ESRT1en',
                                                                 'ESRT2en', 'ESRT3en', 'ESRT_ET','SRKAN', 'SRCNN',
                                                                 'RKAN', 'UNET', 'UKAN', 'RCKAN', 'RCKANP', 'MSA_UKAN',
                                                                  'AttenUKAN', 'AttenMSA_UKAN', 'AttenMSA_UKAN3'],
                        help='model choice, RCAN or DNBSRN, (default is RCAN)')
    # parser.add_argument('--Scanner', default='Single', type=str, help='The scanner type')
    parser.add_argument('--Level', default=1, type=int)
    parser.add_argument('--num_ARBs', default=5, type=int)
    parser.add_argument('--save_pred_image', action='store_true', help="save predicted image")
    parser.add_argument('--model_weights', default='./logs',type=str, help="model weights")
    parser.add_argument('--model_weights_dir', default=None, type=str, help="model weights")

    args = parser.parse_args()

    # save dir
    path_list = os.path.split(args.data_dir)
    args.save_dir = "{}/{}_{}".format(path_list[0], path_list[1], args.model_name)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_weights_dir is None:
        print("Start inference: ", args.model_name)
        test_main( args )
    else:
        weights = glob.glob(f'{args.model_weights_dir}/*.pth')
        for weight in weights:
            args.model_weights = weight
            print("Start inference: ", args.model_name, weight)
            test_main(args)
