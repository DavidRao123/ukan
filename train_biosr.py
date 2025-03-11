import os
from dataset.data_bioSR import BioSRDataset
from models.model_container import modelContainer
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_main( args):
    # record the time of the experiment
    today = datetime.today().strftime('%y%m%d')
    # record some arguments of the experiments
    exp_name = f'{args.exp_name}_{today}_lr{args.lr}_bs{args.batch_size}_Ps{args.patch_size}_Npatch{args.Npatch_per_image}_N{args.Num_epoch}'

    # record the dir of the log
    log_dir = f'{args.logdir}/{exp_name}'

    os.makedirs( os.path.dirname(log_dir), exist_ok=True )
    # os.path.dirname(log_dir) get the dir of the log_dir
    # os.makedirs create a dir for log_dir
    # exist_ok = True make sure that there will not be error when the dir exists.
    swriter = SummaryWriter( log_dir= log_dir )
    # used to write the log of the experiments.

    modelCont = modelContainer(exp_name=exp_name, exp_dir=log_dir, modelname=args.model_name,
                               device=device, n_groups=args.n_groups, n_blocks=args.n_blocks, num_ARBs= args.num_ARBs,
                          n_feats=args.n_feats, loss=args.loss, in_channel=args.in_channel, n_class=args.n_class)
    # create a  model container by passing the arguments or using the default one.
    # default maximum epochs
    valid_loss_min = 1e6
    # default scheduler
    for param_group in modelCont.optimizer.param_groups:
        param_group["lr"] = args.lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(modelCont.optimizer, mode='min', factor=0.3, patience=5,
                                                           verbose=True)
    # train / validation dateset
    trainset = BioSRDataset( data_dir=args.train_dir,patch_size=args.patch_size, npatch_per_image=args.Npatch_per_image, Level=args.Level, genMethod=args.genMethod )
    validationset = BioSRDataset( data_dir=args.val_dir,patch_size=args.patch_size, npatch_per_image=args.Npatch_per_image, Level=args.Level, genMethod=args.genMethod )
    # trainset = SimuDataset( hr_dir=args.train_hr_dir, patch_size=args.patch_size, npatch_per_image=args.Npatch_per_image, simulationMethod=args.simulationMethod )
    # validationset = SimuDataset( hr_dir=args.val_hr_dir, patch_size=args.patch_size, npatch_per_image=args.Npatch_per_image, simulationMethod=args.simulationMethod )

    # dataloader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, num_workers=8, shuffle=True)
    val_loader = DataLoader(validationset, batch_size=args.batch_size, pin_memory=True, num_workers=8, shuffle=True)
    # epoch loop
    for epoch in range(0, args.Num_epoch):
        # training
        if args.WriteImageTensor:
            epoch_loss, images = modelCont.train_epoch(train_loader, SWriter=True)
            swriter.add_image('low', images[0][0,:], epoch)
            swriter.add_image('GT', images[1][0, :], epoch)
            swriter.add_image('pred', images[2][0, :], epoch)
            swriter.add_image('DIFF', torch.abs(images[1][0, :] - images[2][0, :]), epoch)
        else:
            epoch_loss = modelCont.train_epoch(train_loader)
        # validation
        valid_loss = modelCont.valid_epoch(val_loader)
        ## when lr less than 1e-6, don't adjust the lr
        if modelCont.optimizer.param_groups[0]['lr'] > 1e-6:
            scheduler.step( valid_loss )

        # logging and saving
        swriter.add_scalar( '{}/train'.format(args.loss), epoch_loss, epoch )
        swriter.add_scalar( '{}/val'.format(args.loss), valid_loss, epoch )
        swriter.add_scalar( 'lr', modelCont.optimizer.param_groups[0]['lr'],  epoch )
        modelCont.logger.info('Epochs: [%d/%d]; Train loss: %0.6f; Valid loss: %0.6f; mapping: %.6f; lr: %.6f \n' % (
        epoch, args.Num_epoch, epoch_loss, valid_loss, 0, modelCont.optimizer.param_groups[0]['lr']))

        ## save model every 5 epoch
        if (epoch+1) % args.Ninterval == 0:
            modelCont.save_model(type='rcan5g3b', epoch=epoch)

        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            modelCont.save_model(type='rcan5g3b')

## train_biosr.py --train_dir D:/zzhai/data/BioSR/Microtubules/train/ --val_dir D:/zzhai/data/BioSR/Microtubules/val/ --batch_size 1 --exp_name biosr_tmp
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser(description='Training model rcan for image SR')
    parser.add_argument('--train_dir', default='./data/train_dir')
    parser.add_argument('--val_dir', default='./data/val_dir')
    parser.add_argument('--Level', default=1, type=int)

    parser.add_argument('--exp_name', default='bioSR', type=str)
    parser.add_argument('--logdir', default='./logs')
    parser.add_argument('--Num_epoch', default=100, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_ARBs', default=5, type=int)
    parser.add_argument('--Ninterval', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--patch_size', default=256, type=int, help="patch size")
    parser.add_argument('--Npatch_per_image', default=128, type=int, help="number of patch per image")
    parser.add_argument('--in_channel', default=1, type=int, help="number of input channel")
    parser.add_argument('--n_class', default=1, type=int, help="number of output channel")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--loss', default='SmoothL1', choices=['SmoothL1', 'L2', 'MIX' ],
                        help='loss choice, SmoothL1 or L2, (default is SmoothL1)')
    parser.add_argument('--genMethod', default='None',
                        choices=['None', 'GAN', 'MIX', 'COMB'], help='data simultation method choice (default is OTF)')
    parser.add_argument('--model_name', default='RCAN', choices=['RCAN', 'DNBSRN', 'SRCNN', 'SRKAN' ],
                        help='model choice, RCAN or DNBSRN, (default is RCAN)')
    parser.add_argument('--WriteImageTensor', action='store_true')
    ## setting of model architecture
    parser.add_argument('--n_groups', default=5, type=int, help="number of groups")
    parser.add_argument('--n_blocks', default=3, type=int, help="number of blocks")
    parser.add_argument('--n_feats', default=48, type=int, help="number of features")

    args = parser.parse_args()
    args.exp_name = args.exp_name + '_model-' + args.model_name
    print( "Start training: ", args.exp_name )
    train_main(args)
