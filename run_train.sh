python train_biosr.py --train_dir ./data/CMF/train/ --val_dir ./data/CMF/val/ --Num_epoch 200 --Level 1 --batch_size 16 --loss SmoothL1 --model_name AttenMSA_UKAN --exp_name biosr_cm_attenmsaukan3_smoothl1_0806



python train_biosr.py --train_dir ./data/CCPs/train/ --val_dir ./data/CCPs/val/ --Num_epoch 100 --Level 1 --batch_size 4 --patch_size 256 --loss SmoothL1 --model_name UKAN --exp_name biosr_ccps_ukan_smoothl1_bs4_ps192_model-UKAN_250522_lr0.0001_bs4_Ps192_Npatch128_N100

