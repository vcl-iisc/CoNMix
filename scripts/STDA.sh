# python3 classwise_acc.py --gpu_id 0 --s 0 --t 2 --dset office-home --net deit_s --output STDA_nm_weights --output2 STDA_im_weights --max_epoch 50 --interval 50 --batch_size 8 --phase 'test'
# python3 STDA.py --gpu_id 0 --s 1 --t 0 2 3 --dset office-home --net resnet50 --max_epoch 50 --interval 50 --batch_size 256
# python3 STDA.py --gpu_id 1 --s 2 --t 0 1 3 --dset office-home --net resnet50 --max_epoch 50 --interval 50 --batch_size 64
# python3 STDA.py --gpu_id 1 --s 3 --t 0 1 2 --dset office-home --net resnet50 --max_epoch 50 --interval 50 --batch_size 64