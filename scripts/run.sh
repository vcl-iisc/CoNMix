# Office-Home DeiT-S
# src_train
python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 100 --interval 100 --batch_size 256 --s 0 --wandb 1 --output src_train --trte full
python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 256 --s 1 --wandb 1 --output src_train --trte full
python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 256 --s 2 --wandb 1 --output src_train --trte full
python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 256 --s 3 --wandb 1 --output src_train --trte full

# STDA
python3 STDA.py --gpu_id '2,1,0' --s 0 --t 1 2 3 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1
python3 STDA.py --gpu_id '2,1,0' --s 1 --t 0 2 3 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1
python3 STDA.py --gpu_id '2,1,0' --s 2 --t 0 1 3 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1
python3 STDA.py --gpu_id '2,1,0' --s 3 --t 0 1 2 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1

