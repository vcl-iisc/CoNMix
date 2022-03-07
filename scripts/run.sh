# Office-Home DeiT-S
# src_train
# python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 100 --interval 100 --batch_size 256 --s 0 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 256 --s 1 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 100 --interval 100 --batch_size 256 --s 2 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '2,1,0' --dset office-home --net deit_s --max_epoch 70 --interval 70 --batch_size 192 --s 3 --wandb 1 --output src_train --trte full

# STDA
# python3 STDA.py --gpu_id '2,1,0' --s 0 --t 1 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --sdlr 1 --lr 5e-3
# python3 STDA.py --gpu_id '2,1,0' --s 0 --t 2 3 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --suffix noSdlr --sdlr 0 --lr 5e-3

python3 STDA.py --gpu_id '2,1,0' --s 1 --t 0 2 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --sdlr 1 --lr 5e-3
python3 STDA.py --gpu_id '2,1,0' --s 1 --t 3 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --suffix noSdlr --sdlr 0 --lr 5e-3

# python3 STDA.py --gpu_id '2,1,0' --s 2 --t 0 1 3 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --sdlr 1 --lr 5e-3

# python3 STDA.py --gpu_id '2,1,0' --s 3 --t 0 1 2 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --sdlr 1 --lr 5e-3
# python3 STDA.py --gpu_id '2,1,0' --s 3 --t 2 --dset office-home --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1 --suffix noSdlr --sdlr 0 --lr 5e-3

# Office31 DeiT-S
# src_train
# python3 source_train.py --gpu_id '0' --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 80 --s 0 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '1' --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 80 --s 1 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '2' --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 80 --s 2 --wandb 1 --output src_train --trte full

# STDA
# python3 STDA.py --gpu_id '0' --s 0 --t 1 2 --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 64 --input_src src_train --wandb 1
# python3 STDA.py --gpu_id '1' --s 1 --t 0 2 --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 64 --input_src src_train --wandb 1
# python3 STDA.py --gpu_id '0' --s 2 --t 0 1 --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 64 --input_src src_train --wandb 1

# VisDA DeiT-B
# src_train
# python3 source_train.py --gpu_id '0,1' --dset visda-2017 --net deit_b --max_epoch 50 --interval 50 --batch_size 96 --s 0 --wandb 0 --output src_train --trte full --lr 1e-4 

# STDA
# python3 STDA.py --gpu_id '0' --s 0 --t 1 2 --dset office --net deit_s --max_epoch 50 --interval 50 --batch_size 64 --input_src src_train --wandb 1

# Office-Caltech DeiT-S
# src_train
# python3 source_train.py --gpu_id '0' --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --s 0 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '0' --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --s 1 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '0' --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --s 2 --wandb 1 --output src_train --trte full
# python3 source_train.py --gpu_id '0' --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --s 3 --wandb 1 --output src_train --trte full

# STDA
# python3 STDA.py --gpu_id '1' --s 0 --t 1 2 3 --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 64 --input_src src_train --wandb 1 --cls_par 1 --lr 1e-3
# python3 STDA.py --gpu_id '0' --s 1 --t 0 2 3 --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1
# python3 STDA.py --gpu_id '0' --s 2 --t 0 1 3 --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1
# python3 STDA.py --gpu_id '0' --s 3 --t 0 1 2 --dset office-caltech --net deit_s --max_epoch 50 --interval 50 --batch_size 128 --input_src src_train --wandb 1
