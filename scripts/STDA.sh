python3 STDA_nosweep.py --gpu_id 1 --s 2 --t 0 1 3 --dset office-home --net resnet50 --max_epoch 100 --interval 100 --batch_size 64 --lr 1e-2 --cls_par 0.3 --const_par 0.3 --fbnm_par 3 --suffix Finetuned

python3 STDA_nosweep.py --gpu_id 1 --s 3 --t 0 1 2 --dset office-home --net resnet50 --max_epoch 100 --interval 100 --batch_size 64 --lr 1e-2 --cls_par 0.3 --const_par 0.3 --fbnm_par 3 --suffix Finetuned

# python3 STDA.py --gpu_id '0,1' --s 1 --t 2  --dset office --net resnet50 --max_epoch 100 --interval 100 --batch_size 128 --lr 1e-2
# python3 STDA.py --gpu_id '0,1' --s 2 --t 1 --dset office --net resnet50 --max_epoch 100 --interval 100 --batch_size 128 --lr 1e-2
# python3 STDA.py --gpu_id '0,1' --s 3 --t 0 1 2 --dset office-home --net resnet50 --max_epoch 50 --interval 50 --batch_size 64