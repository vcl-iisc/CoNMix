python3 STDA.py --s 0 --t 1 2 3 --dset office-home --net deit_s --max_epoch 30 --interval 30 --batch_size 64
python3 STDA.py --s 1 --t 0 2 3 --dset office-home --net deit_s --max_epoch 30 --interval 30 --batch_size 64
python3 STDA.py --s 2 --t 0 1 3 --dset office-home --net deit_s --max_epoch 30 --interval 30 --batch_size 64
python3 STDA.py --s 3 --t 0 1 2 --dset office-home --net deit_s --max_epoch 30 --interval 30 --batch_size 64