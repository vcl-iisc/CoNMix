python3 source_train.py --gpu_id 0 --batch_size 2048 --dset office-home --s 2 --wandb 0 --output src_train
python3 STDA.py --gpu_id 0 --s 5 --t 2 --max_epoch 20 --interval 20 --lr 1e-5 --batch_size 56 --test_bs 1280 --dset office-home --net vit --wandb 1 --cls_par 0.6 --fbnm_par 7.0 --output_src src_train --output weights/new/STDA_wt_fbnm_rlccsoft --suffix fbnm_rlcc_soft --rlcc 1 --soft_pl 1
python3 bridge_MTDA.py --gpu_id 0 --s 3 --dset domain_net --net vit --output weights/STDA_wt_fbnm_rlccsoft --batch_size 2048
