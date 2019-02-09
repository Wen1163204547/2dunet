#CUDA_VISIBLE_DEVICES=2 python main.py --resume './ckpt/train_2d_0180.ckpt' --test 1 --save_dir './results/val/'
#CUDA_VISIBLE_DEVICES=1 python main.py --resume './ckpt/train_2d_0180.ckpt' --test 1 --save_dir './results/train/'
CUDA_VISIBLE_DEVICES=1 python main_3.py --resume './ckpt_s2/train_2d_0070.ckpt' --test 1 --save_dir './results_2/val/' --mask 'results/val'
