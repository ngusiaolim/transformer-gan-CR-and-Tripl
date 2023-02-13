#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

os.system(f"CUDA_VISIBLE_DEVICES=3 python test.py \
-gen_bs 32 \
-dis_bs 16 \
--dist-url 'tcp://localhost:14256' \
--dist-backend 'nccl' \
--multiprocessing_distributed \
--world-size 1 \
--rank {args.rank} \
--dataset stl10 \
--bottom_width 12 \
--img_size 48 \
--max_iter 500000 \
--gen_model ViT_custom_rp_tripl_attn \
--dis_model ViT_custom_scale2_rp_noise \
--df_dim 384 \
--g_norm pn \
--d_norm pn \
--d_heads 4 \
--d_depth 3 \
--g_depth 5,4,2 \
--dropout 0 \
--latent_dim 256 \
--gf_dim 1024 \
--num_workers 10 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 4 \
--num_eval_imgs 20000 \
--init_type kaiming_normal \
--n_critic 5 \
--val_freq 100000 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--D_downsample pixel \
--arch 1 0 1 1 1 0 0 1 1 1 0 1 0 3 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.05 \
--ema 0.9999 \
--diff_aug translation,cutout,color \
--load_path /public/home/dongsx/transgan/logs/stl_train_latent256_stl_erase_2022_12_24_17_13_08/Model/epoch200_checkpoint.pth \
--exp_name stl_train_latent256_stl_erase")
