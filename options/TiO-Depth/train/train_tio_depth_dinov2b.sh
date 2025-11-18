#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name TiO-Depth-DINOv2B_rc256_KITTI_S_B8\
 --exp_opts options/TiO-Depth/train/tio_depth-dinov2b_384crop256_kitti_stereo.yaml\
 --batch_size 1\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 5\
 --visual_freq 1000 