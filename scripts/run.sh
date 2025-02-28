#!/bin/sh

CUDA_VISIBLE_DEVICES='0'

cd ..

python main_goal_predictor.py \
    --mode eval \
    --model mgp \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --eval_batch_size 50 \
    --eval_max_timestep 15 \
    --gsam_use_map_cache \
    --checkpoint checkpoints/data/mgp_sp.pth

