#!/bin/sh

CUDA_VISIBLE_DEVICES='0'

cd ..
# python main_gpt4_baseline.py \
#     --mode eval \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth \
#     --split val_seen\
#     --output_dir results/gpt-4o/\

python main_gpt4_baseline.py \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --eval_max_timestep 20 \
    --gsam_use_map_cache \
    --checkpoint checkpoints/data/mgp_mturk.pth \
    --split val_seen\
    --output_dir results/qwen/\