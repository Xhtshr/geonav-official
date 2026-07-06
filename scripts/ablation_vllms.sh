#!/bin/sh
CUDA_VISIBLE_DEVICES='0'

cd ..

python main_geonav.py\
    --mode eval\
    --split val_seen\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --output_dir results/geonav/local/qwen32b/\

# python main_geonav.py\
#     --mode eval\
#     --split val_seen\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/local/qwen7b/\
#     --deployment local

# python main_geonav.py\
#     --mode eval\
#     --split val_seen\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/local/llava-hf/\
#     --deployment local