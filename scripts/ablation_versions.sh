#!/bin/sh

CUDA_VISIBLE_DEVICES='0'

cd ..

python main_geonav.py\
    --mode eval\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --learning_rate 0.0015\
    --train_batch_size 12\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --output_dir results/geonav/medium/\
    --ablation full
# python main_geonav.py\
#     --mode eval\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --learning_rate 0.0015\
#     --train_batch_size 12\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/ablations/\
#     --ablation wo_cot

# python main_geonav.py\
#     --mode eval\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --learning_rate 0.0015\
#     --train_batch_size 12\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/ablations/\
#     --ablation wo_sg

# python main_geonav.py\
#     --mode eval\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --learning_rate 0.0015\
#     --train_batch_size 12\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/ablations/\
#     --ablation wo_landmark