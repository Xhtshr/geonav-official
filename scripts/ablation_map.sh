#!/bin/sh

CUDA_VISIBLE_DEVICES='1'

cd ..
# TopV
python main_geonav.py \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --eval_max_timestep 20 \
    --eval_batch_size 50\
    --checkpoint checkpoints/data/mgp_mturk.pth \
    --split test_unseen \
    --output_dir results/ablations/TopNav/test_unseen/\
    --map_type TopV

# python main_geonav.py \
#     --mode eval \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth \
#     --split val_unseen \
#     --output_dir results/ablations/TopNav/val_unseen/\
#     --map_type TopV

# python main_geonav.py \
#     --mode eval \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth \
#     --split test_unseen \
#     --output_dir results/ablations/TopNav/test_unseen/\
#     --map_type TopV

# # stmr
# python main_geonav.py \
#     --mode eval \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_batch_size 50\
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth \
#     --split test_unseen \
#     --output_dir results/ablations/STMR/test_unseen/\
#     --map_type STMR

# python main_geonav.py \
#     --mode eval \
#     --model mgp \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth \
#     --split val_unseen \
#     --map_type STMR

# python main_geonav.py \
#     --mode eval \
#     --model mgp \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth \
#     --split test_unseen \
#     --map_type STMR