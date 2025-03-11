#!/bin/sh

CUDA_VISIBLE_DEVICES='0'

cd ..

python geonav_ex.py \
    --mode eval \
    --model mgp \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --eval_max_timestep 20 \
    --gsam_use_map_cache \
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --split val_seen

# python geonav_ex.py \
#     --mode eval \
#     --model mgp \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --split val_unseen

# python geonav_ex.py \
#     --mode eval \
#     --model mgp \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --split test_unseen


# python onlymap_ex.py \
#     --mode eval \
#     --model mgp \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth
# python gpt4_ex.py \
#     --mode eval \
#     --model mgp \
#     --altitude 50 \
#     --gsam_use_segmentation_mask \
#     --gsam_box_threshold 0.20 \
#     --eval_max_timestep 20 \
#     --gsam_use_map_cache \
#     --checkpoint checkpoints/data/mgp_mturk.pth

# python main_scene_graph.py \
#     --mode eval \
#     --policy cma \
#     --eval_max_timestep 20 \
#     --checkpoint checkpoints/data/cma_mturk.pth
