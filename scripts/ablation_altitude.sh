#!/bin/sh
# 30, 50, 70, 90, 120；设计一些小的case； 补充高度对应的空间分辨率；
# 指标是什么？ 设计能否识别（如果找到了节点算成功），计算GroundingDINO和GT之间的IOU，成功率
CUDA_VISIBLE_DEVICES='0'

cd ..

# python main_geonav.py\
#     --mode eval\
#     --altitude 20\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/altitude/20/

python main_geonav.py\
    --mode eval\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --output_dir results/geonav/altitude/50/

# python main_geonav.py\
#     --mode eval\
#     --altitude 80\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/altitude/80/


# python main_geonav.py\
#     --mode eval\
#     --altitude 100\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --checkpoint checkpoints/data/mgp_mturk.pth\
#     --output_dir results/geonav/altitude/100/