#!/bin/sh

CUDA_VISIBLE_DEVICES='0'
CUDA_VISIBLE_DEVICES='1'
cd ..

python main_geonav.py\
    --mode eval\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --output_dir results/geonav/wo_scm/hard\
    --ablation full\
    --split test_unseen


python main_geonav.py\
    --mode eval\
    --altitude 80\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --split test_unseen\
    --output_dir results/geonav/altitude/80/
    
python main_geonav.py\
    --mode eval\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --output_dir results/geonav/hard/\
    --ablation full\
    --deployment local

    --map_meters 500.0 \
    --map_size 5000\

python main_geonav.py\
    --mode eval\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --checkpoint checkpoints/data/mgp_mturk.pth\
    --output_dir results/geonav/daqu/\
    --ablation full\
    --map_size 500\
    --map_name daqu_block_1\
    --episode_id 2\
    --ann_id 0


conda activate vllm

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model path/to/your/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192