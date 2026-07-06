cd ..
# test_unseen_easy
python main_geonav.py\
    --mode eval\
    --altitude 50\
    --gsam_use_segmentation_mask\
    --gsam_box_threshold 0.20\
    --train_trajectory_type mturk\
    --split test_unseen\
    --eval_batch_size 50\
    --eval_max_timestep 20\
    --output_dir results/geonav/evaluate/test_unseen_easy_4o/


# # val_unseen
# python main_geonav.py\
#     --mode eval\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --split val_unseen\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --output_dir results/geonav/evaluate/val_unseen/
# # val_seen
# python main_geonav.py\
#     --mode eval\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --split val_seen\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --output_dir results/geonav/evaluate/val_seen/
# # test_unseen
# python main_geonav.py\
#     --mode eval\
#     --altitude 50\
#     --gsam_use_segmentation_mask\
#     --gsam_box_threshold 0.20\
#     --train_trajectory_type mturk\
#     --split test_unseen\
#     --eval_batch_size 50\
#     --eval_max_timestep 20\
#     --output_dir results/geonav/evaluate/test_unseen/
