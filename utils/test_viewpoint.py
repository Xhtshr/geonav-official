import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append('/data1/XHT/citynav/')

import json

import torch

from gsamllavanav.parser import parse_args

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.observation import cropclient
from gsamllavanav.space import Point2D, Pose4D

args = parse_args()
objects = get_city_refer_objects()

# Generate RGBs of target objects and its description for episodes to evaluate llm reasoning capability
# for split in ('val_seen', 'val_unseen', 'test_unseen'):
for split in ['val_seen', 'val_unseen', 'test_unseen']:
    for dif in ['easy_simpled', 'medium_simpled', 'hard_simpled']:
        cropclient.load_image_cache(alt_env=args.alt_env)
        test_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories(split, dif, args.altitude))
        target_poses = [Pose4D(eps.target_position.x, eps.target_position.y, eps.target_position.z + 40, eps.start_pose.yaw) for eps in test_episodes]
        target_descriptions = [eps.target_description for eps in test_episodes]
        for i, pose in enumerate(target_poses):
            # Ensure the directory exists
            output_dir = f'utils/{split}_{dif}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Store each target_description with rgb image
            rgb = cropclient.crop_image(test_episodes[i].map_name, pose, (400, 400), 'rgb')
            cv2.imwrite(os.path.join(output_dir, f'{i}.png'), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            with open(os.path.join(output_dir, f'{i}_description.txt'), 'w') as f:
                f.write(target_descriptions[i])
