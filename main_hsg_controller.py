import json

import torch
import numpy as np
from gsamllavanav.parser import parse_args
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.goal_selection import goal_selection_gdino, goal_selection_llava
from ggb.test_viewpoint import Agent # TODO

DEVICE = 'cuda'

from gsamllavanav.observation import cropclient
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.dataset.episode import Episode, EpisodeID
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from tqdm import tqdm, trange
from gsamllavanav.space import Point2D, Pose4D

def run_single_spisode(
        args: ExperimentArgs,
        eps: Episode
):
    pose_log = []
    cropclient.load_image_cache(alt_env=args.alt_env) # prepare the image observation
    pose = eps.start_pose
    done = False
    # find the landmark
    nav_map = LandmarkNavMap(eps.map_name, args.map_shape, args.map_pixels_per_meter,
                             eps.description_landmarks, eps.description_target, eps.description_surroundings, args.gsam_params)
    for t in trange(args.eval_max_timestep, desc='eval timestep', unit='step', colour='#66aa66', position=2, leave=False):
        # decide whether arrived the landmark
        arrived = False
        if arrived:
            scene_graph()
        else:
            
            A_star()
        
        # update map
        if not done:
            gsam_rgb = cropclient.crop_image(eps.map_name, pose, args.gsam_rgb_shape, 'rgb')
            nav_map.update_observations(noisy_pose, gsam_rgb, None, args.gsam_use_map_cache)
            pose_log.append(pose)
        # prepare input
        map = nav_map.to_array()
        rgb = cropclient.crop_image(eps.map_name, pose, (224, 224), 'rgb').transpose(0, 3, 1, 2)
        normalized_depth = cropclient.crop_image(eps.map_name, pose, (256, 256), 'depth').transpose(0, 3, 1, 2) / args.max_depth
        
        # predict
        
        # move
        pose = move(pose, )

args = parse_args()

if args.model == 'mgp':
    from gsamllavanav.train import train
    from gsamllavanav.evaluate import run_episodes_batch
else:
    from gsamllavanav.train_baseline_with_map import train
    from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch

if args.mode == 'train':
    raise NotImplementedError('No implement such models for training')

if args.mode == 'eval':
    # agent = Agent() #TODO
    model_trajectory = args.checkpoint.split('/')[-2]
    epoch = args.checkpoint.split('/')[-1].split('.')[0]

    objects = get_city_refer_objects()

    for split in ('val_seen', 'val_unseen', 'test_unseen'):
        
        test_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories(split, 'all', args.altitude))
        test_episode = test_episodes[0] # only test one

        run_single_spisode(args, test_episode)

        trajectory_logs, pred_goal_logs, pred_progress_logs = run_episodes_batch(args, agent, test_episodes, DEVICE)

        predicted_positions = (goal_selection_gdino if args.eval_goal_selector == 'gdino' else goal_selection_llava)(args, pred_goal_logs)
        for eps_id, pose in predicted_positions.items():
            trajectory_logs[eps_id].append(pose)
        
        metrics = eval_goal_predictor(args, test_episodes, trajectory_logs, pred_goal_logs, pred_progress_logs)

        print(f"{split} -- {metrics.mean_final_pos_to_goal_dist: .1f}, {metrics.success_rate_final_pos_to_goal*100: .2f}, {metrics.success_rate_oracle_pos_to_goal*100: .2f}")
        
        noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
        alt_env = f"_{args.alt_env}" if args.alt_env else ""
        with open(f'{args.model}_{model_trajectory}_{split}_{args.progress_stop_val}{noise}{alt_env}_{args.eval_goal_selector}.json', 'w') as f:
            json.dump({
                'metrics': metrics.to_dict(),
                'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
                'pred_goal_logs': {str(eps_id): [tuple(pos) for pos in pred_goals] for eps_id, pred_goals in pred_goal_logs.items()},
                'pred_progress_logs': {str(eps_id): pred_progresses for eps_id, pred_progresses in pred_progress_logs.items()},
            }, f)

