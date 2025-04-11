import os
import json
import torch
from openai import OpenAI
import numpy as np
from tqdm import trange

from scenegraphnav.parser import parse_args
from gsamllavanav.evaluate import eval_planning_metrics
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from scenegraphnav.evaluate import run_episodes_batch
from scenegraphnav.agent import GeonavAgent

DEVICE = 'cuda'
test_data = 'easy_good_case'

args = parse_args()

if args.mode == 'eval':
    objects = get_city_refer_objects()

    if args.test_one_example:
        # 测试单个样例
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects, 
            load_mturk_trajectories(args.split, 'all', args.altitude),
            max_episodes=1
        )
    else:
        # 测试整个split的所有样例
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(args.split, test_data, args.altitude),
            max_episodes=None
        )
    # 选择规划器运动至 landmark质点位置
    if args.landmark_mode == 'planner':
        # 使用run_episodes_batch处理landmark_mode为'planner'的情况
        trajectory_logs, target_xys = run_episodes_batch(args, None, test_episodes, DEVICE, landmark_mode='planner')
    else:
        from gsamllavanav.observation import cropclient
        cropclient.load_image_cache()
        trajectory_logs = dict()
    
    # 初始化agents列表
    agents = []
    results = []
    
    def initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key):
        if VLM_backbone == 'Qwen-local':
            # 尝试使用32B的本地模型
            from transformers import Qwen2VLForConditionalGeneration
            vlmodel = Qwen2VLForConditionalGeneration.from_pretrained(
                "/data1/FoundationModels/Qwen",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        elif VLM_backbone == 'Qwen-online':
            vlmodel = OpenAI(
                api_key=vl_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif VLM_backbone == 'GPT':
            vlmodel = OpenAI(
                api_key=vl_api_key,
                base_url= 'https://api.chatanywhere.tech' #'https://api.chatanywhere.tech',
            )
        if LLM_backbone == 'Qwen-online':
            llmodel = OpenAI(
                api_key=ll_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif LLM_backbone == 'GPT':
            llmodel = OpenAI(
                api_key=ll_api_key,
                base_url='https://api.chatanywhere.tech'#'https://xiaoai.plus/v1',
            )
        return vlmodel, llmodel
    
    # 为test_episodes的每个episode创建一个Agent，并将episode数据传入Agent
    VLM_backbone = 'GPT' # visual model
    LLM_backbone = 'Qwen-online' # language model
    vl_api_key =  "sk-xHX92exOc6iulrMz8q8BGcXOveU8qVgpfDkvdXdbctOA4rOr"
    ll_api_key = "sk-ca477c37e2214255a5498915ea609ae5"
    os.environ["OPENAI_API_KEY"] = "sk-xHX92exOc6iulrMz8q8BGcXOveU8qVgpfDkvdXdbctOA4rOr"
    # # 设置 OPENAI_BASE_URL 环境变量
    os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech"
    vlmodel, llmodel = initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key)
    strategy_distance_records = {
        'Start': [],
        'Navigate': [],
        'Search': [],
        'Locate': []
    }
    for episode in test_episodes:
        # 创建Agent实例
        agent = GeonavAgent(args, episode.start_pose, episode, vlmodel, set_height=None)
        # 设置目标
        agent.set_target(episode.target_position) # 假设目标是episode的target_position
        # 运行Agent
        res, trajectory_log = agent.run()
        # 更新元组中的距离信息
        for strategy in strategy_distance_records.keys():
            if strategy in agent.strategy_distances:
                strategy_distance_records[strategy].append(agent.strategy_distances[strategy])
        results.append(res)
        trajectory_logs[episode.id] = trajectory_log
        # 累积运行结果
        agent.strategy_distances
    
    metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)

    print(f"{args.split} -- NE {metrics.mean_final_pos_to_goal_dist: .1f}, SR {metrics.success_rate_final_pos_to_goal*100: .2f}, OSR {metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {metrics.success_rate_weighted_by_path_length*100: .2f}metrics")
    for strategy, distances in strategy_distance_records.items():
        if distances:  # 确保列表不为空
            avg_distance = sum(distances) / len(distances)
            print(f"Average distance for {strategy} : {avg_distance:.2f} (meters)")
        else:
            print(f"No data available for {strategy}")
    noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
    alt_env = f"_{args.alt_env}" if args.alt_env else ""
    with open(args.output_dir + f'geonav_{args.split}_{args.ablation}_{args.eval_goal_selector}_{test_data}.json', 'w') as f:
        json.dump({
            'metrics': metrics.to_dict(),
            'success': results,
            'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
            'strategy_averages': {k: (sum(v)/len(v) if len(v)>0 else None) for k,v in strategy_distance_records.items()}
        }, f)
