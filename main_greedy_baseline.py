import json
import torch
from openai import OpenAI
import numpy as np
from tqdm import trange

from scenegraphnav.parser import parse_args
from gsamllavanav.evaluate import eval_goal_predictor, move
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.goal_selection import goal_selection_gdino, goal_selection_llava
from scenegraphnav.evaluate import run_episodes_batch
from gsamllavanav.models.goal_predictor import GoalPredictor
from scenegraphnav.agent import SceneAgent

DEVICE = 'cuda'

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
            load_mturk_trajectories(args.split, 'new', args.altitude),
            max_episodes=None
        )
    # 选择目标预测器或规划器运动至 landmark
    if args.model == 'mgp' and args.landmark_mode == 'predictor':
        model_trajectory = args.checkpoint.split('/')[-2]
        epoch = args.checkpoint.split('/')[-1].split('.')[0]
        model = GoalPredictor(args.map_size).to(DEVICE)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint)['predictor_state_dict'])
        # 导航至 landmark
        trajectory_logs, pred_goal_logs, pred_progress_logs = run_episodes_batch(args, model, test_episodes, DEVICE, landmark_mode='predictor')
    elif args.landmark_mode == 'planner':
        # 使用run_episodes_batch处理landmark_mode为'planner'的情况
        trajectory_logs, target_xys = run_episodes_batch(args, None, test_episodes, DEVICE, landmark_mode='planner')
    
    # 初始化agents列表
    agents = []
    results = []
    def initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key):
        if VLM_backbone == 'Qwen-local':
            # 尝试使用72B的本地模型
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
        elif VLM_backbone == 'GPT-4o':
            vlmodel = OpenAI(
                api_key=vl_api_key,
                base_url='https://xiaoai.plus/v1',
            )
        
        if LLM_backbone == 'Qwen-max':
            llmodel = OpenAI(
                api_key=ll_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif LLM_backbone == 'GPT':
            llmodel = OpenAI(
                api_key=ll_api_key,
                base_url='https://xiaoai.plus/v1',
            )
        
        return vlmodel, llmodel

    # 为test_episodes的每个episode创建一个Agent，并将episode数据传入Agent
    VLM_backbone = 'Qwen2.5-VL-72b' # visual model
    LLM_backbone = 'Qwen-max' # language model
    vl_api_key = "" #qwen
    ll_api_key = "" # ,for 3.5-turbo
    
    vlmodel, llmodel = initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key)

    for episode in test_episodes:
        # 创建Agent实例
        agent = SceneAgent(args, trajectory_logs[episode.id][-1], episode, vlmodel, set_height=None)
        # 设置目标
        agent.set_target(episode.target_position)  # 假设目标是episode的target_position
        # 运行Agent
        res = agent.run()
        results.append(res)
    # results 里面是true or false,计算正确率
    accuracy = sum(results) / len(results) if results else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")

