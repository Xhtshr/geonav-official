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
            load_mturk_trajectories(args.split, 'all', args.altitude),
            max_episodes=10
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
    # 为test_episodes的每个episode创建一个Agent，并将episode数据传入Agent
    VLM_backbone = 'Qwen2.5-VL-72b' # visual model
    LLM_backbone = 'Qwen-max' # language model
    if VLM_backbone == 'Qwen2-vl-7b':
        from transformers import Qwen2VLForConditionalGeneration
        vlmodel = Qwen2VLForConditionalGeneration.from_pretrained(
            "/data1/FoundationModels/Qwen",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    elif VLM_backbone == 'Qwen2.5-VL-72b':
        # base64_image = encode_image("test.png")
        vlmodel = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key="sk-f0de3487904a4a11950ba707623cdbab",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif VLM_backbone == 'GPT-4o':
        vlmodel = OpenAI(
            # 下面两个参数的默认值来自环境变量，可以不加
            api_key="sk-dooWu6cCsNTtSsB7Fb5f2f25Cd164b67A94cFd650442EcB2",
            base_url='https://xiaoai.plus/v1',
        )
    
    if LLM_backbone == 'Qwen-max':
        llmodel = OpenAI(
            # 下面两个参数的默认值来自环境变量，可以不加
            api_key="sk-dooWu6cCsNTtSsB7Fb5f2f25Cd164b67A94cFd650442EcB2",
            base_url='https://xiaoai.plus/v1',
        )
    elif LLM_backbone == 'GPT-4o':
        llmodel = OpenAI(
            # 下面两个参数的默认值来自环境变量，可以不加
            api_key="sk-dooWu6cCsNTtSsB7Fb5f2f25Cd164b67A94cFd650442EcB2",
            base_url='https://xiaoai.plus/v1',
        )
    elif LLM_backbone == 'GPT-3.5-turbo':
        llmodel = OpenAI(
            # 下面两个参数的默认值来自环境变量，可以不加
            api_key="sk-8xBWP046CnOzBAEaC262872c0f4d40EeAc366eB651B7C020",
            base_url='https://xiaoai.plus/v1',
        )


    for episode in test_episodes:
        # 创建Agent实例
        agent = SceneAgent(args, trajectory_logs[episode.id][-1], episode, vlmodel, set_height=None)
        # 设置目标
        agent.set_target(episode.target_position)  # 假设目标是episode的target_position
        # 运行Agent
        res = agent.run()
        results.append(res)
    print(results)
    
    # # 使用VLM推断目标位置
    # predicted_positions = (goal_selection_gdino if args.eval_goal_selector == 'gdino' else goal_selection_llava)(args, pred_goal_logs)
    # for eps_id, pose in predicted_positions.items():
    #     trajectory_logs[eps_id].append(pose)
    
    # metrics = eval_goal_predictor(args, test_episodes, trajectory_logs, pred_goal_logs, pred_progress_logs)

    # print(f"{args.split} -- {metrics.mean_final_pos_to_goal_dist: .1f}, {metrics.success_rate_final_pos_to_goal*100: .2f}, {metrics.success_rate_oracle_pos_to_goal*100: .2f}")
    
    # noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
    # alt_env = f"_{args.alt_env}" if args.alt_env else ""
    # with open(f'llm_controller_{args.split}_{args.progress_stop_val}{noise}{alt_env}_{args.eval_goal_selector}.json', 'w') as f:
    #     json.dump({
    #         'metrics': metrics.to_dict(),
    #         'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
    #         'pred_goal_logs': {str(eps_id): [tuple(pos) for pos in pred_goals] for eps_id, pred_goals in pred_goal_logs.items()},
    #         'pred_progress_logs': {str(eps_id): pred_progresses for eps_id, pred_progresses in pred_progress_logs.items()},
    #     }, f)

