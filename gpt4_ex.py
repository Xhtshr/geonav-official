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
from gsamllavanav.goal_selection import goal_selection_gdino, goal_selection_llava
from scenegraphnav.evaluate import run_episodes_batch
from gsamllavanav.models.goal_predictor import GoalPredictor
from scenegraphnav.agent import ChatAgent

DEVICE = 'cuda'

args = parse_args()

for t in range(5):
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
                load_mturk_trajectories(args.split, f'easy_simpled_{t}', args.altitude),
                max_episodes=None
            )
        # # 选择目标预测器或规划器运动至 landmark
        from gsamllavanav.observation import cropclient
        cropclient.load_image_cache()
        # 初始化agents列表
        agents = []
        results = []
        trajectory_logs = dict()
        def initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key):
            if VLM_backbone == 'Qwen2-vl-7b':
                from transformers import Qwen2VLForConditionalGeneration
                vlmodel = Qwen2VLForConditionalGeneration.from_pretrained(
                    "/data1/FoundationModels/Qwen",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            elif VLM_backbone == 'Qwen2.5-VL-72b':
                vlmodel = OpenAI(
                    api_key=vl_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif VLM_backbone == 'GPT-4o':
                vlmodel = OpenAI(
                    api_key=vl_api_key,
                    base_url='https://api.chatanywhere.tech',
                )
            if LLM_backbone == 'Qwen-max':
                llmodel = OpenAI(
                    api_key=ll_api_key,
                    base_url='https://xiaoai.plus/v1',
                )
            elif LLM_backbone == 'GPT-4o':
                llmodel = OpenAI(
                    api_key=ll_api_key,
                    base_url='https://xiaoai.plus/v1',
                )
            elif LLM_backbone == 'GPT-3.5-turbo':
                llmodel = OpenAI(
                    api_key=ll_api_key,
                    base_url='https://xiaoai.plus/v1',
                )
            
            return vlmodel, llmodel

        # 为test_episodes的每个episode创建一个Agent，并将episode数据传入Agent
        VLM_backbone = 'Qwen2.5-VL-72b' # visual model
        LLM_backbone = 'Qwen-max' # language model
        vl_api_key =  "sk-f0de3487904a4a11950ba707623cdbab"#"sk-xHX92exOc6iulrMz8q8BGcXOveU8qVgpfDkvdXdbctOA4rOr"#qwen2.5 
        ll_api_key = "sk-f0de3487904a4a11950ba707623cdbab"#"sk-dooWu6cCsNTtSsB7Fb5f2f25Cd164b67A94cFd650442EcB2" # "sk-8xBWP046CnOzBAEaC262872c0f4d40EeAc366eB651B7C020",for 3.5-turbo
        
        vlmodel, llmodel = initialize_models(VLM_backbone, LLM_backbone, vl_api_key, ll_api_key)

        for episode in test_episodes:
            # 创建Agent实例
            agent = ChatAgent(args, episode.start_pose, episode, vlmodel, set_height=None)
            # 设置目标
            agent.set_target(episode.target_position)  # 假设目标是episode的target_position
            # 运行Agent
            res, trajectory_log = agent.run(naive=False)
            results.append(res)
            trajectory_logs[episode.id] = trajectory_log
        # results 里面是true or false,计算正确率
        accuracy = sum(results) / len(results) if results else 0
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # # 使用VLM推断目标位置
        # predicted_positions = (goal_selection_gdino if args.eval_goal_selector == 'gdino' else goal_selection_llava)(args, pred_goal_logs)
        # for eps_id, pose in predicted_positions.items():
        #     trajectory_logs[eps_id].append(pose)
        
        metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)

        print(f"{args.split} -- {metrics.mean_final_pos_to_goal_dist: .1f}, {metrics.success_rate_final_pos_to_goal*100: .2f}, {metrics.success_rate_oracle_pos_to_goal*100: .2f}, {metrics.success_rate_weighted_by_path_length*100: .2f}")
        
        noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
        alt_env = f"_{args.alt_env}" if args.alt_env else ""
        with open(f'naive_qwen25_{args.split}_{args.progress_stop_val}{noise}{alt_env}_{args.eval_goal_selector}_{t}.json', 'w') as f:
            json.dump({
                'metrics': metrics.to_dict(),
                'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
            }, f)

