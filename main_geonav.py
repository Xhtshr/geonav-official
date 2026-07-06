import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import os
import json
#import sys 可能会用于调整
from openai import OpenAI
from tqdm import trange
from zai import ZhipuAiClient

from scenegraphnav.parser import parse_args
from gsamllavanav.evaluate import eval_planning_metrics
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from scenegraphnav.evaluate import run_episodes_batch
from scenegraphnav.agent import GeonavAgent, ChatAgent

DEVICE = 'cuda'
test_data = 'all'
gpt_api_key =  ""
#qwen3-vl-plus
qwen_api_key = "sk-xxx"
model_api_key="ms-xxx"
zhipu_api_key=""
#火山引擎豆包seed、deepseek
args = parse_args()

# 假设 arg.output_dir 是传入的参数
output_dir = args.output_dir
# 如果文件夹不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.mode == 'eval':
    objects = get_city_refer_objects()
    # 生成 episodes：如果指定了具体案例，则生成所有然后过滤；否则根据 test_one_example 生成
    if args.map_name is not None and args.episode_id is not None:
        # 指定了具体案例，生成所有 episodes 然后过滤
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(args.split, test_data, args.altitude),
            max_episodes=None
        )
        test_episodes = [ep for ep in test_episodes if ep.map_name == args.map_name and ep.target_object.id == args.episode_id and (args.ann_id is None or ep.description_id == args.ann_id)]
        if not test_episodes:
            print(f"No episode found for map_name={args.map_name}, episode_id={args.episode_id}, ann_id={args.ann_id}")
            exit(1)
        print(f"Selected specific episode: map_name={args.map_name}, episode_id={args.episode_id}, ann_id={args.ann_id}")
    elif args.test_one_example:
        # 测试单个样例（随机或第一个）
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects, 
            load_mturk_trajectories(args.split, 'all', args.altitude),
            max_episodes=1
        )
        print(f"Testing one example: map_name={test_episodes[0].map_name}, episode_id={test_episodes[0].id}")
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
        # 计算指标并保存结果，然后结束程序（避免重复运行 agent）
        # metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)
        # print(f"{args.split} -- NE {metrics.mean_final_pos_to_goal_dist: .1f}, SR {metrics.success_rate_final_pos_to_goal*100: .2f}, OSR {metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {metrics.success_rate_weighted_by_path_length*100: .2f}metrics")
        # with open(args.output_dir + f'geonav_{args.split}_{args.ablation}_{test_data}.json', 'w') as f:
        #     json.dump({
        #         'metrics': metrics.to_dict(),
        #         'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
        #     }, f)
        # sys.exit(0)
    else:
        from gsamllavanav.observation import cropclient
        cropclient.load_image_cache()
        trajectory_logs = dict()
    
    # 初始化agents列表
    agents = []
    results = []
    
    def initialize_models(VLM_backbone, LLM_backbone, model_api_key, qwen_api_key, zhipu_api_key):
        MODEL_CONFIGS = {
            'local': {
                'api_key': "EMPTY",
                'base_url': "http://localhost:8000/v1"
            },
            'online': {
                'base_urls': {
                    'Qwen-online': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                    'Qwen-modelscope': 'https://api-inference.modelscope.cn/v1',
                    'GPT': 'https://api.chatanywhere.tech'
                }
            }
        }
            
        if args.deployment == 'local':
            return OpenAI(**MODEL_CONFIGS['local']), OpenAI(
                api_key=gpt_api_key,
                base_url=MODEL_CONFIGS['online']['base_urls'][LLM_backbone]
            )
        else:
            if VLM_backbone == 'GLM':
                vlmodel = ZhipuAiClient(api_key=zhipu_api_key)
            if LLM_backbone == 'GLM':
                llmodel = ZhipuAiClient(api_key=zhipu_api_key)
            else:
                vlmodel = OpenAI(
                    api_key=qwen_api_key,#gpt_api_key,#qwen_api_key
                    base_url=MODEL_CONFIGS['online']['base_urls'][VLM_backbone]
                )
                llmodel = OpenAI(
                    api_key=qwen_api_key,
                    base_url=MODEL_CONFIGS['online']['base_urls'][LLM_backbone]
                )
            return vlmodel, llmodel
    
    # model configuration
    VLM_backbone = 'Qwen-online'#'GPT'
    LLM_backbone = 'Qwen-online'#'GPT'
    #LLM_backbone = 'Qwen-modelscope'#'GPT'
    vlmodel, llmodel = initialize_models(
        VLM_backbone, 
        LLM_backbone,
        model_api_key,
        qwen_api_key,
        zhipu_api_key
    )

    strategy_distance_records = {
        'Start': [],
        'Navigate': [],
        'Search': [],
        'Locate': []
    }
    output_directory = os.path.join(args.output_dir, "geonav_origin", "val_unseen")
    os.makedirs(output_directory, exist_ok=True)  # 自动创建目录（包括父目录）
    metrics_jsonl_path = os.path.join(output_directory, "episode_metrics_val_unseen-2.jsonl")

    start_idx = 2385
    end_idx = None
    # 确保 end_idx 不越界
    if end_idx is not None:
        end_idx = min(end_idx, len(test_episodes))
    with open(metrics_jsonl_path, 'a') as metrics_f:
        for i, episode in enumerate(test_episodes[start_idx:end_idx], start=start_idx):
            print(f"Running episode {i+1}")
            # 创建Agent实例
            agent = GeonavAgent(args, episode.start_pose, episode, vlmodel, llmodel, set_height=None)
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
            #agent.strategy_distances
            # 实时评估
            current_metrics = eval_planning_metrics(args, test_episodes[start_idx:i+1], trajectory_logs)
            print(f"[Episode {i+1}] NE: {current_metrics.mean_final_pos_to_goal_dist:.1f}, SR: {current_metrics.success_rate_final_pos_to_goal*100:.2f}, OSR {current_metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {current_metrics.success_rate_weighted_by_path_length*100: .2f}metrics")
            # 构造当前行的 JSON 对象（包含 episode 信息 + 累计指标）
            metric_entry = {
                "Episode": i+1,
                "episode_id": episode.id,
                "NE": current_metrics.mean_final_pos_to_goal_dist,
                "SR": current_metrics.success_rate_final_pos_to_goal,
                "OSR": current_metrics.success_rate_oracle_pos_to_goal,
                "SPL": current_metrics.success_rate_weighted_by_path_length,
                "Navigate_steps": agent.strategy_timesteps.get('Navigate', 0),
                "Search_steps": agent.strategy_timesteps.get('Search', 0),
                "Locate_steps": agent.strategy_timesteps.get('Locate', 0),
            }
            '''
            # === 新增：添加每个 strategy 的当前平均距离 ===
            for strategy, distances in strategy_distance_records.items():
                if distances:  # 非空列表
                    avg_dist = sum(distances) / len(distances)
                    metric_entry[f"average_distance_{strategy}"] = avg_dist
                else:
                    metric_entry[f"average_distance_{strategy}"] = None
            '''
            # 写入一行 JSON（JSONL 格式）
            metrics_f.write(json.dumps(metric_entry, ensure_ascii=False) + "\n")
            metrics_f.flush()  # 立即写入磁盘，防止缓存丢失


    metrics = eval_planning_metrics(args, test_episodes, trajectory_logs)

    print(f"{args.split} -- NE {metrics.mean_final_pos_to_goal_dist: .1f}, SR {metrics.success_rate_final_pos_to_goal*100: .2f}, OSR {metrics.success_rate_oracle_pos_to_goal*100: .2f}, SPL {metrics.success_rate_weighted_by_path_length*100: .2f}metrics")
    for strategy, distances in strategy_distance_records.items():
        if distances:  # 确保列表不为空
            avg_distance = sum(distances) / len(distances)
            print(f"Average distance for {strategy} : {avg_distance:.2f} (meters)")
        else:
            print(f"No data available for {strategy}")
    noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
    map_type = f"_{args.map_type}" if args.map_type != 'topdown_map' else ""
    
    with open(args.output_dir + f'geonav_{args.split}_{args.ablation}_{test_data}{map_type}.json', 'w') as f:
        json.dump({
            'metrics': metrics.to_dict(),
            'success': results,
            'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
            'strategy_averages': {k: (sum(v)/len(v) if len(v)>0 else None) for k,v in strategy_distance_records.items()}
        }, f)
