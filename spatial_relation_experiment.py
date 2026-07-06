"""
空间关系测试实验
使用doubao-seed生成问题，gpt-4o回答问题，统计MLLM在空间关系分类上的准确性
"""
import os
import json
import base64
import random
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from collections import defaultdict


# API配置
DOUBAO_CONFIG = {
    'model': 'doubao-seed-1-6-251015',
    'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
    'api_key': ''
}

GPT_CONFIG = {
    'model': 'gpt-4o',
    'base_url': 'https://xiaoai.plus/v1',
    'api_key': ''
}

# 所有可能的空间关系
VALID_RELATIONS = [
    "contains", "adjacent_to", "near_corner", 
    "north_of", "south_of", "east_of", "west_of",
    "northeast_of", "northwest_of", "southeast_of", "southwest_of"
]

# 数据集路径
DATASET_PATHS = {
    '20m': '/data1/XHT/citynav/results/eq6/dataset/20',
    '50m': '/data1/XHT/citynav/results/eq6/dataset/50',
    '80m': '/data1/XHT/citynav/results/eq6/dataset/80'
}


def encode_image(image_path: str) -> str:
    """将图片编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_all_images(height: str = None) -> List[Tuple[str, str]]:
    """
    获取所有图片路径
    Args:
        height: 指定高度 ('20m', '50m', '80m')，如果为None则获取所有高度的图片
    Returns:
        List of (height, image_path) tuples
    """
    images = []
    heights = [height] if height else ['20m', '50m', '80m']
    
    for h in heights:
        dataset_path = Path(DATASET_PATHS[h])
        if dataset_path.exists():
            for img_file in dataset_path.glob('*.png'):
                images.append((h, str(img_file)))
            for img_file in dataset_path.glob('*.jpg'):
                images.append((h, str(img_file)))
    
    return images


def generate_question_with_doubao(image_path: str) -> Dict:
    """
    使用doubao-seed生成一个空间关系问题
    Args:
        image_path: 图片路径
    Returns:
        问题字典，包含：
        - image_path: 图片路径
        - object_pair: 物体对 (object1, object2)
        - gt_relation: Ground Truth关系
        - options: 选项列表（包含一个正确答案和多个错误答案）
        - correct_option: 正确选项索引
    """
    client = OpenAI(
        api_key=DOUBAO_CONFIG['api_key'],
        base_url=DOUBAO_CONFIG['base_url']
    )
    
    # 编码图片
    base64_image = encode_image(image_path)
    
    # 构造prompt让模型识别物体并生成空间关系问题
    prompt = f"""You are an expert in spatial relationships. Please analyze this top-down image and:

1. Identify two distinct objects in the image
2. Determine the spatial relationship between them from this list: {', '.join(VALID_RELATIONS)}
3. Create a multiple-choice question with 4 options (1 correct, 3 incorrect)

Return ONLY a valid JSON object in this exact format (no markdown, no code blocks):
{{
    "object1": "name of first object",
    "object2": "name of second object",
    "ground_truth": "the correct spatial relationship",
    "options": ["option A", "option B", "option C", "option D"],
    "correct_index": 0
}}

The correct relationship must be one of: {', '.join(VALID_RELATIONS)}
The options should be 4 different spatial relationships, with one being the ground truth.
"""
    
    try:
        response = client.chat.completions.create(
            model=DOUBAO_CONFIG['model'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # 解析响应
        content = response.choices[0].message.content.strip()
        
        # 尝试提取JSON（处理可能的markdown包装）
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        result = json.loads(content)
        
        # 验证结果格式
        required_keys = ['object1', 'object2', 'ground_truth', 'options', 'correct_index']
        if not all(key in result for key in required_keys):
            raise ValueError(f"Missing required keys in response: {result}")
        
        # 验证ground_truth在VALID_RELATIONS中
        if result['ground_truth'] not in VALID_RELATIONS:
            print(f"Warning: Ground truth '{result['ground_truth']}' not in valid relations, using a random one")
            result['ground_truth'] = random.choice(VALID_RELATIONS)
        
        # 确保有4个选项
        if len(result['options']) != 4:
            # 如果选项不够，随机添加其他关系
            while len(result['options']) < 4:
                random_relation = random.choice(VALID_RELATIONS)
                if random_relation not in result['options']:
                    result['options'].append(random_relation)
            result['options'] = result['options'][:4]
        
        return {
            'image_path': image_path,
            'object_pair': (result['object1'], result['object2']),
            'gt_relation': result['ground_truth'],
            'options': result['options'],
            'correct_option': int(result['correct_index'])
        }
        
    except Exception as e:
        print(f"Error generating question with Doubao: {e}")
        # 返回一个默认问题
        gt_relation = random.choice(VALID_RELATIONS)
        wrong_relations = random.sample([r for r in VALID_RELATIONS if r != gt_relation], 3)
        options = [gt_relation] + wrong_relations
        random.shuffle(options)
        
        return {
            'image_path': image_path,
            'object_pair': ('object_A', 'object_B'),
            'gt_relation': gt_relation,
            'options': options,
            'correct_option': options.index(gt_relation)
        }


def answer_question_with_gpt(question: Dict) -> Dict:
    """
    使用gpt-4o回答问题
    Args:
        question: 问题字典
    Returns:
        答案字典，包含：
        - predicted_option: 预测的选项索引
        - is_correct: 是否正确
    """
    client = OpenAI(
        api_key=GPT_CONFIG['api_key'],
        base_url=GPT_CONFIG['base_url']
    )
    
    # 编码图片
    base64_image = encode_image(question['image_path'])
    
    # 构造多选题prompt
    object1, object2 = question['object_pair']
    options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(question['options'])])
    
    prompt = f"""Looking at this top-down image, what is the spatial relationship between "{object1}" and "{object2}"?

Please choose the most accurate relationship from the following options:
{options_text}

Respond with ONLY the letter of your choice (A, B, C, or D), nothing else.
"""
    
    try:
        response = client.chat.completions.create(
            model=GPT_CONFIG['model'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        # 解析响应
        answer = response.choices[0].message.content.strip().upper()
        
        # 提取字母
        if 'A' in answer:
            predicted_idx = 0
        elif 'B' in answer:
            predicted_idx = 1
        elif 'C' in answer:
            predicted_idx = 2
        elif 'D' in answer:
            predicted_idx = 3
        else:
            # 如果无法解析，随机选择
            predicted_idx = random.randint(0, 3)
        
        is_correct = (predicted_idx == question['correct_option'])
        
        return {
            'predicted_option': predicted_idx,
            'predicted_relation': question['options'][predicted_idx],
            'is_correct': is_correct,
            'raw_response': answer
        }
        
    except Exception as e:
        print(f"Error answering question with GPT-4o: {e}")
        # 返回随机答案
        predicted_idx = random.randint(0, 3)
        return {
            'predicted_option': predicted_idx,
            'predicted_relation': question['options'][predicted_idx],
            'is_correct': (predicted_idx == question['correct_option']),
            'raw_response': f"Error: {str(e)}"
        }


def run_experiment(num_questions: int = 10, height: str = None, output_file: str = None):
    """
    运行完整实验
    Args:
        num_questions: 要生成的问题数量
        height: 指定高度，如果为None则从所有高度随机采样
        output_file: 输出文件路径
    """
    print(f"=== 开始空间关系测试实验 ===")
    print(f"问题数量: {num_questions}")
    print(f"高度设置: {height if height else '所有高度'}")
    
    # 获取所有图片
    all_images = get_all_images(height)
    
    if not all_images:
        print("错误: 未找到任何图片，请先生成top-down images")
        return
    
    print(f"找到 {len(all_images)} 张图片")
    
    # 随机采样
    sampled_images = random.sample(all_images, min(num_questions, len(all_images)))
    
    # 存储所有结果
    all_results = []
    correct_count = 0
    
    # 按高度分类统计
    stats_by_height = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # 混淆矩阵：记录每种真实关系被预测为各种关系的次数
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    print("\n=== 第一步：使用Doubao-Seed生成问题 ===")
    questions = []
    for h, img_path in tqdm(sampled_images, desc="生成问题"):
        question = generate_question_with_doubao(img_path)
        question['height'] = h
        questions.append(question)
    
    print(f"\n成功生成 {len(questions)} 个问题")
    
    print("\n=== 第二步：使用GPT-4o回答问题 ===")
    for question in tqdm(questions, desc="回答问题"):
        answer = answer_question_with_gpt(question)
        
        # 合并结果
        result = {**question, **answer}
        all_results.append(result)
        
        # 统计
        if answer['is_correct']:
            correct_count += 1
            stats_by_height[question['height']]['correct'] += 1
        
        stats_by_height[question['height']]['total'] += 1
        
        # 更新混淆矩阵
        gt = question['gt_relation']
        pred = answer['predicted_relation']
        confusion_matrix[gt][pred] += 1
    
    # 计算统计结果
    overall_accuracy = correct_count / len(all_results) if all_results else 0
    
    print(f"\n=== 实验结果 ===")
    print(f"总体准确率: {overall_accuracy * 100:.2f}% ({correct_count}/{len(all_results)})")
    
    print(f"\n按高度统计:")
    for h in ['20m', '50m', '80m']:
        if stats_by_height[h]['total'] > 0:
            acc = stats_by_height[h]['correct'] / stats_by_height[h]['total']
            print(f"  {h}: {acc * 100:.2f}% ({stats_by_height[h]['correct']}/{stats_by_height[h]['total']})")
    
    # 计算每种关系的准确率
    print(f"\n按关系类型统计:")
    relation_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in all_results:
        gt = result['gt_relation']
        relation_stats[gt]['total'] += 1
        if result['is_correct']:
            relation_stats[gt]['correct'] += 1
    
    for relation in sorted(relation_stats.keys()):
        stats = relation_stats[relation]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {relation}: {acc * 100:.2f}% ({stats['correct']}/{stats['total']})")
    
    # 保存结果
    if output_file is None:
        output_file = f'spatial_relation_results_{height if height else "all"}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'num_questions': num_questions,
                'height': height,
                'doubao_model': DOUBAO_CONFIG['model'],
                'gpt_model': GPT_CONFIG['model']
            },
            'overall_accuracy': overall_accuracy,
            'stats_by_height': dict(stats_by_height),
            'relation_stats': dict(relation_stats),
            'confusion_matrix': {k: dict(v) for k, v in confusion_matrix.items()},
            'detailed_results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'stats_by_height': dict(stats_by_height),
        'relation_stats': dict(relation_stats),
        'all_results': all_results
    }


def run_multiple_trials(num_trials: int = 5, questions_per_trial: int = 10, height: str = None):
    """
    运行多次实验，计算统计结果
    关键改进：先生成固定的问题集，然后对同一批问题让GPT-4o多次回答
    这样可以测试模型在相同问题上的稳定性和一致性
    
    Args:
        num_trials: 实验次数（对同一批问题测试多少次）
        questions_per_trial: 问题数量
        height: 指定高度
    """
    print(f"=== 运行多次实验（固定问题集） ===")
    print(f"实验次数: {num_trials}")
    print(f"问题数量: {questions_per_trial}")
    print(f"高度设置: {height if height else '所有高度'}")
    
    # ====== 第一步：生成固定的问题集 ======
    print(f"\n=== 步骤1: 使用Doubao-Seed生成固定问题集 ===")
    
    # 获取所有图片
    all_images = get_all_images(height)
    
    if not all_images:
        print("错误: 未找到任何图片，请先生成top-down images")
        return
    
    print(f"找到 {len(all_images)} 张图片")
    
    # 随机采样
    if questions_per_trial > len(all_images):
        print(f"提示: 请求问题数({questions_per_trial}) > 图片总数({len(all_images)})，将重复使用图片生成问题。")
        # random.choices 允许重复采样 (有放回采样)
        sampled_images = random.choices(all_images, k=questions_per_trial)
    else:
        # 如果图片够多，依然使用不重复采样
        sampled_images = random.sample(all_images, questions_per_trial)
    
    # 生成固定的问题集
    questions = []
    for h, img_path in tqdm(sampled_images, desc="生成问题"):
        question = generate_question_with_doubao(img_path)
        question['height'] = h
        questions.append(question)
    
    print(f"\n成功生成 {len(questions)} 个固定问题")
    
    # 保存问题集供参考
    questions_file = f'spatial_relation_questions_{height if height else "all"}.json'
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"问题集已保存至: {questions_file}")
    
    # ====== 第二步：对同一批问题多次测试GPT-4o ======
    print(f"\n=== 步骤2: 对同一批问题进行 {num_trials} 次GPT-4o测试 ===")
    
    all_accuracies = []
    all_trial_results = []
    
    for trial in range(num_trials):
        print(f"\n--- 第 {trial + 1}/{num_trials} 次测试 ---")
        
        trial_results = []
        correct_count = 0
        
        # 对同一批问题让GPT-4o回答
        for question in tqdm(questions, desc=f"GPT-4o回答(轮次{trial+1})"):
            answer = answer_question_with_gpt(question)
            
            # 合并结果
            result = {**question, **answer, 'trial': trial}
            trial_results.append(result)
            
            if answer['is_correct']:
                correct_count += 1
        
        accuracy = correct_count / len(questions) if questions else 0
        all_accuracies.append(accuracy)
        all_trial_results.append(trial_results)
        
        print(f"第 {trial + 1} 次测试准确率: {accuracy * 100:.2f}% ({correct_count}/{len(questions)})")
        
        # 保存每次测试的详细结果
        output_file = f'spatial_relation_trial_{trial}_{height if height else "all"}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'trial': trial,
                'accuracy': accuracy,
                'questions': questions,
                'results': trial_results
            }, f, indent=2, ensure_ascii=False)
    
    # ====== 第三步：统计分析 ======
    print(f"\n=== 步骤3: 统计分析结果 ===")
    
    # 计算统计量
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    min_acc = np.min(all_accuracies)
    max_acc = np.max(all_accuracies)
    
    print(f"\n【整体统计】（基于同一批 {len(questions)} 个问题的 {num_trials} 次测试）")
    print(f"  平均准确率: {mean_acc * 100:.2f}%")
    print(f"  标准差: {std_acc * 100:.2f}%")
    print(f"  最小值: {min_acc * 100:.2f}%")
    print(f"  最大值: {max_acc * 100:.2f}%")
    print(f"  95%置信区间: [{(mean_acc - 1.96*std_acc)*100:.2f}%, {(mean_acc + 1.96*std_acc)*100:.2f}%]")
    print(f"  所有准确率: {[f'{acc*100:.2f}%' for acc in all_accuracies]}")
    
    # 分析每个问题的一致性
    print(f"\n【问题级别分析】")
    question_consistency = []
    for q_idx in range(len(questions)):
        # 统计这个问题在各次测试中被答对的次数
        correct_times = sum(1 for trial_results in all_trial_results 
                           if trial_results[q_idx]['is_correct'])
        consistency_rate = correct_times / num_trials
        question_consistency.append(consistency_rate)
    
    always_correct = sum(1 for c in question_consistency if c == 1.0)
    always_wrong = sum(1 for c in question_consistency if c == 0.0)
    sometimes_correct = len(questions) - always_correct - always_wrong
    
    print(f"  总是答对的问题: {always_correct}/{len(questions)} ({always_correct/len(questions)*100:.1f}%)")
    print(f"  总是答错的问题: {always_wrong}/{len(questions)} ({always_wrong/len(questions)*100:.1f}%)")
    print(f"  有时对有时错的问题: {sometimes_correct}/{len(questions)} ({sometimes_correct/len(questions)*100:.1f}%)")
    print(f"  平均一致性: {np.mean(question_consistency)*100:.2f}%")
    
    # 保存汇总结果
    summary_file = f'spatial_relation_summary_{height if height else "all"}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_type': 'fixed_questions_multiple_trials',
            'description': '对同一批问题进行多次测试，评估模型稳定性和一致性',
            'num_trials': num_trials,
            'questions_per_trial': questions_per_trial,
            'height': height,
            'overall_statistics': {
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'min_accuracy': float(min_acc),
                'max_accuracy': float(max_acc),
                'confidence_interval_95': [
                    float(mean_acc - 1.96*std_acc),
                    float(mean_acc + 1.96*std_acc)
                ],
                'all_accuracies': [float(a) for a in all_accuracies]
            },
            'consistency_analysis': {
                'always_correct': always_correct,
                'always_wrong': always_wrong,
                'sometimes_correct': sometimes_correct,
                'average_consistency': float(np.mean(question_consistency)),
                'question_consistency_rates': [float(c) for c in question_consistency]
            },
            'questions_file': questions_file
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总结果已保存至: {summary_file}")
    print(f"\n=== 实验完成 ===")
    print(f"问题集文件: {questions_file}")
    print(f"汇总文件: {summary_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='空间关系测试实验')
    parser.add_argument('--num_questions', type=int, default=10, help='问题数量')
    parser.add_argument('--height', type=str, choices=['20m', '50m', '80m'], 
                        help='指定高度，如果不指定则从所有高度采样')
    parser.add_argument('--num_trials', type=int, default=1, help='实验次数')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    if args.num_trials > 1:
        run_multiple_trials(
            num_trials=args.num_trials,
            questions_per_trial=args.num_questions,
            height=args.height
        )
    else:
        run_experiment(
            num_questions=args.num_questions,
            height=args.height,
            output_file=args.output
        )
