"""
分析空间关系实验结果
生成统计报告和可视化图表
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_result_file(filepath):
    """加载结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_single_experiment(result_data):
    """分析单次实验结果"""
    print("\n" + "="*60)
    print("单次实验分析")
    print("="*60)
    
    # 基本信息
    config = result_data.get('config', {})
    print(f"\n配置信息:")
    print(f"  问题数量: {config.get('num_questions', 'N/A')}")
    print(f"  高度设置: {config.get('height', '所有高度')}")
    print(f"  Doubao模型: {config.get('doubao_model', 'N/A')}")
    print(f"  GPT模型: {config.get('gpt_model', 'N/A')}")
    
    # 总体准确率
    overall_acc = result_data.get('overall_accuracy', 0)
    print(f"\n总体准确率: {overall_acc*100:.2f}%")
    
    # 按高度统计
    stats_by_height = result_data.get('stats_by_height', {})
    if stats_by_height:
        print(f"\n按高度统计:")
        for height, stats in sorted(stats_by_height.items()):
            total = stats.get('total', 0)
            correct = stats.get('correct', 0)
            acc = correct / total if total > 0 else 0
            print(f"  {height}: {acc*100:.2f}% ({correct}/{total})")
    
    # 按关系类型统计
    relation_stats = result_data.get('relation_stats', {})
    if relation_stats:
        print(f"\n按关系类型统计:")
        # 按准确率排序
        sorted_relations = sorted(
            relation_stats.items(),
            key=lambda x: x[1].get('correct', 0) / x[1].get('total', 1) if x[1].get('total', 0) > 0 else 0,
            reverse=True
        )
        for relation, stats in sorted_relations:
            total = stats.get('total', 0)
            correct = stats.get('correct', 0)
            acc = correct / total if total > 0 else 0
            print(f"  {relation:15s}: {acc*100:.2f}% ({correct}/{total})")
    
    # 混淆矩阵
    confusion = result_data.get('confusion_matrix', {})
    if confusion:
        print(f"\n混淆矩阵（前5个最常见的错误）:")
        errors = []
        for true_rel, preds in confusion.items():
            for pred_rel, count in preds.items():
                if true_rel != pred_rel and count > 0:
                    errors.append((true_rel, pred_rel, count))
        
        errors.sort(key=lambda x: x[2], reverse=True)
        for true_rel, pred_rel, count in errors[:5]:
            print(f"  {true_rel} → {pred_rel}: {count}次")


def analyze_multiple_experiments(result_files):
    """分析多次实验结果"""
    print("\n" + "="*60)
    print("多次实验综合分析")
    print("="*60)
    
    all_accuracies = []
    all_relation_stats = defaultdict(lambda: {'correct': [], 'total': []})
    
    for filepath in result_files:
        try:
            data = load_result_file(filepath)
            
            # 收集总体准确率
            acc = data.get('overall_accuracy', 0)
            all_accuracies.append(acc)
            
            # 收集关系统计
            relation_stats = data.get('relation_stats', {})
            for relation, stats in relation_stats.items():
                all_relation_stats[relation]['correct'].append(stats.get('correct', 0))
                all_relation_stats[relation]['total'].append(stats.get('total', 0))
        
        except Exception as e:
            print(f"警告: 无法加载文件 {filepath}: {e}")
    
    if not all_accuracies:
        print("错误: 没有有效的实验结果")
        return
    
    # 统计总体准确率
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    min_acc = np.min(all_accuracies)
    max_acc = np.max(all_accuracies)
    
    print(f"\n总体准确率统计 (基于 {len(all_accuracies)} 次实验):")
    print(f"  平均值: {mean_acc*100:.2f}%")
    print(f"  标准差: {std_acc*100:.2f}%")
    print(f"  最小值: {min_acc*100:.2f}%")
    print(f"  最大值: {max_acc*100:.2f}%")
    print(f"  95%置信区间: [{(mean_acc - 1.96*std_acc)*100:.2f}%, {(mean_acc + 1.96*std_acc)*100:.2f}%]")
    
    # 统计每种关系的平均准确率
    print(f"\n各关系类型平均准确率:")
    for relation in sorted(all_relation_stats.keys()):
        stats = all_relation_stats[relation]
        corrects = stats['correct']
        totals = stats['total']
        
        if totals:
            # 计算每次实验的准确率
            accs = [c/t if t > 0 else 0 for c, t in zip(corrects, totals)]
            mean_rel_acc = np.mean(accs)
            std_rel_acc = np.std(accs)
            total_count = sum(totals)
            
            print(f"  {relation:15s}: {mean_rel_acc*100:.2f}% ± {std_rel_acc*100:.2f}% (n={total_count})")


def create_summary_report(result_files, output_file):
    """创建汇总报告"""
    report = {
        'num_experiments': len(result_files),
        'experiment_files': [str(f) for f in result_files],
        'statistics': {}
    }
    
    all_accuracies = []
    
    for filepath in result_files:
        try:
            data = load_result_file(filepath)
            acc = data.get('overall_accuracy', 0)
            all_accuracies.append(acc)
        except:
            continue
    
    if all_accuracies:
        report['statistics'] = {
            'mean_accuracy': float(np.mean(all_accuracies)),
            'std_accuracy': float(np.std(all_accuracies)),
            'min_accuracy': float(np.min(all_accuracies)),
            'max_accuracy': float(np.max(all_accuracies)),
            'all_accuracies': [float(a) for a in all_accuracies]
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总报告已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析空间关系实验结果')
    parser.add_argument('--file', type=str, help='单个结果文件路径')
    parser.add_argument('--pattern', type=str, default='spatial_relation_*.json',
                        help='多个结果文件的匹配模式')
    parser.add_argument('--output', type=str, default='analysis_summary.json',
                        help='汇总报告输出文件')
    
    args = parser.parse_args()
    
    if args.file:
        # 分析单个文件
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"错误: 文件不存在: {filepath}")
            return
        
        data = load_result_file(filepath)
        analyze_single_experiment(data)
    
    else:
        # 分析多个文件
        result_files = list(Path('/data1/XHT/citynav').glob(args.pattern))
        
        # 排除summary文件
        result_files = [f for f in result_files if 'summary' not in f.name and 'analysis' not in f.name]
        
        if not result_files:
            print(f"错误: 未找到匹配的结果文件: {args.pattern}")
            return
        
        print(f"找到 {len(result_files)} 个结果文件:")
        for f in result_files:
            print(f"  - {f.name}")
        
        if len(result_files) == 1:
            data = load_result_file(result_files[0])
            analyze_single_experiment(data)
        else:
            analyze_multiple_experiments(result_files)
            create_summary_report(result_files, args.output)


if __name__ == '__main__':
    main()
