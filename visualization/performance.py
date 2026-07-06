import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hex2color  # 导入 hex2color

# 数据定义
task_levels = ['Easy', 'Medium', 'Hard']
success_rates = [26.53, 22.92, 16.67]
navigation_stages = ['Start', 'Navigate', 'Search', 'Locate']
distances = {
    'Easy': [80.46, 72.28, 59.86, 41.60],
    'Medium': [156.87, 54.54, 64.46, 53.16],
    'Hard': [193.19, 73.76, 76.53, 68.78]
}

# 使用给定的颜色HEX值，增加透明度（Alpha值设为0.6）
colors = ['#CAC8EF', '#9BDCFC', '#C9EFBE', '#F0CFEA']
colors_with_alpha = [(*hex2color(color), 0.6) for color in colors]  # 设置透明度为0.6
hatch_patterns = ['/', '\\', '|', '-']  # 简单的条纹样式

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图参数
bar_width = 0.15  # 调整柱状图宽度，使间距更小
index = np.arange(len(task_levels))

# 绘制柱状图，分别为四个导航阶段
for i, stage in enumerate(navigation_stages):
    values = [distances[level][i] for level in task_levels]
    ax1.bar(index + i * bar_width, values, bar_width,
            color=colors_with_alpha[i], edgecolor='black', hatch=hatch_patterns[i],
            label=stage)

# 设置左侧纵轴（距离）的标签和刻度
ax1.set_xlabel('Task Difficulty', fontsize=20)
ax1.set_ylabel('Distance to Target (meters)', fontsize=20)
ax1.set_xticks(index + 1.5 * bar_width)
ax1.set_xticklabels(task_levels, fontsize=18)
ax1.tick_params(axis='y', labelsize=18)

# 创建右侧纵轴，用于成功率，并将折线图点对齐到各簇中心
ax2 = ax1.twinx()
line_x = index + 1.5 * bar_width  # 让折线对应簇的中心位置
ax2.plot(
    line_x, success_rates, 
    color='lightcoral', linestyle='--', linewidth=3.5,
    marker='o', markersize=8, label='Success Rate (%)',
    zorder=5  # 提升折线图层级，确保图例能叠加在上面
)
ax2.set_ylabel('Success Rate (%)', fontsize=20)
ax2.set_ylim(0, 35)  # 增大右侧纵轴的范围
ax2.tick_params(axis='y', labelsize=18)

# 从两个坐标轴收集图例，并合并
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# 将所有图例项合并到一个图例中，放在下方，避开重叠
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc='upper center', fontsize=12, borderaxespad=1.5, 
           bbox_to_anchor=(0.5, -0.1), ncol=5, handleheight=1.5)

# 调整布局，防止标签重叠
plt.tight_layout()
plt.show()


plt.savefig('distance_success_rate.png', dpi=300)
