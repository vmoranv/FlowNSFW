#!/usr/bin/env python3
"""精美好看的性能对比图."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
})

models = ['FlowNSFW v2.0', 'FlowNSFW v1.0', 'YOLOv11']
accuracy = [93.3, 71.1, 57.8]
nsfw_recall = [96.0, 48.0, 24.0]
sfw_accuracy = [90.0, 100.0, 100.0]
speed = [1.62, 3.51, 0.22]

colors_v2 = ['#58a6ff', '#3fb950', '#f0883e']
colors_v1 = ['#484f58', '#484f58', '#484f58']
colors_yolo = ['#21262d', '#21262d', '#21262d']
speed_colors = ['#58a6ff', '#3fb950', '#f0883e']

x = np.arange(len(models))
width = 0.18

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor('#0d1117')

# ===== LEFT: Accuracy metrics =====
ax1 = fig.add_subplot(1, 2, 1)

for i, (model, acc, ns, sf) in enumerate(zip(models, accuracy, nsfw_recall, sfw_accuracy)):
    bars = ax1.bar([i - width, i, i + width], [acc, ns, sf], width,
                   color=[colors_v2[i], colors_v2[i], colors_v2[i]],
                   edgecolor='#30363d', linewidth=0.5, alpha=0.9)
    bars[0].set_alpha(1.0)
    bars[1].set_alpha(0.65)
    bars[2].set_alpha(0.35)
    # 数值标签
    for bar, val in zip(bars, [acc, ns, sf]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color='#e6edf3')

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#58a6ff', alpha=1.0, label='Accuracy'),
    Patch(facecolor='#58a6ff', alpha=0.65, label='NSFW Recall'),
    Patch(facecolor='#58a6ff', alpha=0.35, label='SFW Accuracy'),
]
ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.2,
           facecolor='#161b22', edgecolor='#30363d')

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 110)
ax1.set_ylabel('Score (%)')
ax1.set_title('Detection Performance', fontweight='bold', pad=12)
ax1.grid(axis='y', alpha=0.3, linestyle='-')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 高亮 v2.0 bar
ax1.annotate('96% NSFW\nRecall',
            xy=(0, nsfw_recall[0]), xytext=(0.38, 78),
            ha='center', fontsize=10, fontweight='bold',
            color='#58a6ff',
            arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.5,
                          connectionstyle='arc3,rad=-0.2'))

# ===== RIGHT: Speed =====
ax2 = fig.add_subplot(1, 2, 2)

bars = ax2.bar(x, speed, 0.5, color=speed_colors, edgecolor='#30363d',
               linewidth=0.5)

for bar, s, model in zip(bars, speed, models):
    label = f'{s:.2f}s' if s > 0.5 else f'{s*1000:.0f}ms'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
             label, ha='center', fontsize=11, fontweight='bold', color='#e6edf3')

# v2 speedup annotation
ax2.annotate('2.2× faster\nthan v1.0',
            xy=(0, speed[0]), xytext=(0.7, 2.8),
            ha='center', fontsize=11, fontweight='bold',
            color='#f0883e',
            arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.5,
                          connectionstyle='arc3,rad=-0.15'))

# v1 slowdown
ax2.annotate('16× slower\nthan YOLO',
            xy=(1, speed[1]), xytext=(1.7, 3.8),
            ha='center', fontsize=10, fontweight='bold',
            color='#8b949e',
            arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.2,
                          connectionstyle='arc3,rad=-0.15'))

ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylabel('Inference Time (seconds)')
ax2.set_title('Inference Speed', fontweight='bold', pad=12)
ax2.grid(axis='y', alpha=0.3, linestyle='-')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, 5)

fig.suptitle('FlowNSFW — Production Benchmark', fontsize=18, fontweight='bold',
             color='#e6edf3', y=1.03)

# 底部数据说明
fig.text(0.5, -0.02,
         '45 video validation set (25 NSFW + 20 SFW) | A10 24GB | Mamba2 CUDA backend',
         ha='center', fontsize=9, color='#8b949e', style='italic',
         transform=fig.transFigure)

plt.tight_layout()
plt.savefig('assets/performance_comparison.png', dpi=200, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.savefig('assets/performance_comparison.pdf', dpi=200, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("Charts saved!")
