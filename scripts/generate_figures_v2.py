#!/usr/bin/env python3
"""生成最终三模型对比图表."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 三模型最终数据
models = ['FlowNSFW\nv2.0 (Ours)', 'FlowNSFW\nv1.0', 'YOLOv11\nDetect']
accuracy = [93.3, 71.1, 57.8]
nsfw_recall = [96.0, 48.0, 24.0]
sfw_accuracy = [90.0, 100.0, 100.0]
speed = [1.62, 3.51, 0.22]

x = np.arange(len(models))
width = 0.22

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Chart 1: Accuracy Metrics ---
bars1 = ax1.bar(x - width, accuracy, width, label='Accuracy', color='#2ecc71')
bars2 = ax1.bar(x, nsfw_recall, width, label='NSFW Recall', color='#e74c3c')
bars3 = ax1.bar(x + width, sfw_accuracy, width, label='SFW Accuracy', color='#3498db')

ax1.set_ylabel('Score (%)')
ax1.set_title('Detection Performance (45 videos)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim(0, 112)
ax1.grid(axis='y', alpha=0.3)

# 标注 v2.0 的关键数字
ax1.annotate('96% Recall', xy=(x[0], nsfw_recall[0]),
            xytext=(x[0]-0.4, 105), ha='center', fontsize=10,
            color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# 标注 v1 和 YOLO 的 NSFW Recall 下降
ax1.annotate('48%', xy=(x[1], nsfw_recall[1]),
            xytext=(x[1]+0.3, 53), ha='center', fontsize=9, color='#c0392b')
ax1.annotate('24%', xy=(x[2], nsfw_recall[2]),
            xytext=(x[2]+0.3, 29), ha='center', fontsize=9, color='#c0392b')

# --- Chart 2: Speed ---
colors = ['#2ecc71', '#e67e22', '#95a5a6']
bars4 = ax2.bar(x, speed, width*2.5, color=colors)

# v2 speedup
ax2.annotate(f'2.2x faster\nthan v1.0', xy=(x[0], speed[0]),
            xytext=(x[0]+0.6, 2.5), ha='center', fontsize=10,
            fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

for bar, s in zip(bars4, speed):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{s:.2f}s' if s > 0.5 else f'{s*1000:.0f}ms', ha='center', fontsize=9)

ax2.set_ylabel('Seconds per video')
ax2.set_title('Inference Speed')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('FlowNSFW v2.0 — Final Benchmark', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('assets/performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('assets/performance_comparison.pdf', dpi=150, bbox_inches='tight')
print("Final charts saved!")
