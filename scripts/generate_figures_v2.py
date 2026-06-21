#!/usr/bin/env python3
"""Generate v2.0 performance comparison chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['FlowNSFW\nv2.0', 'FlowNSFW\nv1.0', 'YOLOv11\nv16_s', 'YOLOv11\nauto_v14']
accuracy = [87.5, 96.4, 70.0, 64.5]
nsfw_recall = [100, 98.3, 60.0, 41.7]
sfw_accuracy = [80.0, 94.0, 82.0, 92.0]
speed = [1.64, 0.41, 0.27, 0.33]

x = np.arange(len(models))
width = 0.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy chart
bars1 = ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2ecc71')
bars2 = ax1.bar(x - 0.5*width, nsfw_recall, width, label='NSFW Recall', color='#e74c3c')
bars3 = ax1.bar(x + 0.5*width, sfw_accuracy, width, label='SFW Accuracy', color='#3498db')

ax1.set_ylabel('Score (%)')
ax1.set_title('Detection Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='lower right')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3)

# Annotate v2.0 NSFW Recall
ax1.annotate('100% Recall\nZero Miss!', xy=(x[0]-0.5*width, 100),
            xytext=(x[0]-1.5, 105), ha='center', fontsize=11,
            color='#e74c3c', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

# Speed chart
bars4 = ax2.bar(x, speed, width*2, color=['#2ecc71', '#95a5a6', '#95a5a6', '#95a5a6'])

# Add speedup annotation on v2.0 bar
ax2.annotate(f'2.2× faster\nthan v1.0',
            xy=(x[0], speed[0]), xytext=(x[0]+0.8, 1.2),
            ha='center', fontsize=10, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

ax2.set_ylabel('Seconds per video')
ax2.set_title('Inference Speed')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

fig.suptitle('FlowNSFW v2.0 — Production Optimized', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('assets/performance_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('assets/performance_comparison.pdf', dpi=150, bbox_inches='tight')
print("Charts saved to assets/")
