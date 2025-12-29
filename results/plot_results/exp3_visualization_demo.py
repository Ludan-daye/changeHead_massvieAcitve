#!/usr/bin/env python3
"""
Exp3 可视化示例脚本 - 使用GPT-2数据演示
生成多种图表类型供用户选择
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# 设置中文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
MODEL = 'gpt2'
EXP3_DIR = Path('PROJECT_ROOT/results/experiments/exp3')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp3_figures/gpt2')

# 读取数据
model_dir = EXP3_DIR / MODEL
with open(model_dir / 'summary.json', 'r') as f:
    summary = json.load(f)

# 提取关键数据
attribution = summary['attribution']
baseline = attribution['baseline']
ablate_u = attribution['ablate_u_mean']
ablate_v = attribution['ablate_v_mean']
ablate_both = attribution['ablate_both_mean']
u_pct = attribution['u_attribution_pct']
v_pct = attribution['v_attribution_pct']
interaction_pct = attribution['interaction_pct']
total_explained = attribution['total_explained']
interpretation = attribution['interpretation']

print(f"Model: {MODEL}")
print(f"Layer: {summary['layer']}")
print(f"Interpretation: {interpretation}")
print(f"U: {u_pct:.2f}%, V: {v_pct:.2f}%, Interaction: {interaction_pct:.2f}%")
print(f"Total Explained: {total_explained:.2f}%")

# ========== 图表1: 归因分解堆叠柱状图 ==========
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

categories = ['U Attribution', 'V Attribution', 'Interaction']
values = [u_pct, v_pct, interaction_pct]
colors = ['#3498db', '#e74c3c', '#f39c12']

bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Attribution Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title(f'{MODEL.upper()} - U×V Attribution Decomposition\nLayer {summary["layer"]} | Mode: {interpretation.capitalize()}',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加总解释度标注
ax.text(0.98, 0.98, f'Total Explained: {total_explained:.1f}%',
        transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{MODEL}_attribution_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / f'{MODEL}_attribution_decomposition.pdf', dpi=300, bbox_inches='tight')
print(f"✓ 图表1已保存: attribution_decomposition")

plt.close()

# ========== 图表2: 消融实验4条柱对比图 ==========
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

conditions = ['Baseline\n(Full W)', 'Ablate U\n(W = I⊗Σ⊗Vᵀ)', 'Ablate V\n(W = U⊗Σ⊗I)', 'Ablate Both\n(W = I⊗Σ⊗I)']
ma_values = [baseline, ablate_u, ablate_v, ablate_both]
colors_ablation = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

bars = ax.bar(conditions, ma_values, color=colors_ablation, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, val in zip(bars, ma_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加变化箭头和百分比
for i in range(1, 4):
    change = ma_values[i] - baseline
    change_pct = (change / baseline) * 100
    ax.annotate('', xy=(i, ma_values[i]), xytext=(0, baseline),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red' if change < 0 else 'green', alpha=0.6))

ax.set_ylabel('Massive Activation Value', fontsize=12, fontweight='bold')
ax.set_title(f'{MODEL.upper()} - Ablation Experiment Results\nLayer {summary["layer"]} | Impact of U and V Matrices',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加解释文本
explanation = (
    f"Mode: {interpretation.capitalize()}\n"
    f"U Effect: {u_pct:.1f}% | V Effect: {v_pct:.1f}%\n"
    f"Interaction: {interaction_pct:.1f}%"
)
ax.text(0.02, 0.98, explanation,
        transform=ax.transAxes,
        ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{MODEL}_ablation_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / f'{MODEL}_ablation_comparison.pdf', dpi=300, bbox_inches='tight')
print(f"✓ 图表2已保存: ablation_comparison")

plt.close()

# ========== 图表3: 三元归因雷达图 ==========
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = fig.add_subplot(111, projection='polar')

# 归一化数据到0-1范围（用于雷达图显示）
max_val = max(abs(u_pct), abs(v_pct), abs(interaction_pct), 10)  # 至少10%作为基准
norm_values = [
    abs(u_pct) / max_val,
    abs(v_pct) / max_val,
    abs(interaction_pct) / max_val
]

# 设置雷达图的角度
categories_radar = ['U Attribution', 'V Attribution', 'U×V Interaction']
N = len(categories_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
norm_values += norm_values[:1]  # 闭合图形
angles += angles[:1]

# 绘制雷达图
ax.plot(angles, norm_values, 'o-', linewidth=2, color='#3498db', label=f'{MODEL.upper()}')
ax.fill(angles, norm_values, alpha=0.25, color='#3498db')

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=11, fontweight='bold')

# 添加实际数值标签
for angle, val, orig_val in zip(angles[:-1], norm_values[:-1], [u_pct, v_pct, interaction_pct]):
    ax.text(angle, val + 0.1, f'{orig_val:.1f}%',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_ylim(0, 1.2)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax.grid(True, alpha=0.3)

plt.title(f'{MODEL.upper()} - U×V Attribution Radar Chart\n'
          f'Layer {summary["layer"]} | Mode: {interpretation.capitalize()} | Total: {total_explained:.1f}%',
          fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{MODEL}_attribution_radar.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / f'{MODEL}_attribution_radar.pdf', dpi=300, bbox_inches='tight')
print(f"✓ 图表3已保存: attribution_radar")

plt.close()

# ========== 图表4: 归因分解瀑布图 ==========
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 瀑布图数据
steps = ['Baseline', 'U Effect', 'V Effect', 'Interaction', 'Total Explained']
# 计算绝对值变化
u_effect = (baseline - ablate_u)
v_effect = (baseline - ablate_v)
interaction_effect = attribution['interaction_effect']

cumulative = [0, u_effect, u_effect + v_effect, u_effect + v_effect + interaction_effect]
effects = [baseline, u_effect, v_effect, interaction_effect, attribution['u_main_effect'] + attribution['v_main_effect'] + attribution['interaction_effect']]

colors_waterfall = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

# 绘制瀑布图
for i, (step, effect, color) in enumerate(zip(steps, effects, colors_waterfall)):
    if i == 0:  # Baseline
        ax.bar(i, effect, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
    elif i == len(steps) - 1:  # Total
        total_effect = cumulative[-1]
        ax.bar(i, total_effect, bottom=0, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
    else:
        ax.bar(i, effect, bottom=cumulative[i-1], color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    if i == 0:
        label_y = effect / 2
    elif i == len(steps) - 1:
        label_y = cumulative[-1] / 2
    else:
        label_y = cumulative[i-1] + effect / 2

    ax.text(i, label_y, f'{effect:.1f}',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xticks(range(len(steps)))
ax.set_xticklabels(steps, fontsize=11, fontweight='bold')
ax.set_ylabel('Effect Magnitude', fontsize=12, fontweight='bold')
ax.set_title(f'{MODEL.upper()} - Attribution Waterfall Chart\nLayer {summary["layer"]} | Cumulative Effects Breakdown',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'{MODEL}_attribution_waterfall.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / f'{MODEL}_attribution_waterfall.pdf', dpi=300, bbox_inches='tight')
print(f"✓ 图表4已保存: attribution_waterfall")

plt.close()

print(f"\n{'='*50}")
print(f"✅ 所有图表已生成完成！")
print(f"保存位置: {OUTPUT_DIR}")
print(f"共生成: 4种图表类型 × 2种格式 = 8个文件")
print(f"{'='*50}")
