#!/usr/bin/env python3
"""
实验7可视化示例 - Mistral-7B
方向与幅度分解归因分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
MODEL = 'mistral_7b_v03'
MODEL_DISPLAY = 'Mistral-7B'
DATA_DIR = Path('PROJECT_ROOT/results/experiments/exp7')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp7_figures') / MODEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
with open(DATA_DIR / MODEL / 'summary.json', 'r') as f:
    data = json.load(f)

layer = data['layer']
baseline = data['attribution']['baseline']
ablate_dir = data['attribution']['ablate_direction_mean']
ablate_mag = data['attribution']['ablate_magnitude_mean']
ablate_both = data['attribution']['ablate_both_mean']

dir_effect = data['attribution']['direction_effect']
mag_effect = data['attribution']['magnitude_effect']
int_effect = data['attribution']['interaction_effect']

dir_pct = data['attribution']['direction_attribution_pct']
mag_pct = data['attribution']['magnitude_attribution_pct']
int_pct = data['attribution']['interaction_pct']

sigma_ratio = data['svd_info']['sigma_ratio']


# ============================================================
# 图1: MA值对比（4种条件）- 纵向柱状图
# ============================================================
def create_ma_comparison():
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    conditions = ['Baseline', 'Ablate\nDirection', 'Ablate\nMagnitude', 'Ablate\nBoth']
    values = [baseline, ablate_dir, ablate_mag, ablate_both]

    # 使用渐变色系（从深到浅）- 参考示例图
    colors = ['#4A5F8C', '#6B8EC4', '#D97E53', '#E6A67C']

    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white',
                   linewidth=1.5, width=0.65)

    # 添加数值标签（在柱子顶部）
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('MA Value', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.35, axis='y', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_MA_Comparison_4Conditions_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图1: {filename}")


# ============================================================
# 图2: 归因分解堆叠图（显示累积效应）
# ============================================================
def create_attribution_stacked():
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # 将归因分解为正负部分
    categories = ['Baseline', 'Cumulative\nAttribution']

    # 计算正向贡献和负向贡献
    positive_contrib = max(0, dir_effect) + max(0, mag_effect) + max(0, int_effect)
    negative_contrib = min(0, dir_effect) + min(0, mag_effect) + min(0, int_effect)

    # 分解各部分
    dir_pos = max(0, dir_effect)
    mag_pos = max(0, mag_effect)
    int_pos = max(0, int_effect)

    # 堆叠数据
    colors_stack = ['#4A5F8C', '#6B8EC4', '#89ABDE']
    labels_stack = ['Direction', 'Magnitude', 'Interaction']

    x = [1]
    bottoms = [0]

    # 绘制堆叠柱
    for i, (val, color, label) in enumerate(zip([dir_pos, mag_pos, int_pos],
                                                  colors_stack, labels_stack)):
        ax.bar(x, val, bottom=bottoms[-1], color=color, alpha=0.85,
               edgecolor='white', linewidth=1.5, width=0.5, label=label)
        # 添加标签
        if val > 1:
            ax.text(x[0], bottoms[-1] + val/2, f'{val:.1f}',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        bottoms.append(bottoms[-1] + val)

    # Baseline参考线
    ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2.5,
               label=f'Baseline ({baseline:.1f})', alpha=0.7)

    ax.set_ylabel('Cumulative Attribution', fontsize=14, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['Attribution\nComponents'], fontsize=12)
    ax.set_xlim(0.4, 1.6)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.35, axis='y', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Attribution_Stacked_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图2: {filename}")


# ============================================================
# 图3: 归因效应对比（分组柱状图 - 绝对值 vs 百分比）
# ============================================================
def create_attribution_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    categories = ['Direction', 'Magnitude', 'Interaction']
    effects_abs = [dir_effect, mag_effect, int_effect]
    effects_pct = [dir_pct, mag_pct, int_pct]

    # 渐变色系
    colors = ['#4A5F8C', '#D97E53', '#7BA05B']

    x = np.arange(len(categories))
    width = 0.6

    # 左图：绝对值
    bars1 = ax1.bar(x, effects_abs, width, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.5)

    # 添加数值标签
    for bar, val in zip(bars1, effects_abs):
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            offset = max(effects_abs) * 0.02
        else:
            va = 'top'
            offset = -max(effects_abs) * 0.02
        ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:.1f}', ha='center', va=va, fontsize=11, fontweight='bold')

    ax1.set_ylabel('Effect on MA (absolute)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.grid(True, alpha=0.35, axis='y', linestyle='-', linewidth=0.7)
    ax1.set_axisbelow(True)
    ax1.set_title('(a) Absolute Effects', fontsize=13, fontweight='bold', pad=10)

    # 右图：百分比
    bars2 = ax2.bar(x, effects_pct, width, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=1.5)

    # 100%参考线
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2,
               label='100% Baseline', alpha=0.6)

    # 添加数值标签
    for bar, val in zip(bars2, effects_pct):
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            offset = max(abs(e) for e in effects_pct) * 0.02
        else:
            va = 'top'
            offset = -max(abs(e) for e in effects_pct) * 0.02
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:.1f}%', ha='center', va=va, fontsize=11, fontweight='bold')

    ax2.set_ylabel('Attribution (%)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.35, axis='y', linestyle='-', linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.set_title('(b) Percentage Attribution', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Attribution_Comparison_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图3: {filename}")


# ============================================================
# 图4: 分解可视化（堆叠/瀑布图）
# ============================================================
def create_decomposition_waterfall():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 瀑布图数据
    labels = ['Baseline', 'Remove\nDirection', 'Remove\nMagnitude', 'Interaction', 'Ablate\nBoth']

    # 计算累积值
    cumulative = [baseline]
    cumulative.append(cumulative[-1] - dir_effect)  # 移除direction效应
    cumulative.append(cumulative[-1] - mag_effect)  # 移除magnitude效应
    cumulative.append(cumulative[-1] + int_effect)  # 加上交互
    # 最后应该等于ablate_both

    colors_waterfall = ['#A5A5A5', '#5B9BD5', '#ED7D31', '#70AD47', '#C5504B']

    # 绘制瀑布图
    x = np.arange(len(labels))

    # Baseline柱
    ax.bar(0, cumulative[0], color=colors_waterfall[0], alpha=0.7, edgecolor='black', linewidth=0.8, width=0.6)

    # 后续柱子（显示变化）
    for i in range(1, len(cumulative)):
        bottom = min(cumulative[i-1], cumulative[i])
        height = abs(cumulative[i] - cumulative[i-1])
        ax.bar(i, height, bottom=bottom, color=colors_waterfall[i], alpha=0.7,
               edgecolor='black', linewidth=0.8, width=0.6)

        # 连接线
        ax.plot([i-0.3, i-0.3], [cumulative[i-1], cumulative[i-1]], 'k--', linewidth=1, alpha=0.3)
        ax.plot([i-0.3, i+0.3], [cumulative[i-1], cumulative[i]], 'k--', linewidth=1, alpha=0.3)

    # 添加数值标签
    for i, val in enumerate(cumulative):
        ax.text(i, val + 1, f'{val:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('MA Value', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(cumulative) * 1.15)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Decomposition_Waterfall_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图4: {filename}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"生成 {MODEL_DISPLAY} 的 Exp7 可视化")
    print(f"{'='*80}\n")

    print(f"模型信息:")
    print(f"  - Layer: {layer}")
    print(f"  - σ₁/σ₂: {sigma_ratio:.2f}")
    print(f"  - Baseline MA: {baseline:.2f}")
    print(f"  - Direction effect: {dir_effect:.2f} ({dir_pct:.1f}%)")
    print(f"  - Magnitude effect: {mag_effect:.2f} ({mag_pct:.1f}%)")
    print(f"  - Interaction: {int_effect:.2f} ({int_pct:.1f}%)")
    print()

    print("生成图表:")
    create_ma_comparison()
    create_attribution_stacked()
    create_attribution_comparison()
    create_decomposition_waterfall()

    print(f"\n{'='*80}")
    print(f"✅ {MODEL_DISPLAY} 完成！")
    print(f"保存位置: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
