#!/usr/bin/env python3
"""
实验7可视化示例 - Mistral-7B (按照示例风格)
方向与幅度分解归因分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
MODEL = 'mistral_7b_v03'
MODEL_DISPLAY = 'Mistral-7B'
DATA_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp7')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp7_figures') / MODEL
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

# 示例图的配色
COLOR_BLUE = '#7BA5D5'
COLOR_RED = '#D97E73'
COLOR_BLUE_DARK = '#5B85B5'
COLOR_RED_DARK = '#B95E53'


# ============================================================
# 图1: MA值对比 - 分组柱状图 (Baseline vs Ablations)
# ============================================================
def create_ma_comparison():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Baseline', 'Ablate\nDirection', 'Ablate\nMagnitude', 'Ablate\nBoth']
    values = [baseline, ablate_dir, ablate_mag, ablate_both]

    # 分为两组：保留 vs 消融
    group1_values = [baseline, 0, 0, 0]  # 蓝色
    group2_values = [0, ablate_dir, ablate_mag, ablate_both]  # 红色

    x = np.arange(len(categories))
    width = 0.45

    # 绘制柱子
    bars1 = ax.bar(x, group1_values, width, label='Baseline',
                   color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, group2_values, width, label='Ablated',
                   color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标签
    for i, val in enumerate(values):
        if val > 0:
            ax.text(i, val + max(values)*0.02, f'{val:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加虚线趋势
    if len(values) >= 3:
        x_smooth = np.linspace(0, len(categories)-1, 100)
        spl = make_interp_spline(x, values, k=2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, color=COLOR_RED_DARK, linestyle='--',
               linewidth=2.5, alpha=0.7, label='MA Trend')

    ax.set_ylabel('MA Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, max(values) * 1.2)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_MA_Comparison_4Conditions_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图1: {filename}")


# ============================================================
# 图2: 归因分解 - 分组柱状图 (Direction vs Magnitude)
# ============================================================
def create_attribution_grouped():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Direction', 'Magnitude', 'Interaction']
    effects_abs = [dir_effect, mag_effect, int_effect]
    effects_pct = [dir_pct, mag_pct, int_pct]

    x = np.arange(len(categories))
    width = 0.25

    # 绘制分组柱状图
    bars1 = ax.bar(x - width/2, effects_abs, width, label='Absolute Effect',
                   color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)

    # 右侧用百分比（归一化到同一尺度）
    scale_factor = max(effects_abs) / max([abs(p) for p in effects_pct]) if max([abs(p) for p in effects_pct]) > 0 else 1
    effects_pct_scaled = [p * scale_factor for p in effects_pct]
    bars2 = ax.bar(x + width/2, effects_pct_scaled, width, label='Percentage (scaled)',
                   color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, effects_abs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(effects_abs)*0.02,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i, (bar, val) in enumerate(zip(bars2, effects_pct)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(effects_abs)*0.02,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加分布曲线
    if len(effects_abs) >= 2:
        x_smooth = np.linspace(-width/2, len(categories)-1+width/2, 100)
        # 绝对值曲线
        spl1 = make_interp_spline(x - width/2, effects_abs, k=2)
        y_smooth1 = spl1(np.linspace(x[0] - width/2, x[-1] - width/2, 100))
        ax.plot(np.linspace(x[0] - width/2, x[-1] - width/2, 100), y_smooth1,
               color=COLOR_BLUE_DARK, linestyle='--', linewidth=2, alpha=0.7,
               label='Absolute Trend')

        # 百分比曲线
        spl2 = make_interp_spline(x + width/2, effects_pct_scaled, k=2)
        y_smooth2 = spl2(np.linspace(x[0] + width/2, x[-1] + width/2, 100))
        ax.plot(np.linspace(x[0] + width/2, x[-1] + width/2, 100), y_smooth2,
               color=COLOR_RED_DARK, linestyle='--', linewidth=2, alpha=0.7,
               label='Percentage Trend')

    ax.set_ylabel('Effect Magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attribution Component', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Attribution_Grouped_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图2: {filename}")


# ============================================================
# 图3: 消融效果对比 - 分组柱状图
# ============================================================
def create_ablation_comparison():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Direction', 'Magnitude', 'Both']
    baseline_vals = [baseline, baseline, baseline]
    ablated_vals = [ablate_dir, ablate_mag, ablate_both]

    x = np.arange(len(categories))
    width = 0.25

    # 绘制分组柱状图
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, ablated_vals, width, label='After Ablation',
                   color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, val in zip(bars2, ablated_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加分布曲线
    x_smooth = np.linspace(x[0] - width/2, x[-1] - width/2, 50)
    ax.plot(x_smooth, [baseline]*len(x_smooth),
           color=COLOR_BLUE_DARK, linestyle='--', linewidth=2.5, alpha=0.7,
           label='Baseline Level')

    x_smooth2 = np.linspace(x[0] + width/2, x[-1] + width/2, 100)
    spl = make_interp_spline(x + width/2, ablated_vals, k=2)
    y_smooth = spl(x_smooth2)
    ax.plot(x_smooth2, y_smooth,
           color=COLOR_RED_DARK, linestyle='--', linewidth=2.5, alpha=0.7,
           label='Ablation Trend')

    ax.set_ylabel('MA Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ablation Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, max(baseline_vals + ablated_vals) * 1.15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Ablation_Comparison_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图3: {filename}")


# ============================================================
# 图4: 归因百分比 - 单柱状图 + 趋势线
# ============================================================
def create_attribution_percentage():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Direction\nContribution', 'Magnitude\nContribution', 'Interaction\nEffect']
    percentages = [dir_pct, mag_pct, int_pct]

    x = np.arange(len(categories))
    width = 0.4

    # 根据正负值选择颜色
    colors = [COLOR_BLUE if p >= 0 else COLOR_RED for p in percentages]

    bars = ax.bar(x, percentages, width, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1)

    # 100%参考线
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2,
              label='100% Baseline', alpha=0.6)

    # 添加数值标签
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            offset = max([abs(p) for p in percentages]) * 0.03
        else:
            va = 'top'
            offset = -max([abs(p) for p in percentages]) * 0.03
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
               f'{pct:.1f}%', ha='center', va=va, fontsize=11, fontweight='bold')

    # 添加趋势线
    if len(percentages) >= 2:
        x_smooth = np.linspace(0, len(categories)-1, 100)
        spl = make_interp_spline(x, percentages, k=2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, color=COLOR_RED_DARK, linestyle='--',
               linewidth=2.5, alpha=0.7, label='Attribution Trend')

    ax.set_ylabel('Attribution Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis='y')

    # 动态Y轴
    max_pct = max([abs(p) for p in percentages])
    ax.set_ylim(-max_pct*0.2, max(max_pct, 100) * 1.2)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Attribution_Percentage_Layer{layer}_sigma{sigma_ratio:.2f}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图4: {filename}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"生成 {MODEL_DISPLAY} 的 Exp7 可视化 (示例风格)")
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
    create_attribution_grouped()
    create_ablation_comparison()
    create_attribution_percentage()

    print(f"\n{'='*80}")
    print(f"✅ {MODEL_DISPLAY} 完成！")
    print(f"保存位置: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
