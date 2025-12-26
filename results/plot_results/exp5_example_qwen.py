#!/usr/bin/env python3
"""
实验5可视化示例 - Qwen2.5-7B
V矩阵消融实验
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
MODEL = 'qwen2.5_7b'
DATA_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp5')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp5_figures') / MODEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
with open(DATA_DIR / MODEL / 'v_ablation_results.json', 'r') as f:
    data = json.load(f)

k_values = data['k_values']
baseline_ma = data['baseline']['top1_avg']
sigma_ratio = data['sigma_ratio']
critical_layer = data['critical_layer']

# 提取数据
remove_changes = [data['ablation_results']['remove_top_k'][str(k)]['change_percent'] for k in k_values]
keep_changes = [data['ablation_results']['keep_top_k'][str(k)]['change_percent'] for k in k_values]

remove_ma_values = [data['ablation_results']['remove_top_k'][str(k)]['top1_avg'] for k in k_values]
keep_ma_values = [data['ablation_results']['keep_top_k'][str(k)]['top1_avg'] for k in k_values]

# 计算累积奇异值占比
sv_top10 = data['singular_values_top10']
total_sv_approx = sum(sv_top10) * 100  # 粗略估计总和
cumulative_sv_ratios = []
for k in k_values:
    if k <= 10:
        cumulative_sv_ratios.append(sum(sv_top10[:k]) / total_sv_approx * 100)
    else:
        # 对于k>10，使用removed_sigma_ratio
        ratio = data['ablation_results']['keep_top_k'][str(k)]['kept_sigma_ratio']
        cumulative_sv_ratios.append(ratio * 100)

# MA保留率（基于Keep Top-k）
ma_retention_rates = [(keep_ma / baseline_ma * 100) for keep_ma in keep_ma_values]


# ============================================================
# 图1: V方向消融效果对比（双面柱状图） - 参考例子风格
# ============================================================
def create_ablation_comparison():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x = np.arange(len(k_values))
    width = 0.25

    # Remove Top-k (取绝对值)
    remove_abs = [abs(c) for c in remove_changes]
    bars1 = ax.bar(x - width/2, remove_abs, width,
                   label='Remove Top-k', color='#5B9BD5', alpha=0.7, edgecolor='none')

    # Keep Top-k (取绝对值)
    keep_abs = [abs(c) for c in keep_changes]
    bars2 = ax.bar(x + width/2, keep_abs, width,
                   label='Keep Top-k', color='#ED7D31', alpha=0.7, edgecolor='none')

    # 添加平滑趋势线（仿照例子图的虚线）
    if len(x) >= 3:
        x_smooth = np.linspace(x[0] - width/2, x[-1] - width/2, 100)
        spl1 = make_interp_spline(x - width/2, remove_abs, k=2)
        y1_smooth = spl1(x_smooth)
        ax.plot(x_smooth, y1_smooth, '--', color='#5B9BD5', linewidth=2, alpha=0.8)

        x_smooth2 = np.linspace(x[0] + width/2, x[-1] + width/2, 100)
        spl2 = make_interp_spline(x + width/2, keep_abs, k=2)
        y2_smooth = spl2(x_smooth2)
        ax.plot(x_smooth2, y2_smooth, '--', color='#ED7D31', linewidth=2, alpha=0.8)

    ax.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MA Reduction (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}' for k in k_values], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(max(remove_abs), max(keep_abs)) * 1.15)

    plt.tight_layout()
    filename = f'Qwen2.5-7B_V_Matrix_Ablation_Effects_Layer{critical_layer}_sigma{sigma_ratio:.2f}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图1: {filename}")


# ============================================================
# 图2: MA值变化趋势图（折线图 - 平滑）
# ============================================================
def create_ma_trend():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Baseline
    ax.axhline(y=baseline_ma, color='gray', linestyle='--', linewidth=2,
               label=f'Baseline (MA={baseline_ma:.1f})', alpha=0.6)

    # 使用样条插值平滑曲线
    x_log = np.log10(k_values)
    x_smooth = np.linspace(x_log[0], x_log[-1], 300)

    # Remove Top-k 平滑曲线
    spl_remove = make_interp_spline(x_log, remove_ma_values, k=3)
    y_remove_smooth = spl_remove(x_smooth)
    ax.plot(10**x_smooth, y_remove_smooth, color='#5B9BD5', linewidth=3,
            label='Remove Top-k', alpha=0.9)
    ax.scatter(k_values, remove_ma_values, color='#5B9BD5', s=80,
               edgecolors='white', linewidths=2, zorder=5)

    # Keep Top-k 平滑曲线
    spl_keep = make_interp_spline(x_log, keep_ma_values, k=3)
    y_keep_smooth = spl_keep(x_smooth)
    ax.plot(10**x_smooth, y_keep_smooth, color='#ED7D31', linewidth=3,
            label='Keep Top-k', alpha=0.9)
    ax.scatter(k_values, keep_ma_values, color='#ED7D31', s=80,
               edgecolors='white', linewidths=2, zorder=5)

    ax.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MA Value (Top-1)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    filename = f'Qwen2.5-7B_MA_Value_Trends_under_V_Ablation_Layer{critical_layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图2: {filename}")


# ============================================================
# 图3: 累积奇异值能量 vs MA保留率（双Y轴图 - 修复标签错位）
# ============================================================
def create_energy_vs_retention():
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    x = np.arange(len(k_values))

    # 左Y轴：累积奇异值占比
    color1 = '#5B9BD5'
    ax1.bar(x, cumulative_sv_ratios, alpha=0.6, color=color1, width=0.4)
    ax1.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative SV Energy (%)', fontsize=13,
                   fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{k}' for k in k_values], fontsize=11)

    # 右Y轴：MA保留率
    ax2 = ax1.twinx()
    color2 = '#ED7D31'
    ax2.plot(x, ma_retention_rates, 'o-', color=color2, linewidth=3,
             markersize=9, markerfacecolor='white', markeredgewidth=2.5)
    ax2.set_ylabel('MA Retention Rate (%)', fontsize=13, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    ax2.set_ylim(0, 110)

    # 添加数值标签（MA保留率）- 调整位置避免错位
    for i, rate in enumerate(ma_retention_rates):
        # 根据数值位置智能调整标签位置
        if rate < 50:
            va = 'bottom'
            offset = 5
        else:
            va = 'top'
            offset = -5
        ax2.text(i, rate + offset, f'{rate:.1f}%', ha='center', va=va,
                fontsize=9, color=color2, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=color1, alpha=0.6, label='Cumulative SV Energy'),
        Line2D([0], [0], color=color2, linewidth=3, marker='o',
               markerfacecolor='white', markeredgewidth=2, markersize=8,
               label='MA Retention Rate')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    filename = f'Qwen2.5-7B_SV_Energy_vs_MA_Retention_Keep_Top-k_Layer{critical_layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图3: {filename}")


# ============================================================
# 图4: V1主方向重要性（参考例子风格）
# ============================================================
def create_v1_importance():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 提取v1数据
    remove_v1_change = data['ablation_results']['remove_top_k']['1']['change_percent']
    keep_v1_change = data['ablation_results']['keep_top_k']['1']['change_percent']
    remove_v1_ma = data['ablation_results']['remove_top_k']['1']['top1_avg']
    keep_v1_ma = data['ablation_results']['keep_top_k']['1']['top1_avg']

    categories = ['Baseline', 'Remove v₁', 'Keep only v₁']
    values = [baseline_ma, remove_v1_ma, keep_v1_ma]
    colors = ['#A5A5A5', '#5B9BD5', '#ED7D31']

    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='none', width=0.45)

    ax.set_ylabel('MA Value (Top-1)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)

    # 添加数值标签（在柱子顶部）
    for bar, val, cat in zip(bars, values, categories):
        height = bar.get_height()
        # 显示MA值
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 显示变化百分比（除了baseline）
        if cat == 'Remove v₁':
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{remove_v1_change:.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        elif cat == 'Keep only v₁':
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{keep_v1_change:.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    plt.tight_layout()
    filename = f'Qwen2.5-7B_V1_Importance_sigma{sigma_ratio:.2f}_Layer{critical_layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图4: {filename}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"生成 Qwen2.5-7B 的 Exp5 可视化")
    print(f"{'='*80}\n")

    print(f"模型信息:")
    print(f"  - Critical Layer: {critical_layer}")
    print(f"  - σ₁/σ₂ Ratio: {sigma_ratio:.2f}")
    print(f"  - Baseline MA: {baseline_ma:.2f}")
    print(f"  - Remove v₁ effect: {remove_changes[0]:.1f}%")
    print(f"  - Keep only v₁: {keep_changes[0]:.1f}%")
    print()

    print("生成图表:")
    create_ablation_comparison()
    create_ma_trend()
    create_energy_vs_retention()
    create_v1_importance()

    print(f"\n{'='*80}")
    print(f"✅ Qwen2.5-7B 完成！")
    print(f"保存位置: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
