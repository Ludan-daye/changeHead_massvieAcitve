#!/usr/bin/env python3
"""
实验6可视化 - Qwen2.5-7B
V矩阵Keep Top-k实验 - 细粒度分析
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
MODEL_DISPLAY = 'Qwen2.5-7B'
DATA_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp6')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp6_figures') / MODEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
with open(DATA_DIR / MODEL / 'summary.json', 'r') as f:
    data = json.load(f)

k_values = data['k_values']
layer = data['layer']
baseline = data['baseline_mean']
sigma_ratio = data['svd_info']['sigma_ratio']

# 提取各k值的MA数据
ma_values = [data['results_by_k'][str(k)]['mean'] for k in k_values]
ma_stds = [data['results_by_k'][str(k)]['std'] for k in k_values]

# 恢复率
recovery_rates = [(ma / baseline * 100) for ma in ma_values]

# 边际贡献
marginal_contribs = [data['marginal_contributions'][str(k)]['marginal_contribution'] for k in k_values]

# 累积奇异值占比
cumulative_sv_ratios = [data['results_by_k'][str(k)]['kept_sigma_ratio'] * 100 for k in k_values]


# ============================================================
# 图1: MA恢复曲线（Keep Top-k）
# ============================================================
def create_ma_recovery_curve():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Baseline参考线
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=2,
               label=f'Baseline (MA={baseline:.1f})', alpha=0.6)

    # 平滑曲线
    x_positions = np.arange(len(k_values))
    x_smooth = np.linspace(0, len(k_values)-1, 300)
    spl = make_interp_spline(x_positions, ma_values, k=3)
    y_smooth = spl(x_smooth)

    ax.plot(x_smooth, y_smooth, color='#5B9BD5', linewidth=3,
            label='Keep Top-k', alpha=0.9)
    ax.scatter(x_positions, ma_values, color='#5B9BD5', s=100,
               edgecolors='white', linewidths=2, zorder=5)

    ax.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MA Value', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # 设置合适的Y轴范围
    y_min = min(ma_values)
    y_max = max(ma_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_MA_Recovery_Curve_Keep_Top-k_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图1: {filename}")


# ============================================================
# 图2: MA恢复率曲线（柱状图）
# ============================================================
def create_recovery_rate_chart():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x_positions = np.arange(len(k_values))
    colors = ['#ED7D31' if r < 100 else '#70AD47' for r in recovery_rates]

    bars = ax.bar(x_positions, recovery_rates, color=colors, alpha=0.7,
                  edgecolor='none', width=0.6)

    # 100%参考线
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2,
               label='100% Recovery', alpha=0.6)

    ax.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MA Recovery Rate (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.set_ylim(0, max(recovery_rates) * 1.1)

    # 添加数值标签
    for bar, rate in zip(bars, recovery_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_MA_Recovery_Rate_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图2: {filename}")


# ============================================================
# 图3: 边际贡献分析（瀑布图 + 趋势线）
# ============================================================
def create_marginal_contribution_chart():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x_positions = np.arange(len(k_values))
    colors = ['#70AD47' if mc > 0 else '#C5504B' for mc in marginal_contribs]

    bars = ax.bar(x_positions, marginal_contribs, color=colors, alpha=0.75,
                  edgecolor='black', linewidth=0.8, width=0.5)

    # 添加平滑趋势线
    if len(x_positions) >= 3:
        x_smooth = np.linspace(0, len(k_values)-1, 200)
        spl = make_interp_spline(x_positions, marginal_contribs, k=2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, color='#1F4788', linestyle='--',
                linewidth=1.5, alpha=0.7, label='Trend')

    # 0基准线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Marginal MA Contribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=11)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y')

    # 添加数值标签
    for bar, mc in zip(bars, marginal_contribs):
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            offset = 0.5
        else:
            va = 'top'
            offset = -0.5
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{mc:.1f}', ha='center', va=va, fontsize=9, fontweight='bold')

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Marginal_Contribution_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图3: {filename}")


# ============================================================
# 图4: 累积奇异值 vs MA恢复（双Y轴）
# ============================================================
def create_sv_vs_recovery():
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    x_positions = np.arange(len(k_values))

    # 左Y轴：累积奇异值占比
    color1 = '#5B9BD5'
    ax1.bar(x_positions, cumulative_sv_ratios, alpha=0.6, color=color1, width=0.4)
    ax1.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative SV Energy (%)', fontsize=13,
                   fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([str(k) for k in k_values], fontsize=11)

    # 右Y轴：MA恢复率
    ax2 = ax1.twinx()
    color2 = '#ED7D31'
    ax2.plot(x_positions, recovery_rates, 'o-', color=color2, linewidth=3,
             markersize=9, markerfacecolor='white', markeredgewidth=2.5)
    ax2.set_ylabel('MA Recovery Rate (%)', fontsize=13, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.4)

    # 动态Y轴范围
    max_recovery = max(recovery_rates)
    if max_recovery > 110:
        ax2.set_ylim(0, max_recovery * 1.1)
    else:
        ax2.set_ylim(0, 110)

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=color1, alpha=0.6, label='Cumulative SV Energy'),
        Line2D([0], [0], color=color2, linewidth=3, marker='o',
               markerfacecolor='white', markeredgewidth=2, markersize=8,
               label='MA Recovery Rate')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    filename = f'{MODEL_DISPLAY}_SV_Energy_vs_MA_Recovery_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图4: {filename}")


# ============================================================
# 图5: 前5个V方向的增量贡献（放大分析）
# ============================================================
def create_early_contributions():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 只看前4个k值：1,2,3,5
    early_k_indices = [0, 1, 2, 3]  # k=1,2,3,5
    early_k_values = [k_values[i] for i in early_k_indices]
    early_ma_values = [ma_values[i] for i in early_k_indices]
    early_recovery = [recovery_rates[i] for i in early_k_indices]

    x_positions = np.arange(len(early_k_values))

    # 阶梯式累积图
    ax.plot(x_positions, early_ma_values, 'o-', color='#5B9BD5', linewidth=3,
            markersize=12, markerfacecolor='white', markeredgewidth=3, label='Cumulative MA')
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=2,
               label=f'Baseline ({baseline:.1f})', alpha=0.6)

    # 添加数值标签（MA值 + 恢复率）
    for i, (k, ma, rec) in enumerate(zip(early_k_values, early_ma_values, early_recovery)):
        ax.text(i, ma + 5, f'{ma:.1f}\n({rec:.1f}%)',
                ha='center', fontsize=10, fontweight='bold', color='#5B9BD5')

    ax.set_xlabel('Number of V Components (k)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MA Value', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'k={k}' for k in early_k_values], fontsize=11)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_ylim(min(early_ma_values) * 0.95, max(early_ma_values) * 1.08)

    plt.tight_layout()
    filename = f'{MODEL_DISPLAY}_Early_V_Contributions_Layer{layer}'
    fig.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图5: {filename}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"生成 {MODEL_DISPLAY} 的 Exp6 可视化")
    print(f"{'='*80}\n")

    print(f"模型信息:")
    print(f"  - Layer: {layer}")
    print(f"  - σ₁/σ₂ Ratio: {sigma_ratio:.2f}")
    print(f"  - Baseline MA: {baseline:.2f}")
    print(f"  - k=1 MA: {ma_values[0]:.2f} ({recovery_rates[0]:.1f}%)")
    print(f"  - k=100 MA: {ma_values[-1]:.2f} ({recovery_rates[-1]:.1f}%)")
    print()

    print("生成图表:")
    create_ma_recovery_curve()
    create_recovery_rate_chart()
    create_marginal_contribution_chart()
    create_sv_vs_recovery()
    create_early_contributions()

    print(f"\n{'='*80}")
    print(f"✅ {MODEL_DISPLAY} 完成！")
    print(f"保存位置: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
