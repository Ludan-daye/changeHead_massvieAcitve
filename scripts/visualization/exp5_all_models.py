#!/usr/bin/env python3
"""
实验5可视化 - 所有模型
V矩阵消融实验
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path('PROJECT_ROOT/results/experiments/exp5')
OUTPUT_BASE = Path('PROJECT_ROOT/results/plot_results/exp5_figures')

# 模型配置
MODEL_CONFIGS = {
    'gpt2': {'display_name': 'GPT-2'},
    'bloom_7b1': {'display_name': 'BLOOM-7B1'},
    'falcon_7b': {'display_name': 'Falcon-7B'},
    'gptj_6b': {'display_name': 'GPT-J-6B'},
    'mistral_7b_v03': {'display_name': 'Mistral-7B'},
    'opt_6.7b': {'display_name': 'OPT-6.7B'},
    'qwen2.5_7b': {'display_name': 'Qwen2.5-7B'},
    'llama2_13b': {'display_name': 'LLaMA2-13B'}
}


def load_model_data(model_id):
    """加载模型数据"""
    data_file = DATA_DIR / model_id / 'v_ablation_results.json'
    if not data_file.exists():
        return None

    with open(data_file, 'r') as f:
        return json.load(f)


def create_ablation_comparison(model_id, display_name, data, output_dir):
    """图1: V方向消融效果对比"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    k_values = data['k_values']
    critical_layer = data['critical_layer']
    sigma_ratio = data['sigma_ratio']

    remove_changes = [data['ablation_results']['remove_top_k'][str(k)]['change_percent'] for k in k_values]
    keep_changes = [data['ablation_results']['keep_top_k'][str(k)]['change_percent'] for k in k_values]

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

    # 添加平滑趋势线
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
    filename = f'{display_name}_V_Matrix_Ablation_Effects_Layer{critical_layer}_sigma{sigma_ratio:.2f}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    return filename


def create_ma_trend(model_id, display_name, data, output_dir):
    """图2: MA值变化趋势图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    k_values = data['k_values']
    critical_layer = data['critical_layer']
    baseline_ma = data['baseline']['top1_avg']

    remove_ma_values = [data['ablation_results']['remove_top_k'][str(k)]['top1_avg'] for k in k_values]
    keep_ma_values = [data['ablation_results']['keep_top_k'][str(k)]['top1_avg'] for k in k_values]

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
    filename = f'{display_name}_MA_Value_Trends_under_V_Ablation_Layer{critical_layer}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    return filename


def create_energy_vs_retention(model_id, display_name, data, output_dir):
    """图3: 累积奇异值能量 vs MA保留率"""
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    k_values = data['k_values']
    critical_layer = data['critical_layer']
    baseline_ma = data['baseline']['top1_avg']

    keep_ma_values = [data['ablation_results']['keep_top_k'][str(k)]['top1_avg'] for k in k_values]

    # 计算累积奇异值占比
    sv_top10 = data['singular_values_top10']
    total_sv_approx = sum(sv_top10) * 100
    cumulative_sv_ratios = []
    for k in k_values:
        if k <= 10:
            cumulative_sv_ratios.append(sum(sv_top10[:k]) / total_sv_approx * 100)
        else:
            ratio = data['ablation_results']['keep_top_k'][str(k)]['kept_sigma_ratio']
            cumulative_sv_ratios.append(ratio * 100)

    # MA保留率
    ma_retention_rates = [(keep_ma / baseline_ma * 100) for keep_ma in keep_ma_values]

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

    # 动态设置Y轴范围
    max_retention = max(ma_retention_rates)
    if max_retention > 110:
        ax2.set_ylim(0, max_retention * 1.1)
    else:
        ax2.set_ylim(0, 110)

    # 添加数值标签
    for i, rate in enumerate(ma_retention_rates):
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
    filename = f'{display_name}_SV_Energy_vs_MA_Retention_Keep_Top-k_Layer{critical_layer}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    return filename


def create_v1_importance(model_id, display_name, data, output_dir):
    """图4: V1主方向重要性"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    critical_layer = data['critical_layer']
    sigma_ratio = data['sigma_ratio']
    baseline_ma = data['baseline']['top1_avg']

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

    # 添加数值标签
    for bar, val, cat in zip(bars, values, categories):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

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
    filename = f'{display_name}_V1_Importance_sigma{sigma_ratio:.2f}_Layer{critical_layer}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    return filename


def process_model(model_id):
    """处理单个模型"""
    config = MODEL_CONFIGS.get(model_id)
    if not config:
        print(f"❌ 未知模型: {model_id}")
        return False

    display_name = config['display_name']

    # 加载数据
    data = load_model_data(model_id)
    if data is None:
        print(f"❌ {display_name}: 数据文件不存在")
        return False

    # 创建输出目录
    output_dir = OUTPUT_BASE / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"处理模型: {display_name}")
    print(f"{'='*80}")
    print(f"Critical Layer: {data['critical_layer']}")
    print(f"σ₁/σ₂ Ratio: {data['sigma_ratio']:.2f}")
    print(f"Baseline MA: {data['baseline']['top1_avg']:.1f}")
    print()

    print("生成图表:")
    try:
        f1 = create_ablation_comparison(model_id, display_name, data, output_dir)
        print(f"  ✓ 图1: {f1}")

        f2 = create_ma_trend(model_id, display_name, data, output_dir)
        print(f"  ✓ 图2: {f2}")

        f3 = create_energy_vs_retention(model_id, display_name, data, output_dir)
        print(f"  ✓ 图3: {f3}")

        f4 = create_v1_importance(model_id, display_name, data, output_dir)
        print(f"  ✓ 图4: {f4}")

        print(f"\n✅ {display_name} 完成")
        return True
    except Exception as e:
        print(f"\n❌ {display_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    models = ['gpt2', 'bloom_7b1', 'falcon_7b', 'gptj_6b',
              'mistral_7b_v03', 'opt_6.7b', 'qwen2.5_7b', 'llama2_13b']

    print(f"\n{'='*80}")
    print(f"实验5: V矩阵消融实验 - 生成所有模型图表")
    print(f"{'='*80}\n")

    success_count = 0
    for model in models:
        if process_model(model):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"完成统计: {success_count}/{len(models)} 个模型成功")
    print(f"保存位置: {OUTPUT_BASE}")
    print(f"{'='*80}\n")
