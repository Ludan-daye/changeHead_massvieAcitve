#!/usr/bin/env python3
"""
实验7可视化 - 批量生成所有模型
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

# 示例图的配色
COLOR_BLUE = '#7BA5D5'
COLOR_RED = '#D97E73'
COLOR_BLUE_DARK = '#5B85B5'
COLOR_RED_DARK = '#B95E53'

# 模型配置
MODEL_CONFIGS = {
    'gpt2': {'display_name': 'GPT-2'},
    'bloom_7b1': {'display_name': 'BLOOM-7B1'},
    'falcon_7b': {'display_name': 'Falcon-7B'},
    'gptj_6b': {'display_name': 'GPT-J-6B'},
    'opt_7b': {'display_name': 'OPT-7B'},
    'qwen2.5_7b': {'display_name': 'Qwen2.5-7B'},
    'llama2_13b': {'display_name': 'LLaMA2-13B'}
}

DATA_DIR = Path('PROJECT_ROOT/results/experiments/exp7')
OUTPUT_BASE = Path('PROJECT_ROOT/results/plot_results/exp7_figures')


def create_ma_comparison(data, model_display, output_dir):
    """图1: MA值对比"""
    layer = data['layer']
    baseline = data['attribution']['baseline']
    ablate_dir = data['attribution']['ablate_direction_mean']
    ablate_mag = data['attribution']['ablate_magnitude_mean']
    ablate_both = data['attribution']['ablate_both_mean']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Baseline', 'Ablate\nDirection', 'Ablate\nMagnitude', 'Ablate\nBoth']
    values = [baseline, ablate_dir, ablate_mag, ablate_both]

    group1_values = [baseline, 0, 0, 0]
    group2_values = [0, ablate_dir, ablate_mag, ablate_both]

    x = np.arange(len(categories))
    width = 0.45

    bars1 = ax.bar(x, group1_values, width, label='Baseline',
                   color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, group2_values, width, label='Ablated',
                   color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

    for i, val in enumerate(values):
        if val > 0:
            ax.text(i, val + max(values)*0.02, f'{val:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

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
    filename = f'{model_display}_MA_Comparison_4Conditions_Layer{layer}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def create_attribution_grouped(data, model_display, output_dir):
    """图2: 归因分解 - 分组柱状图"""
    layer = data['layer']
    dir_effect = data['attribution']['direction_effect']
    mag_effect = data['attribution']['magnitude_effect']
    int_effect = data['attribution']['interaction_effect']
    dir_pct = data['attribution']['direction_attribution_pct']
    mag_pct = data['attribution']['magnitude_attribution_pct']
    int_pct = data['attribution']['interaction_pct']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Direction', 'Magnitude', 'Interaction']
    effects_abs = [dir_effect, mag_effect, int_effect]
    effects_pct = [dir_pct, mag_pct, int_pct]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width/2, effects_abs, width, label='Absolute Effect',
                   color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)

    scale_factor = max(effects_abs) / max([abs(p) for p in effects_pct]) if max([abs(p) for p in effects_pct]) > 0 else 1
    effects_pct_scaled = [p * scale_factor for p in effects_pct]
    bars2 = ax.bar(x + width/2, effects_pct_scaled, width, label='Percentage (scaled)',
                   color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

    for i, (bar, val) in enumerate(zip(bars1, effects_abs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(effects_abs)*0.02,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i, (bar, val) in enumerate(zip(bars2, effects_pct)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(effects_abs)*0.02,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    if len(effects_abs) >= 2:
        spl1 = make_interp_spline(x - width/2, effects_abs, k=2)
        y_smooth1 = spl1(np.linspace(x[0] - width/2, x[-1] - width/2, 100))
        ax.plot(np.linspace(x[0] - width/2, x[-1] - width/2, 100), y_smooth1,
               color=COLOR_BLUE_DARK, linestyle='--', linewidth=2, alpha=0.7,
               label='Absolute Trend')

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
    filename = f'{model_display}_Attribution_Grouped_Layer{layer}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def create_ablation_comparison(data, model_display, output_dir):
    """图3: 消融效果对比"""
    layer = data['layer']
    baseline = data['attribution']['baseline']
    ablate_dir = data['attribution']['ablate_direction_mean']
    ablate_mag = data['attribution']['ablate_magnitude_mean']
    ablate_both = data['attribution']['ablate_both_mean']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Direction', 'Magnitude', 'Both']
    baseline_vals = [baseline, baseline, baseline]
    ablated_vals = [ablate_dir, ablate_mag, ablate_both]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, ablated_vals, width, label='After Ablation',
                   color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar, val in zip(bars2, ablated_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

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
    filename = f'{model_display}_Ablation_Comparison_Layer{layer}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def create_attribution_percentage(data, model_display, output_dir):
    """图4: 归因百分比"""
    layer = data['layer']
    dir_pct = data['attribution']['direction_attribution_pct']
    mag_pct = data['attribution']['magnitude_attribution_pct']
    int_pct = data['attribution']['interaction_pct']
    sigma_ratio = data['svd_info']['sigma_ratio']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = ['Direction\nContribution', 'Magnitude\nContribution', 'Interaction\nEffect']
    percentages = [dir_pct, mag_pct, int_pct]

    x = np.arange(len(categories))
    width = 0.4

    colors = [COLOR_BLUE if p >= 0 else COLOR_RED for p in percentages]

    bars = ax.bar(x, percentages, width, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1)

    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2,
              label='100% Baseline', alpha=0.6)

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

    max_pct = max([abs(p) for p in percentages])
    ax.set_ylim(-max_pct*0.2, max(max_pct, 100) * 1.2)

    plt.tight_layout()
    filename = f'{model_display}_Attribution_Percentage_Layer{layer}_sigma{sigma_ratio:.2f}'
    fig.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close()


def process_model(model_name, model_display):
    """处理单个模型"""
    print(f"\n{'='*80}")
    print(f"处理模型: {model_display}")
    print("="*80)

    # 加载数据
    summary_file = DATA_DIR / model_name / 'summary.json'
    if not summary_file.exists():
        print(f"  ❌ 未找到 {model_name} 的summary.json")
        return

    with open(summary_file, 'r') as f:
        data = json.load(f)

    # 检查数据有效性
    baseline = data['attribution']['baseline']
    if np.isnan(baseline) or baseline == 0:
        print(f"  ⚠️  {model_name} 数据异常 (baseline={baseline})")
        return

    layer = data['layer']
    sigma_ratio = data['svd_info']['sigma_ratio']
    dir_pct = data['attribution']['direction_attribution_pct']
    mag_pct = data['attribution']['magnitude_attribution_pct']

    print(f"Layer: {layer}")
    print(f"σ₁/σ₂: {sigma_ratio:.2f}")
    print(f"Baseline MA: {baseline:.2f}")
    print(f"Direction: {dir_pct:.1f}%, Magnitude: {mag_pct:.1f}%")

    # 创建输出目录
    output_dir = OUTPUT_BASE / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("\n生成图表:")
    try:
        create_ma_comparison(data, model_display, output_dir)
        print(f"  ✓ 图1: MA对比")
    except Exception as e:
        print(f"  ❌ 图1失败: {e}")

    try:
        create_attribution_grouped(data, model_display, output_dir)
        print(f"  ✓ 图2: 归因分组")
    except Exception as e:
        print(f"  ❌ 图2失败: {e}")

    try:
        create_ablation_comparison(data, model_display, output_dir)
        print(f"  ✓ 图3: 消融对比")
    except Exception as e:
        print(f"  ❌ 图3失败: {e}")

    try:
        create_attribution_percentage(data, model_display, output_dir)
        print(f"  ✓ 图4: 归因百分比")
    except Exception as e:
        print(f"  ❌ 图4失败: {e}")

    print(f"\n✅ {model_display} 完成")


if __name__ == '__main__':
    print(f"\n{'='*80}")
    print("Exp7 可视化 - 批量生成所有模型")
    print("="*80)

    for model_name, config in MODEL_CONFIGS.items():
        process_model(model_name, config['display_name'])

    print(f"\n{'='*80}")
    print("✅ 全部完成！")
    print(f"保存位置: {OUTPUT_BASE}")
    print("="*80)
