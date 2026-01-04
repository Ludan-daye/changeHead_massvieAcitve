#!/usr/bin/env python3
"""
Exp7: 生成消融效果对比组图（4x2布局，无图例、大字体）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# Configuration
DATA_DIR = Path(__file__).resolve().parents[2] / 'results/experiments/exp7'
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'results/plot_results/combined_figures_no_label'

# 配色
COLOR_BLUE = '#7BA5D5'
COLOR_RED = '#D97E73'
COLOR_RED_DARK = '#B95E53'
COLOR_BLUE_DARK = '#5B85B5'

# Model configuration
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J'},
    {'key': 'bloom_7b1', 'display': 'BLOOM'},
    {'key': 'falcon_7b', 'display': 'Falcon'},
    {'key': 'opt_6.7b', 'display': 'OPT'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5'},
    {'key': 'llama2_13b', 'display': 'LLaMA2'},
]


def load_model_data(model_key):
    """加载模型数据"""
    data_file = DATA_DIR / model_key / 'summary.json'
    if not data_file.exists():
        return None
    with open(data_file, 'r') as f:
        return json.load(f)


def create_exp7_combined():
    """创建exp7的4x2组图"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    axes = axes.flatten()

    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        data = load_model_data(model_key)
        if data is None:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Found',
                   ha='center', va='center', fontsize=14, color='red'
            ax.axis('off'
            continue

        baseline = data['attribution']['baseline']
        if np.isnan(baseline) or baseline == 0:
            ax.text(0.5, 0.5, f'{model_display}\nInvalid Data',
                   ha='center', va='center', fontsize=14, color='red'
            ax.axis('off'
            continue

        ablate_dir = data['attribution']['ablate_direction_mean']
        ablate_mag = data['attribution']['ablate_magnitude_mean']
        ablate_both = data['attribution']['ablate_both_mean']

        categories = ['Direction', 'Magnitude', 'Both']
        baseline_vals = [baseline, baseline, baseline]
        ablated_vals = [ablate_dir, ablate_mag, ablate_both]

        x = np.arange(len(categories))
        width = 0.3

        # 柱状图
        bars1 = ax.bar(x - width/2, baseline_vals, width,
                       color=COLOR_BLUE, alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, ablated_vals, width,
                       color=COLOR_RED, alpha=0.8, edgecolor='black', linewidth=1)

        # 数值标签（放大字体）
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold'

        for bar, val in zip(bars2, ablated_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold'

        # 趋势线
        x_smooth = np.linspace(x[0] - width/2, x[-1] - width/2, 50)
        ax.plot(x_smooth, [baseline]*len(x_smooth),
               color=COLOR_BLUE_DARK, linestyle='--', linewidth=2.5, alpha=0.7)

        try:
            x_smooth2 = np.linspace(x[0] + width/2, x[-1] + width/2, 100)
            spl = make_interp_spline(x + width/2, ablated_vals, k=2)
            y_smooth = spl(x_smooth2)
            ax.plot(x_smooth2, y_smooth,
                   color=COLOR_RED_DARK, linestyle='--', linewidth=2.5, alpha=0.7)
        except:
            pass

        # 设置坐标轴（放大X轴标签）
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=16, fontweight='bold'
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0, max(baseline_vals + ablated_vals) * 1.18)

        # 稀疏Y轴刻度
        y_max = max(baseline_vals + ablated_vals)
        y_ticks = np.linspace(0, y_max, 5)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

        # 边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # 网格
        ax.grid(True, alpha=0.2, axis='y'

        # 底部添加模型名称（放大字体）
        ax.set_xlabel(f'({chr(97+idx)}) {model_display}', fontsize=18, fontweight='bold', labelpad=8)

    # 调整布局
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1,
                       hspace=0.25, wspace=0.2)

    # 保存
    output_png = OUTPUT_DIR / 'exp7_Ablation_Comparison_Combined.png'
    output_pdf = OUTPUT_DIR / 'exp7_Ablation_Comparison_Combined.pdf'
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white'
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white'
    plt.close()
    print(f"✓ 保存: {output_png.name}")
    print(f"✓ 保存: {output_pdf.name}")


if __name__ == '__main__':
    print("="*60)
    print("Exp7: 生成消融效果对比组图（4x2布局，无图例、大字体）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_exp7_combined()

    print("\n✅ All done!")
    print(f"保存位置: {OUTPUT_DIR}")
