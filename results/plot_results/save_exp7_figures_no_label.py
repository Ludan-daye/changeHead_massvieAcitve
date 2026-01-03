#!/usr/bin/env python3
"""
Exp7: 生成消融效果对比图（不带模型名称、无图例、大字体、稀疏坐标）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# 配置
DATA_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp7')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/combined_figures_no_label')

# 配色
COLOR_BLUE = '#7BA5D5'
COLOR_RED = '#D97E73'
COLOR_RED_DARK = '#B95E53'
COLOR_BLUE_DARK = '#5B85B5'

# 模型配置
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]


def load_model_data(model_key):
    """加载模型数据"""
    data_file = DATA_DIR / model_key / 'summary.json'
    if not data_file.exists():
        return None
    with open(data_file, 'r') as f:
        return json.load(f)


def save_exp7_single(model_config):
    """保存单个exp7模型的消融效果对比图（不带模型名称、无图例）"""
    model_key = model_config['key']
    model_display = model_config['display']
    
    data = load_model_data(model_key)
    if data is None:
        print(f"  ⚠ {model_display}: Data not found")
        return False

    # 检查数据有效性
    baseline = data['attribution']['baseline']
    if np.isnan(baseline) or baseline == 0:
        print(f"  ⚠ {model_display}: Invalid data (baseline={baseline})")
        return False

    ablate_dir = data['attribution']['ablate_direction_mean']
    ablate_mag = data['attribution']['ablate_magnitude_mean']
    ablate_both = data['attribution']['ablate_both_mean']

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

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
               f'{height:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    for bar, val in zip(bars2, ablated_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + baseline*0.01,
               f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 趋势线
    x_smooth = np.linspace(x[0] - width/2, x[-1] - width/2, 50)
    ax.plot(x_smooth, [baseline]*len(x_smooth),
           color=COLOR_BLUE_DARK, linestyle='--', linewidth=2, alpha=0.7)

    try:
        x_smooth2 = np.linspace(x[0] + width/2, x[-1] + width/2, 100)
        spl = make_interp_spline(x + width/2, ablated_vals, k=2)
        y_smooth = spl(x_smooth2)
        ax.plot(x_smooth2, y_smooth,
               color=COLOR_RED_DARK, linestyle='--', linewidth=2, alpha=0.7)
    except:
        pass

    # 设置坐标轴（放大X轴标签）
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0, max(baseline_vals + ablated_vals) * 1.15)
    
    # 稀疏Y轴刻度
    y_max = max(baseline_vals + ablated_vals)
    y_ticks = np.linspace(0, y_max, 5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 网格
    ax.grid(True, alpha=0.2, axis='y')

    # 不添加图例和模型名称

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'exp7_{model_display}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {model_display}: exp7_{model_display}.png")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Exp7: 保存消融效果对比图（不带模型名称）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Exp7 (8张) ---")
    for model in MODEL_CONFIGS:
        save_exp7_single(model)

    print("\n✅ All done!")
    print(f"保存位置: {OUTPUT_DIR}")
