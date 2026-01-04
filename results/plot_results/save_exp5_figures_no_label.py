#!/usr/bin/env python3
"""
Exp5: 生成V矩阵消融效果图（不带模型名称、无图例、大字体、稀疏坐标）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# Configuration
DATA_DIR = Path(__file__).resolve().parents[2] / 'results/experiments/exp5'
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'results/plot_results/combined_figures_no_label'

# Model configuration
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
    data_file = DATA_DIR / model_key / 'v_ablation_results.json'
    if not data_file.exists():
        return None
    with open(data_file, 'r') as f:
        return json.load(f)


def save_exp5_single(model_config):
    """保存单个exp5模型的V矩阵消融效果图（不带模型名称、无图例）"""
    model_key = model_config['key']
    model_display = model_config['display']
    
    data = load_model_data(model_key)
    if data is None:
        print(f"  ⚠ {model_display}: Data not found")
        return False

    k_values = data['k_values']
    
    remove_changes = [data['ablation_results']['remove_top_k'][str(k)]['change_percent'] for k in k_values]
    keep_changes = [data['ablation_results']['keep_top_k'][str(k)]['change_percent'] for k in k_values]

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

    x = np.arange(len(k_values))
    width = 0.25

    # Remove Top-k (取绝对值)
    remove_abs = [abs(c) for c in remove_changes]
    bars1 = ax.bar(x - width/2, remove_abs, width,
                   color='#5B9BD5', alpha=0.7, edgecolor='none'

    # Keep Top-k (取绝对值)
    keep_abs = [abs(c) for c in keep_changes]
    bars2 = ax.bar(x + width/2, keep_abs, width,
                   color='#ED7D31', alpha=0.7, edgecolor='none'

    # 添加平滑趋势线
    if len(x) >= 3:
        try:
            x_smooth = np.linspace(x[0] - width/2, x[-1] - width/2, 100)
            spl1 = make_interp_spline(x - width/2, remove_abs, k=2)
            y1_smooth = spl1(x_smooth)
            ax.plot(x_smooth, y1_smooth, '--', color='#5B9BD5', linewidth=2, alpha=0.8)

            x_smooth2 = np.linspace(x[0] + width/2, x[-1] + width/2, 100)
            spl2 = make_interp_spline(x + width/2, keep_abs, k=2)
            y2_smooth = spl2(x_smooth2)
            ax.plot(x_smooth2, y2_smooth, '--', color='#ED7D31', linewidth=2, alpha=0.8)
        except:
            pass

    # 设置坐标轴（稀疏刻度、大字体）
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}' for k in k_values], fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_ylim(0, max(max(remove_abs), max(keep_abs)) * 1.15)
    
    # 稀疏Y轴刻度
    y_max = max(max(remove_abs), max(keep_abs))
    y_ticks = np.linspace(0, y_max, 5)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 不添加图例和模型名称

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'exp5_{model_display}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white'
    plt.close()
    print(f"  ✓ {model_display}: exp5_{model_display}.png")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Exp5: 保存V矩阵消融效果图（不带模型名称）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Exp5 (8张) ---")
    for model in MODEL_CONFIGS:
        save_exp5_single(model)

    print("\n✅ All done!")
    print(f"保存位置: {OUTPUT_DIR}")
