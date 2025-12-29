#!/usr/bin/env python3
"""
从数据重新绘制Exp2合并图
- 无总标题
- 大字号横坐标轴
- 稀疏刻度
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
RESULTS_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp2')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp2_figures')

# 模型配置
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2', 'critical_layer': 3},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B', 'critical_layer': 22},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1', 'critical_layer': 28},
    {'key': 'falcon_7b', 'display': 'Falcon-7B', 'critical_layer': 3},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B', 'critical_layer': 3},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B', 'critical_layer': 31},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B', 'critical_layer': 3},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B', 'critical_layer': 30},
]

# 颜色配置
COLOR_BASELINE = '#7BA5D5'    # 蓝色 - Baseline
COLOR_ABLATED = '#D97E73'     # 红色 - Layer Disabled
COLOR_FILL = '#FFB6B0'        # 浅红色 - 填充区域

def load_exp2_data(model_key):
    """加载Exp2数据"""
    summary_file = RESULTS_DIR / model_key / 'summary.json'

    # 特殊情况：LLaMA2数据在另一个位置
    if not summary_file.exists() and 'llama' in model_key:
        alt_path = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/models/llama2_13b/exp2b_mlp_layer_ablation/summary.json')
        if alt_path.exists():
            summary_file = alt_path

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data

def create_combined_figure():
    """创建合并图"""

    # 创建2行4列的子图 - 增大尺寸
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))
    axes = axes.flatten()

    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']
        critical_layer = model_config['critical_layer']

        # 加载数据
        data = load_exp2_data(model_key)
        if data is None:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Available',
                   ha='center', va='center', fontsize=24, color='red',
                   fontweight='bold')
            ax.axis('off')
            continue

        # 提取消融数据
        ablation = data.get('ablation', {})
        if not ablation:
            ax.text(0.5, 0.5, f'{model_display}\nNo Data',
                   ha='center', va='center', fontsize=24, color='red',
                   fontweight='bold')
            ax.axis('off')
            continue

        # 排序层并获取MA值
        layers = sorted([int(k) for k in ablation.keys()])
        ma_ablated = [ablation[str(layer)] for layer in layers]

        # 计算baseline（假设为最大MA值）
        baseline = max(ma_ablated)
        ma_baseline = [baseline] * len(layers)

        # 绘制图形 - 加粗线条
        ax.plot(layers, ma_baseline, color=COLOR_BASELINE, linewidth=4,
               linestyle='--', label='Baseline', zorder=3)
        ax.plot(layers, ma_ablated, color=COLOR_ABLATED, linewidth=4,
               label='Ablated', zorder=4)

        # 填充区域
        ax.fill_between(layers, ma_baseline, ma_ablated,
                        color=COLOR_FILL, alpha=0.4, zorder=1)

        # 标记关键层
        if critical_layer < len(layers):
            ax.axvline(x=critical_layer, color='darkgray', linestyle=':',
                      linewidth=3, alpha=0.8, zorder=2)

        # 设置坐标轴标签 - 超大字号
        ax.set_xlabel('Layer Index', fontsize=26, fontweight='bold', labelpad=10)
        ax.set_ylabel('MA Value', fontsize=26, fontweight='bold', labelpad=10)

        # 横坐标刻度 - 更稀疏、更大字号
        n_layers = len(layers)
        if n_layers <= 12:
            tick_step = 3
        elif n_layers <= 30:
            tick_step = 6
        else:
            tick_step = 10

        tick_positions = list(range(0, n_layers, tick_step))
        if (n_layers - 1) not in tick_positions:
            tick_positions.append(n_layers - 1)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=24, fontweight='bold')

        # 纵坐标字体 - 加大加粗
        ax.tick_params(axis='y', labelsize=22, width=2, length=8)
        ax.tick_params(axis='x', width=2, length=8)

        # 不显示标题（按要求去掉模型名称）
        # ax.set_title(model_display, fontsize=28, fontweight='bold', pad=20)

        # 加强网格
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.2, color='gray')

        # 加粗边框
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

        print(f"✓ Plotted {model_display}")

    # 调整布局 - 去掉标题后减少顶部边距
    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08,
                       hspace=0.25, wspace=0.3)

    # 保存为 exp2_combined_all_models_final
    output_file_png = OUTPUT_DIR / 'exp2_combined_all_models_final.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_all_models_final.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Redrawing Exp2 combined figure from data...\n")
    create_combined_figure()
    print("\n✅ Done!")
