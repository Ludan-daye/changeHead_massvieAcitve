#!/usr/bin/env python3
"""
合并Exp2的8个模型2D对比图为一个大图
布局：2行 × 4列
图例：底部中央统一放置
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
RESULTS_DIR = Path('PROJECT_ROOT/results/experiments/exp2')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')

# 模型配置（按特定顺序排列，只包含有数据的7个模型）
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2', 'critical_layer': 3},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B', 'critical_layer': 22},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1', 'critical_layer': 28},
    {'key': 'falcon_7b', 'display': 'Falcon-7B', 'critical_layer': 3},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B', 'critical_layer': 3},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B', 'critical_layer': 31},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B', 'critical_layer': 3},
]

# 颜色配置（参考示例图）
COLOR_BASELINE = '#666666'  # 深灰色
COLOR_ABLATED = '#D97E73'   # 橙红色
COLOR_CRITICAL = '#7BA5D5'  # 蓝色

def load_exp2_data(model_key):
    """加载Exp2数据"""
    summary_file = RESULTS_DIR / model_key / 'summary.json'

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data

def create_combined_figure():
    """创建合并的大图"""

    # 创建2行4列的子图
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.98, top=0.94, bottom=0.12)

    axes = []
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    # 为每个模型绘制子图
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']
        critical_layer = model_config['critical_layer']

        # 加载数据
        data = load_exp2_data(model_key)
        if data is None:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Available',
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 提取消融数据
        ablation = data.get('ablation', {})
        if not ablation:
            ax.text(0.5, 0.5, f'{model_display}\nNo Data',
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 排序层
        layers = sorted([int(k) for k in ablation.keys()])
        ma_values = [ablation[str(layer)] for layer in layers]

        # 计算基准线（假设关键层的最小值附近为基准）
        baseline = max(ma_values)  # 使用最大值作为基准参考

        # 绘制热力图风格的2D对比
        # 将层分组显示
        n_layers = len(layers)

        # 创建2D网格
        grid_rows = int(np.ceil(np.sqrt(n_layers)))
        grid_cols = int(np.ceil(n_layers / grid_rows))

        # 绘制每个层的方块
        for i, (layer, ma) in enumerate(zip(layers, ma_values)):
            row = i // grid_cols
            col = i % grid_cols

            # 归一化颜色（相对于基准）
            if layer == critical_layer:
                color = COLOR_CRITICAL
                alpha = 1.0
                edgecolor = 'black'
                linewidth = 3
            else:
                ratio = ma / baseline if baseline > 0 else 0
                if ratio > 0.8:
                    color = COLOR_BASELINE
                    alpha = 0.9
                else:
                    color = COLOR_ABLATED
                    alpha = 0.7
                edgecolor = 'gray'
                linewidth = 1

            rect = Rectangle((col, grid_rows - 1 - row), 1, 1,
                           facecolor=color, alpha=alpha,
                           edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)

            # 添加层号文本
            if n_layers <= 32:  # 只在层数不太多时显示
                ax.text(col + 0.5, grid_rows - 1 - row + 0.5, str(layer),
                       ha='center', va='center', fontsize=8, color='white',
                       fontweight='bold' if layer == critical_layer else 'normal')

        # 设置坐标轴
        ax.set_xlim(0, grid_cols)
        ax.set_ylim(0, grid_rows)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # 添加标题
        ax.set_title(f'{model_display}\n(Critical Layer: {critical_layer})',
                    fontsize=14, fontweight='bold', pad=10)

        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # 添加统一的图例（底部中央）
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_BASELINE, alpha=0.9, edgecolor='gray',
                      label='Non-critical Layers (MA > 80% baseline)'),
        mpatches.Patch(facecolor=COLOR_ABLATED, alpha=0.7, edgecolor='gray',
                      label='Suppressed Layers (MA < 80% baseline)'),
        mpatches.Patch(facecolor=COLOR_CRITICAL, alpha=1.0, edgecolor='black', linewidth=3,
                      label='Critical Layer (MA source)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=13, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, 0.02))

    # 添加总标题
    fig.suptitle('Exp2: MLP Layer-wise Ablation Analysis - 2D Comparison Across 8 Models',
                fontsize=18, fontweight='bold', y=0.98)

    # 保存
    output_file_png = OUTPUT_DIR / 'exp2_combined_2d_comparison.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_2d_comparison.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Creating combined Exp2 2D comparison figure...")
    create_combined_figure()
    print("Done!")
