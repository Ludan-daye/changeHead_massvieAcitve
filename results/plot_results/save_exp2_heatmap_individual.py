#!/usr/bin/env python3
"""
Generate 8 individual Exp2 2D heatmap matrix figures
- New color style
- Large font labels
- Fix x-axis label overlapping issue
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration - use relative paths from repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results/experiments/exp2'
OUTPUT_DIR = PROJECT_ROOT / 'results/plot_results/exp2_heatmap_individual'

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


def load_exp2_data(model_key):
    """加载Exp2数据"""
    summary_file = RESULTS_DIR / model_key / 'summary.json'

    # 特殊情况：LLaMA2数据在另一个位置
    if not summary_file.exists() and 'llama' in model_key:
        alt_path = PROJECT_ROOT / 'results/archive/by_model/llama2_13b/exp2_llama2_13b/layer_3_results.json'
        if alt_path.exists():
            with open(alt_path, 'r') as f:
                raw_data = json.load(f)
            ablation = {}
            for layer, values in raw_data.items():
                if isinstance(values, dict) and 'top1_mean' in values:
                    ablation[layer] = values['top1_mean']
            return {'ablation': ablation}

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data


def create_heatmap_figure(model_config):
    """为单个模型创建2D热力图矩阵"""
    model_key = model_config['key']
    model_display = model_config['display']
    critical_layer = model_config['critical_layer']

    # 加载数据
    data = load_exp2_data(model_key)
    if data is None:
        print(f"✗ {model_display}: Data not found")
        return False

    # 提取消融数据
    ablation = data.get('ablation', {})
    if not ablation:
        print(f"✗ {model_display}: No ablation data")
        return False

    # 排序层并获取MA值
    layers = sorted([int(k) for k in ablation.keys()])
    values = np.array([ablation[str(layer)] for layer in layers])
    n_layers = len(layers)

    # 计算网格尺寸（尽量接近正方形）
    grid_size = int(np.ceil(np.sqrt(n_layers)))
    
    # 填充到完整网格
    padded_values = np.full(grid_size * grid_size, np.nan)
    padded_values[:n_layers] = values
    matrix = padded_values.reshape(grid_size, grid_size)

    # 计算baseline和阈值
    baseline = np.nanmax(values)
    threshold_80 = baseline * 0.8

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 创建颜色矩阵
    # 灰色: MA >= 80% baseline (Non-critical)
    # 红色: MA < 80% baseline (Suppressed)
    # 蓝色: Critical layer
    
    colors = np.zeros((grid_size, grid_size, 3))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx >= n_layers:
                # 超出范围，设为白色
                colors[i, j] = [1, 1, 1]
            elif idx == critical_layer:
                # 关键层，蓝色
                colors[i, j] = [0.3, 0.5, 0.8]  # 蓝色
            elif values[idx] >= threshold_80:
                # 非关键层，灰色
                colors[i, j] = [0.7, 0.7, 0.7]  # 灰色
            else:
                # 被抑制层，红色（根据抑制程度调整深浅）
                suppression = 1 - (values[idx] / baseline)
                colors[i, j] = [0.9, 0.5 - suppression * 0.3, 0.5 - suppression * 0.3]  # 红色

    ax.imshow(colors, aspect='equal')

    # 添加层编号标注
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n_layers:
                # 根据背景颜色选择文字颜色
                if idx == critical_layer:
                    text_color = 'white'
                    fontweight = 'bold'
                else:
                    text_color = 'black'
                    fontweight = 'normal'
                
                ax.text(j, i, str(idx), ha='center', va='center',
                       fontsize=14, color=text_color, fontweight=fontweight)

    # 设置标题
    ax.set_title(f'{model_display}\n(Critical Layer: {critical_layer})', 
                fontsize=16, fontweight='bold', pad=15)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 添加边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # 调整布局
    plt.tight_layout()

    # 保存
    output_file_png = OUTPUT_DIR / f'exp2_heatmap_{model_key}.png'
    output_file_pdf = OUTPUT_DIR / f'exp2_heatmap_{model_key}.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"✓ {model_display} saved: {output_file_png.name}")

    plt.close()
    return True


def create_legend():
    """创建图例"""
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # 创建图例色块
    legend_colors = [
        ([0.7, 0.7, 0.7], 'Non-critical Layers (MA ≥ 80% baseline)'),
        ([0.9, 0.4, 0.4], 'Suppressed Layers (MA < 80% baseline)'),
        ([0.3, 0.5, 0.8], 'Critical Layer (MA source)'),
    ]
    
    for i, (color, label) in enumerate(legend_colors):
        rect = plt.Rectangle((0.1, 0.7 - i * 0.25), 0.15, 0.15, 
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.3, 0.775 - i * 0.25, label, fontsize=12, va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'legend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Legend saved: {output_file.name}")


def main():
    """主函数"""
    print("="*60)
    print("生成8个单独的Exp2 2D热力图矩阵小图")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for model_config in MODEL_CONFIGS:
        if create_heatmap_figure(model_config):
            success_count += 1

    # 创建图例
    create_legend()

    print(f"\n✅ 完成! 成功生成 {success_count}/8 个热力图")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
