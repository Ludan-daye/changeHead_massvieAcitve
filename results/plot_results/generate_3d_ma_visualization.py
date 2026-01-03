#!/usr/bin/env python3
"""
Generate 3D Massive Activation visualization
Mimics reference figure style: 3D bar chart showing MA distribution across layers
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import matplotlib.colors as mcolors

# Configuration - use relative paths from repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results/experiments/exp1'
OUTPUT_DIR = PROJECT_ROOT / 'results/plot_results/exp1_3d_figures'

# 模型配置（8个模型）
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'mistral_7b', 'display': 'Mistral-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]


def load_baseline_data(model_key):
    """加载baseline数据"""
    baseline_file = RESULTS_DIR / model_key / 'baseline' / 'results.json'
    
    if not baseline_file.exists():
        print(f"Warning: {baseline_file} not found")
        return None
    
    with open(baseline_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_3d_figure_style2(model_config):
    """为单个模型创建3D MA可视化 - 模仿参考图风格"""
    model_key = model_config['key']
    model_display = model_config['display']
    
    # 加载数据
    data = load_baseline_data(model_key)
    if data is None:
        print(f"✗ {model_display}: Data not found")
        return False
    
    # 提取各层的数据
    layers = sorted([int(k) for k in data.keys()])
    n_layers = len(layers)
    
    # 提取top1, top2, top3值作为3个维度
    top1_values = [data[str(layer)]['top1_mean'] for layer in layers]
    top2_values = [data[str(layer)]['top2_mean'] for layer in layers]
    top3_values = [data[str(layer)]['top3_mean'] for layer in layers]
    
    # 创建3D图 - 类似参考图的风格
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置白色背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 为每个层绘制3个柱子（top1, top2, top3）
    bar_width = 0.25
    
    for i, layer in enumerate(layers):
        # Top1 - 红色/橙色
        ax.bar3d(i, 0, 0, bar_width, bar_width, top1_values[i], 
                color='#E74C3C', alpha=0.9, edgecolor='black', linewidth=0.3)
        # Top2 - 黄色
        ax.bar3d(i, 0.3, 0, bar_width, bar_width, top2_values[i], 
                color='#F39C12', alpha=0.9, edgecolor='black', linewidth=0.3)
        # Top3 - 绿色
        ax.bar3d(i, 0.6, 0, bar_width, bar_width, top3_values[i], 
                color='#27AE60', alpha=0.9, edgecolor='black', linewidth=0.3)
    
    # 设置坐标轴
    ax.set_xlabel('Layer', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Rank', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_zlabel('Activation Value', fontsize=11, fontweight='bold', labelpad=8)
    
    # 设置标题
    ax.set_title(f'{model_display}', fontsize=14, fontweight='bold', pad=15)
    
    # 设置刻度
    if n_layers <= 15:
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layers, fontsize=8)
    else:
        step = max(1, n_layers // 8)
        ax.set_xticks(range(0, n_layers, step))
        ax.set_xticklabels([layers[i] for i in range(0, n_layers, step)], fontsize=8)
    
    ax.set_yticks([0.15, 0.45, 0.75])
    ax.set_yticklabels(['Top1', 'Top2', 'Top3'], fontsize=9)
    
    # 调整视角 - 类似参考图
    ax.view_init(elev=20, azim=-60)
    
    # 网格线
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file_png = OUTPUT_DIR / f'exp1_3d_{model_key}.png'
    output_file_pdf = OUTPUT_DIR / f'exp1_3d_{model_key}.pdf'
    
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"✓ {model_display} saved: {output_file_png.name}")
    
    plt.close()
    return True


def create_combined_3d_figure():
    """创建组合的3D图（2x3布局）- 新风格"""
    fig = plt.figure(figsize=(18, 12))
    
    valid_models = []
    for model_config in MODEL_CONFIGS:
        data = load_baseline_data(model_config['key'])
        if data is not None:
            valid_models.append((model_config, data))
    
    for idx, (model_config, data) in enumerate(valid_models):
        model_display = model_config['display']
        
        # 提取数据
        layers = sorted([int(k) for k in data.keys()])
        n_layers = len(layers)
        top1_values = [data[str(layer)]['top1_mean'] for layer in layers]
        top2_values = [data[str(layer)]['top2_mean'] for layer in layers]
        top3_values = [data[str(layer)]['top3_mean'] for layer in layers]
        
        # 创建子图
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.set_facecolor('white')
        
        # 绘制3个柱子
        bar_width = 0.25
        for i, layer in enumerate(layers):
            ax.bar3d(i, 0, 0, bar_width, bar_width, top1_values[i], 
                    color='#E74C3C', alpha=0.9, edgecolor='black', linewidth=0.2)
            ax.bar3d(i, 0.3, 0, bar_width, bar_width, top2_values[i], 
                    color='#F39C12', alpha=0.9, edgecolor='black', linewidth=0.2)
            ax.bar3d(i, 0.6, 0, bar_width, bar_width, top3_values[i], 
                    color='#27AE60', alpha=0.9, edgecolor='black', linewidth=0.2)
        
        ax.set_xlabel('Layer', fontsize=9, labelpad=5)
        ax.set_ylabel('', fontsize=9)
        ax.set_zlabel('Value', fontsize=9, labelpad=5)
        ax.set_title(model_display, fontsize=11, fontweight='bold')
        
        # 简化刻度
        if n_layers <= 15:
            ax.set_xticks(range(0, n_layers, 2))
            ax.set_xticklabels([layers[i] for i in range(0, n_layers, 2)], fontsize=7)
        else:
            step = max(1, n_layers // 6)
            ax.set_xticks(range(0, n_layers, step))
            ax.set_xticklabels([layers[i] for i in range(0, n_layers, step)], fontsize=7)
        
        ax.set_yticks([0.15, 0.45, 0.75])
        ax.set_yticklabels(['T1', 'T2', 'T3'], fontsize=7)
        ax.view_init(elev=20, azim=-60)
        ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.suptitle('Massive Activation Visualization Across Models', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    output_file = OUTPUT_DIR / 'exp1_3d_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Combined figure saved: {output_file.name}")
    
    plt.close()


def main():
    """主函数"""
    print("="*60)
    print("生成3D Massive Activation可视化图")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for model_config in MODEL_CONFIGS:
        if create_3d_figure_style2(model_config):
            success_count += 1
    
    # 生成组合图
    create_combined_3d_figure()
    
    print(f"\n✅ 完成! 成功生成 {success_count}/{len(MODEL_CONFIGS)} 个3D图")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
