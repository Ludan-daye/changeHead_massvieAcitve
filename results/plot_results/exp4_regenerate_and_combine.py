#!/usr/bin/env python3
"""
Exp4: 重新生成累积能量分布图（大字体、稀疏坐标、无图例），然后组合成4×2图组
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置
EXP4_DIR = Path('PROJECT_ROOT/results/experiments/exp4')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp4_figures')

# 8个模型配置
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2', 'ma_layer': 11,
     'file': 'svd_analysis.json', 'data_path': 'svd_analysis', 'sv_key': 'singular_values'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B', 'ma_layer': 0,
     'file': 'svd_analysis.json', 'data_path': 'svd_analysis', 'sv_key': 'singular_values'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1', 'ma_layer': 0,
     'file': 'attention_svd.json', 'data_path': None, 'sv_key': 'singular_values', 'sub_key': 'dense'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B', 'ma_layer': 0,
     'file': 'mlp_svd.json', 'data_path': None, 'sv_key': 'top_singular_values'},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B', 'ma_layer': 25,
     'file': 'svd_analysis.json', 'data_path': 'svd_analysis', 'sv_key': 'singular_values'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B', 'ma_layer': 0,
     'file': 'attention_svd.json', 'data_path': None, 'sv_key': 'singular_values', 'sub_key': 'o_proj'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B', 'ma_layer': 3,
     'file': 'svd_analysis.json', 'data_path': None, 'sv_key': 'singular_values'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B', 'ma_layer': 22,
     'file': 'svd_analysis.json', 'data_path': 'svd_analysis', 'sv_key': 'singular_values'},
]


def load_svd_data(model_config):
    """加载SVD数据"""
    model_key = model_config['key']
    svd_file = EXP4_DIR / model_key / model_config['file']
    
    if not svd_file.exists():
        return None
    
    with open(svd_file, 'r') as f:
        data = json.load(f)
    
    return data


def get_layer_sv(data, layer_id, config):
    """获取指定层的奇异值"""
    try:
        layer_str = str(layer_id)
        
        if config.get('data_path'):
            layer_data = data[config['data_path']][layer_str]
        else:
            layer_data = data[layer_str]
        
        if config.get('sub_key'):
            layer_data = layer_data[config['sub_key']]
        
        sv = layer_data[config['sv_key']]
        return np.array(sv)
    except (KeyError, TypeError):
        return None


def get_all_layers(data, config):
    """获取所有层ID"""
    if config.get('data_path'):
        all_keys = list(data[config['data_path']].keys())
    else:
        all_keys = list(data.keys())
    
    layer_ids = sorted([int(k) for k in all_keys if k.isdigit()])
    return layer_ids


def select_layers(layer_ids, ma_layer):
    """选择要展示的层（最多4个）"""
    if len(layer_ids) <= 4:
        return layer_ids
    
    selected = [layer_ids[0]]  # 第一层
    
    # 中间层
    mid = len(layer_ids) // 2
    if layer_ids[mid] not in selected:
        selected.append(layer_ids[mid])
    
    # MA层
    if ma_layer in layer_ids and ma_layer not in selected:
        selected.append(ma_layer)
    
    # 最后一层
    if layer_ids[-1] not in selected:
        selected.append(layer_ids[-1])
    
    return sorted(selected)[:4]


def plot_single_model(ax, model_config):
    """为单个模型绘制累积能量分布图"""
    data = load_svd_data(model_config)
    if data is None:
        return False
    
    layer_ids = get_all_layers(data, model_config)
    if not layer_ids:
        return False
    
    ma_layer = model_config['ma_layer']
    layers_to_plot = select_layers(layer_ids, ma_layer)
    
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    for i, layer_id in enumerate(layers_to_plot):
        sv = get_layer_sv(data, layer_id, model_config)
        if sv is None:
            continue
        
        # 计算累积能量
        energy = sv ** 2
        cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
        x = np.arange(1, len(cumulative_energy) + 1)
        
        # MA层用红色粗线
        if layer_id == ma_layer:
            ax.plot(x, cumulative_energy, 'o-', color='#e74c3c', linewidth=2.5,
                   markersize=4, alpha=0.9)
        else:
            color_idx = i % len(colors)
            ax.plot(x, cumulative_energy, 'o-', color=colors[color_idx], linewidth=2,
                   markersize=3, alpha=0.7)
    
    # 90%参考线
    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 设置坐标轴
    ax.set_ylim(0, 105)
    
    # 稀疏的x轴刻度
    max_x = len(cumulative_energy) if 'cumulative_energy' in dir() else 20
    x_ticks = np.linspace(0, max_x, 5, dtype=int)
    ax.set_xticks(x_ticks)
    
    # 稀疏的y轴刻度
    ax.set_yticks([0, 25, 50, 75, 100])
    
    # 大字体刻度标签
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 网格
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    
    return True


def main():
    print("="*60)
    print("Exp4: Regenerate and Combine Energy Distribution")
    print("="*60 + "\n")

    # 创建4列2行的子图
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    axes = axes.flatten()

    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_display = model_config['display']

        success = plot_single_model(ax, model_config)
        
        if not success:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Found',
                   ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            print(f"⚠ {model_display}: Data not found")
            continue

        # 底部添加模型名称
        ax.set_xlabel(model_display, fontsize=13, fontweight='bold', labelpad=3)
        print(f"✓ {model_display}")

    # 紧凑布局
    plt.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.12,
                       hspace=0.25, wspace=0.18)

    # 保存（高DPI）
    output_png = OUTPUT_DIR / 'exp4_combined_energy_large_font.png'
    output_pdf = OUTPUT_DIR / 'exp4_combined_energy_large_font.pdf'

    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_png}")
    print(f"✅ Combined figure saved: {output_pdf}")

    plt.close()
    print("\n✅ All done!")


if __name__ == '__main__':
    main()
