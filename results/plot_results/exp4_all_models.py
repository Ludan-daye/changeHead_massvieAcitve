#!/usr/bin/env python3
"""
Exp4 所有模型图表生成
为每个有数据的模型生成5张SVD分析图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
EXP4_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp4')
EXP1_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp1')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp4_figures')

MODEL_DISPLAY_NAMES = {
    'gpt2': 'GPT-2',
    'bloom_7b1': 'BLOOM-7B1',
    'falcon_7b': 'Falcon-7B',
    'gptj_6b': 'GPT-J-6B',
    'mistral_7b_v03': 'Mistral-7B',
    'opt_6.7b': 'OPT-6.7B',
    'qwen2.5_7b': 'Qwen2.5-7B',
    'llama2_13b': 'LLaMA2-13B'
}

# 模型配置：数据文件和格式
MODEL_CONFIGS = {
    'gpt2': {
        'file': 'svd_analysis.json',
        'data_path': lambda data, layer_id: data['svd_analysis'][str(layer_id)],
        'sv_key': 'singular_values',
        'ratio_key': 'ratio_s1_s2',
        'ma_layer': 11  # MA峰值在深层
    },
    'bloom_7b1': {
        'file': 'attention_svd.json',
        'data_path': lambda data, layer_id: data[str(layer_id)]['dense'],
        'sv_key': 'singular_values',
        'ratio_key': 'ratio_s1_s2',
        'ma_layer': 0  # 从exp1 README获取
    },
    'falcon_7b': {
        'file': 'mlp_svd.json',
        'data_path': lambda data, layer_id: data[str(layer_id)],
        'sv_key': 'top_singular_values',
        'ratio_key': 'ratio',
        'ma_layer': 0
    },
    'gptj_6b': {
        'file': 'svd_analysis.json',
        'data_path': lambda data, layer_id: data['svd_analysis'][str(layer_id)],
        'sv_key': 'singular_values',
        'ratio_key': 'ratio_s1_s2',
        'ma_layer': 0
    },
    'mistral_7b_v03': {
        'file': 'attention_svd.json',
        'data_path': lambda data, layer_id: data[str(layer_id)]['o_proj'],  # 使用o_proj
        'sv_key': 'singular_values',
        'ratio_key': 'ratio_s1_s2',
        'ma_layer': 0
    },
    'opt_6.7b': {
        'file': 'svd_analysis.json',
        'data_path': lambda data, layer_id: data['svd_analysis'][str(layer_id)],
        'sv_key': 'singular_values',
        'ratio_key': 'sigma1_sigma2_ratio',
        'ma_layer': 25  # 从exp1 README获取
    },
    'qwen2.5_7b': {
        'file': 'svd_analysis.json',
        'data_path': lambda data, layer_id: data[str(layer_id)],
        'sv_key': 'singular_values',
        'ratio_key': 'ratio_s1_s2',
        'ma_layer': 3
    },
    'llama2_13b': {
        'file': 'svd_analysis.json',
        'data_path': lambda data, layer_id: data['svd_analysis'][str(layer_id)],
        'sv_key': 'singular_values',
        'ratio_key': 'ratio_s1_s2',
        'ma_layer': 22  # MA峰值在Layer 22
    }
}

def load_model_data(model):
    """加载模型数据"""
    # 从配置获取MA层
    config = MODEL_CONFIGS[model]
    ma_layer = config['ma_layer']

    # 读取SVD数据
    svd_file = EXP4_DIR / model / config['file']
    with open(svd_file, 'r') as f:
        svd_data = json.load(f)

    return ma_layer, svd_data, config

def get_layer_data(svd_data, layer_id, config):
    """从SVD数据中提取指定层的数据"""
    try:
        layer_data = config['data_path'](svd_data, layer_id)
        ratio = layer_data[config['ratio_key']]

        # 如果有完整的奇异值列表
        if config.get('sv_key'):
            sv = layer_data[config['sv_key']]
        else:
            # 只有summary数据（sigma1, sigma2, ratio）
            sv = None

        return sv, ratio
    except (KeyError, TypeError):
        return None, None

def create_singular_value_decay(model, ma_layer, svd_data, config, output_dir):
    """图1: 奇异值衰减曲线"""
    # 如果只有summary数据，跳过
    if config.get('summary_only'):
        print(f"  ⚠ 图1: 跳过（仅summary数据）")
        return

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # 选择要展示的层（如果数据中存在）
    all_layers = list(svd_data.keys())
    if 'experiment' in all_layers:
        all_layers.remove('experiment')
    if 'svd_analysis' in all_layers:
        all_layers = list(svd_data['svd_analysis'].keys())

    # 转换为整数并排序
    layer_ids = sorted([int(l) for l in all_layers if l.isdigit()])

    # 选择有代表性的层（最多5个）
    if len(layer_ids) > 5:
        # 前2层、中间1层、MA层、最后1层
        selected = [layer_ids[0], layer_ids[1]]
        mid = len(layer_ids) // 2
        if layer_ids[mid] not in selected:
            selected.append(layer_ids[mid])
        if ma_layer not in selected:
            selected.append(ma_layer)
        if layer_ids[-1] not in selected and len(selected) < 5:
            selected.append(layer_ids[-1])
        layers_to_plot = sorted(selected)[:5]
    else:
        layers_to_plot = layer_ids

    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
    plotted = False

    for i, layer_id in enumerate(layers_to_plot):
        sv, ratio = get_layer_data(svd_data, layer_id, config)
        if sv is None:
            continue

        x = np.arange(1, len(sv) + 1)
        plotted = True

        # MA层用红色突出
        if layer_id == ma_layer:
            ax.plot(x, sv, 'o-', color='#e74c3c', linewidth=3,
                   markersize=6, label=f'Layer {layer_id} (MA Layer)',
                   alpha=0.9, markeredgecolor='white', markeredgewidth=1)
        else:
            color_idx = i if i < len(colors) else i % len(colors)
            ax.plot(x, sv, 'o-', color=colors[color_idx], linewidth=2,
                   markersize=4, label=f'Layer {layer_id}', alpha=0.7)

    if not plotted:
        print(f"  ⚠ 图1: 无有效数据")
        plt.close()
        return

    ax.set_xlabel('Singular Value Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Singular Value Magnitude', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    filename = f'{model}_Singular_Value_Decay_Comparison'
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"  ✓ 图1: {filename}")
    plt.close()

def create_ratio_comparison(model, ma_layer, svd_data, config, output_dir):
    """图2: σ1/σ2比率对比"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 获取所有层
    all_layers = list(svd_data.keys())
    if 'experiment' in all_layers:
        all_layers.remove('experiment')
    if 'svd_analysis' in all_layers:
        all_layers = list(svd_data['svd_analysis'].keys())

    layer_ids = sorted([int(l) for l in all_layers if l.isdigit()])

    # 选择有代表性的层
    if len(layer_ids) > 5:
        selected = [layer_ids[0], layer_ids[1]]
        mid = len(layer_ids) // 2
        if layer_ids[mid] not in selected:
            selected.append(layer_ids[mid])
        if ma_layer not in selected:
            selected.append(ma_layer)
        if layer_ids[-1] not in selected and len(selected) < 5:
            selected.append(layer_ids[-1])
        layers_to_plot = sorted(selected)[:5]
    else:
        layers_to_plot = layer_ids

    layers = []
    ratios = []
    colors_list = []

    for layer_id in layers_to_plot:
        sv, ratio = get_layer_data(svd_data, layer_id, config)
        if ratio is None:
            continue

        layers.append(f'Layer {layer_id}')
        ratios.append(ratio)
        colors_list.append('#e74c3c' if layer_id == ma_layer else '#6495ED')

    if len(ratios) == 0:
        print(f"  ⚠ 图2: 无有效数据")
        plt.close()
        return

    x_pos = np.arange(len(layers))
    bars = ax.bar(x_pos, ratios, color=colors_list, alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.6)

    # 标注数值
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 参考线
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='No Dominance (σ1=σ2)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, fontsize=11)
    ax.set_ylabel('σ1/σ2 Ratio', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0, top=max(ratios) * 1.2)

    plt.tight_layout()

    filename = f'{model}_Sigma1_Sigma2_Ratio_Comparison'
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"  ✓ 图2: {filename}")
    plt.close()

def create_energy_distribution(model, ma_layer, svd_data, config, output_dir):
    """图3: 累积能量分布"""
    # 如果只有summary数据，跳过
    if config.get('summary_only'):
        print(f"  ⚠ 图3: 跳过（仅summary数据）")
        return

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # 获取所有层
    all_layers = list(svd_data.keys())
    if 'experiment' in all_layers:
        all_layers.remove('experiment')
    if 'svd_analysis' in all_layers:
        all_layers = list(svd_data['svd_analysis'].keys())

    layer_ids = sorted([int(l) for l in all_layers if l.isdigit()])

    # 选择层
    if len(layer_ids) > 5:
        selected = [layer_ids[0], layer_ids[1]]
        mid = len(layer_ids) // 2
        if layer_ids[mid] not in selected:
            selected.append(layer_ids[mid])
        if ma_layer not in selected:
            selected.append(ma_layer)
        if layer_ids[-1] not in selected and len(selected) < 5:
            selected.append(layer_ids[-1])
        layers_to_plot = sorted(selected)[:5]
    else:
        layers_to_plot = layer_ids

    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
    plotted = False

    for i, layer_id in enumerate(layers_to_plot):
        sv, ratio = get_layer_data(svd_data, layer_id, config)
        if sv is None:
            continue

        sv_array = np.array(sv)
        energy = sv_array ** 2
        cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
        x = np.arange(1, len(cumulative_energy) + 1)
        plotted = True

        if layer_id == ma_layer:
            ax.plot(x, cumulative_energy, 'o-', color='#e74c3c', linewidth=3,
                   markersize=6, label=f'Layer {layer_id} (MA Layer)',
                   alpha=0.9, markeredgecolor='white', markeredgewidth=1)
        else:
            color_idx = i if i < len(colors) else i % len(colors)
            ax.plot(x, cumulative_energy, 'o-', color=colors[color_idx], linewidth=2,
                   markersize=4, label=f'Layer {layer_id}', alpha=0.7)

    if not plotted:
        print(f"  ⚠ 图3: 无有效数据")
        plt.close()
        return

    # 参考线
    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='90% Energy')

    ax.set_xlabel('Number of Singular Values', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Energy (%)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()

    filename = f'{model}_Cumulative_Energy_Distribution'
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"  ✓ 图3: {filename}")
    plt.close()

def create_ma_layer_singular_values(model, ma_layer, svd_data, config, output_dir):
    """图4: MA层奇异值大小"""
    # 如果只有summary数据，跳过
    if config.get('summary_only'):
        print(f"  ⚠ 图4: 跳过（仅summary数据）")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sv, ratio = get_layer_data(svd_data, ma_layer, config)
    if sv is None:
        print(f"  ⚠ 图4: 无法获取Layer {ma_layer}的数据")
        plt.close()
        return

    sv_array = np.array(sv)
    x = np.arange(1, len(sv_array) + 1)

    ax.bar(x, sv_array, color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注σ1和σ2
    ax.text(1, sv_array[0], f'σ1={sv_array[0]:.2f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='darkred')
    if len(sv_array) > 1:
        ax.text(2, sv_array[1], f'σ2={sv_array[1]:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='darkred')

    plt.tight_layout()

    filename = f'{model}_Layer{ma_layer}_Singular_Values'
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"  ✓ 图4: {filename}")
    plt.close()

def create_ma_layer_normalized(model, ma_layer, svd_data, config, output_dir):
    """图5: MA层归一化奇异值"""
    # 如果只有summary数据，跳过
    if config.get('summary_only'):
        print(f"  ⚠ 图5: 跳过（仅summary数据）")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sv, ratio = get_layer_data(svd_data, ma_layer, config)
    if sv is None:
        print(f"  ⚠ 图5: 无法获取Layer {ma_layer}的数据")
        plt.close()
        return

    sv_array = np.array(sv)
    x = np.arange(1, len(sv_array) + 1)

    # 归一化
    sv_normalized = sv_array / sv_array[0]
    ax.plot(x, sv_normalized, 'o-', color='#3498db', linewidth=2.5,
            markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Magnitude (relative to σ1)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    if ratio and ratio > 0:
        ax.axhline(y=1/ratio, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'σ2/σ1 = {1/ratio:.3f}')
        ax.legend(loc='upper right', fontsize=10)

    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    filename = f'{model}_Layer{ma_layer}_Normalized_Spectrum'
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"  ✓ 图5: {filename}")
    plt.close()

def process_model(model):
    """处理单个模型"""
    print(f"\n{'='*80}")
    print(f"处理模型: {MODEL_DISPLAY_NAMES[model]}")
    print(f"{'='*80}")

    try:
        # 加载数据
        ma_layer, svd_data, config = load_model_data(model)
        print(f"MA层: Layer {ma_layer}")

        # 创建输出目录
        output_dir = OUTPUT_DIR / model
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成5张图
        print(f"\n生成图表:")
        create_singular_value_decay(model, ma_layer, svd_data, config, output_dir)
        create_ratio_comparison(model, ma_layer, svd_data, config, output_dir)
        create_energy_distribution(model, ma_layer, svd_data, config, output_dir)
        create_ma_layer_singular_values(model, ma_layer, svd_data, config, output_dir)
        create_ma_layer_normalized(model, ma_layer, svd_data, config, output_dir)

        print(f"\n✅ {MODEL_DISPLAY_NAMES[model]} 完成")
        return True

    except Exception as e:
        print(f"\n❌ {MODEL_DISPLAY_NAMES[model]} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 主程序
if __name__ == "__main__":
    print("="*80)
    print("Exp4 所有模型图表生成")
    print("="*80)

    models_to_process = ['bloom_7b1', 'falcon_7b', 'gptj_6b', 'mistral_7b_v03', 'opt_6.7b', 'qwen2.5_7b']

    success_count = 0
    for model in models_to_process:
        if process_model(model):
            success_count += 1

    print("\n" + "="*80)
    print(f"✅ 完成! 成功处理 {success_count}/{len(models_to_process)} 个模型")
    print(f"每个模型生成 5 个图表 × 2 格式 = 10 个文件")
    print(f"总计: {success_count * 10} 个文件")
    print(f"保存位置: {OUTPUT_DIR}")
    print("="*80)
