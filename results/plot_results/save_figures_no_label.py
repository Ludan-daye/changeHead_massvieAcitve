#!/usr/bin/env python3
"""
保存exp2和exp4的16张单独小图（不带模型名称），模型名称作为文件名
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pathlib import Path

# 配置
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/combined_figures_no_label')
EXP2_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp2')
EXP2_ARCHIVE_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/archive/by_model')
EXP4_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp4')

# Exp2模型配置
EXP2_MODELS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]

# Exp4模型配置
EXP4_MODELS = [
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


# ==================== Exp2 函数 ====================

def load_exp2_data(model_name):
    """加载Exp2数据"""
    # 先尝试主目录
    summary_path = EXP2_DIR / model_name / 'summary.json'
    baseline_path = EXP2_DIR / model_name / 'baseline.json'

    # 如果不存在，尝试archive目录（特别是llama2_13b）
    if not summary_path.exists() or not baseline_path.exists():
        summary_path = EXP2_ARCHIVE_DIR / model_name / 'exp2b_mlp_layer_ablation' / 'summary.json'
        baseline_path = EXP2_ARCHIVE_DIR / model_name / 'exp2b_mlp_layer_ablation' / 'baseline.json'

    if not summary_path.exists() or not baseline_path.exists():
        return None

    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)

    ablation = summary_data.get('ablation', {})
    layers = sorted([int(k) for k in ablation.keys()])
    disabled_values = np.array([ablation[str(l)] for l in layers])

    baseline_values = []
    if 'results' in baseline_data:
        results = baseline_data['results']
        for layer in layers:
            layer_data = results.get(str(layer), {})
            if isinstance(layer_data, dict) and 'mean' in layer_data:
                value = layer_data['mean']
                baseline_values.append(value if np.isfinite(value) else 0)
            else:
                baseline_values.append(0)

    return {
        'model': model_name,
        'layers': np.array(layers),
        'disabled_values': disabled_values,
        'baseline_values': np.array(baseline_values)
    }


def save_exp2_single(model_config):
    """保存单个exp2模型的图（不带模型名称）"""
    model_key = model_config['key']
    model_display = model_config['display']
    
    data = load_exp2_data(model_key)
    if data is None:
        print(f"  ⚠ {model_display}: Data not found")
        return False

    layers = data['layers']
    disabled_values = data['disabled_values']
    baseline_values = data['baseline_values']

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

    # 创建平滑曲线
    if len(layers) > 3:
        layers_smooth = np.linspace(layers.min(), layers.max(), 300)
        try:
            if np.all(np.isfinite(disabled_values)) and np.all(np.isfinite(baseline_values)):
                spl_disabled = make_interp_spline(layers, disabled_values, k=3)
                disabled_smooth = spl_disabled(layers_smooth)
                spl_baseline = make_interp_spline(layers, baseline_values, k=3)
                baseline_smooth = spl_baseline(layers_smooth)
            else:
                layers_smooth = layers
                disabled_smooth = disabled_values
                baseline_smooth = baseline_values
        except:
            layers_smooth = layers
            disabled_smooth = disabled_values
            baseline_smooth = baseline_values
    else:
        layers_smooth = layers
        disabled_smooth = disabled_values
        baseline_smooth = baseline_values

    # 绘制曲线
    ax.plot(layers_smooth, baseline_smooth, '-', color='#2ecc71', linewidth=2.5, alpha=0.95)
    ax.scatter(layers, baseline_values, s=40, color='#2ecc71', edgecolors='white', linewidth=1.0, alpha=0.9)
    ax.plot(layers_smooth, disabled_smooth, '-', color='#e74c3c', linewidth=2.5, alpha=0.95)
    ax.scatter(layers, disabled_values, s=40, color='#e74c3c', edgecolors='white', linewidth=1.0, alpha=0.9)

    # 填充区域
    ax.fill_between(layers_smooth, baseline_smooth, disabled_smooth,
                    where=(disabled_smooth > baseline_smooth), color='#e74c3c', alpha=0.2)
    ax.fill_between(layers_smooth, baseline_smooth, disabled_smooth,
                    where=(disabled_smooth <= baseline_smooth), color='#2ecc71', alpha=0.2)

    # 设置坐标轴
    all_values = np.concatenate([disabled_values, baseline_values])
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    ax.set_xlim(layers.min() - 1, layers.max() + 1)

    # 稀疏刻度
    n_layers = len(layers)
    x_tick_step = max(1, n_layers // 5)
    ax.set_xticks(np.arange(0, layers.max() + 1, x_tick_step))
    y_ticks = np.linspace(y_min, y_max, 4)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(alpha=0.2, linestyle='-', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 不添加xlabel（模型名称）

    plt.tight_layout()
    # 文件名使用模型显示名称
    output_file = OUTPUT_DIR / f'exp2_{model_display}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {model_display}: exp2_{model_display}.png")
    return True


# ==================== Exp4 函数 ====================

def load_exp4_data(model_config):
    """加载Exp4数据"""
    svd_file = EXP4_DIR / model_config['key'] / model_config['file']
    if not svd_file.exists():
        return None
    with open(svd_file, 'r') as f:
        return json.load(f)


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
        return np.array(layer_data[config['sv_key']])
    except (KeyError, TypeError):
        return None


def get_all_layers(data, config):
    """获取所有层ID"""
    if config.get('data_path'):
        all_keys = list(data[config['data_path']].keys())
    else:
        all_keys = list(data.keys())
    return sorted([int(k) for k in all_keys if k.isdigit()])


def select_layers(layer_ids, ma_layer):
    """选择要展示的层"""
    if len(layer_ids) <= 4:
        return layer_ids
    selected = [layer_ids[0]]
    mid = len(layer_ids) // 2
    if layer_ids[mid] not in selected:
        selected.append(layer_ids[mid])
    if ma_layer in layer_ids and ma_layer not in selected:
        selected.append(ma_layer)
    if layer_ids[-1] not in selected:
        selected.append(layer_ids[-1])
    return sorted(selected)[:4]


def save_exp4_single(model_config):
    """保存单个exp4模型的图（不带模型名称）"""
    model_key = model_config['key']
    model_display = model_config['display']
    
    data = load_exp4_data(model_config)
    if data is None:
        print(f"  ⚠ {model_display}: Data not found")
        return False

    layer_ids = get_all_layers(data, model_config)
    if not layer_ids:
        print(f"  ⚠ {model_display}: No layers found")
        return False

    ma_layer = model_config['ma_layer']
    layers_to_plot = select_layers(layer_ids, ma_layer)

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    max_x = 20

    for i, layer_id in enumerate(layers_to_plot):
        sv = get_layer_sv(data, layer_id, model_config)
        if sv is None:
            continue

        energy = sv ** 2
        cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
        x = np.arange(1, len(cumulative_energy) + 1)
        max_x = max(max_x, len(cumulative_energy))

        if layer_id == ma_layer:
            ax.plot(x, cumulative_energy, 'o-', color='#e74c3c', linewidth=2.5, markersize=4, alpha=0.9)
        else:
            ax.plot(x, cumulative_energy, 'o-', color=colors[i % len(colors)], linewidth=2, markersize=3, alpha=0.7)

    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(np.linspace(0, max_x, 5, dtype=int))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 不添加xlabel（模型名称）

    plt.tight_layout()
    # 文件名使用模型显示名称
    output_file = OUTPUT_DIR / f'exp4_{model_display}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {model_display}: exp4_{model_display}.png")
    return True


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("="*60)
    print("保存16张小图（不带模型名称）到 combined_figures_no_label")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Exp2 (8张) ---")
    for model in EXP2_MODELS:
        save_exp2_single(model)

    print("\n--- Exp4 (8张) ---")
    for model in EXP4_MODELS:
        save_exp4_single(model)

    print("\n✅ All done!")
    print(f"保存位置: {OUTPUT_DIR}")
