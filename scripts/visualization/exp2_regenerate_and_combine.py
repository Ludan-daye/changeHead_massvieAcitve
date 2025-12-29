#!/usr/bin/env python3
"""
Exp2: 重新生成每个模型的2D对比图（大字体、稀疏坐标），然后组合成4×2图组
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from pathlib import Path

# 配置
BASE_RESULTS_DIR = Path('PROJECT_ROOT/results/models')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')

# 8个模型配置
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_7b', 'display': 'OPT-7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]


def load_exp2_data(model_name):
    """加载Exp2数据"""
    summary_path = BASE_RESULTS_DIR / model_name / 'exp2b_mlp_layer_ablation' / 'summary.json'
    baseline_path = BASE_RESULTS_DIR / model_name / 'exp2b_mlp_layer_ablation' / 'baseline.json'

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


def plot_single_model(data, ax, model_display):
    """为单个模型绘制2D对比图（大字体、稀疏坐标、无图例）"""
    layers = data['layers']
    disabled_values = data['disabled_values']
    baseline_values = data['baseline_values']

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

    # 绘制baseline曲线
    ax.plot(layers_smooth, baseline_smooth, '-', color='#2ecc71',
           linewidth=2.5, alpha=0.95, zorder=2)
    ax.scatter(layers, baseline_values, s=40, color='#2ecc71',
              edgecolors='white', linewidth=1.0, zorder=3, alpha=0.9)

    # 绘制禁用值曲线
    ax.plot(layers_smooth, disabled_smooth, '-', color='#e74c3c',
           linewidth=2.5, alpha=0.95, zorder=2)
    ax.scatter(layers, disabled_values, s=40, color='#e74c3c',
              edgecolors='white', linewidth=1.0, zorder=3, alpha=0.9)

    # 填充区域
    ax.fill_between(layers_smooth, baseline_smooth, disabled_smooth,
                    where=(disabled_smooth > baseline_smooth),
                    color='#e74c3c', alpha=0.2, zorder=0)
    ax.fill_between(layers_smooth, baseline_smooth, disabled_smooth,
                    where=(disabled_smooth <= baseline_smooth),
                    color='#2ecc71', alpha=0.2, zorder=0)

    # 设置坐标轴范围
    all_values = np.concatenate([disabled_values, baseline_values])
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    
    x_min, x_max = layers.min(), layers.max()
    ax.set_xlim(x_min - 1, x_max + 1)

    # 稀疏的x轴刻度（最多5-6个刻度）
    n_layers = len(layers)
    if n_layers <= 15:
        x_tick_step = max(1, n_layers // 5)
    else:
        x_tick_step = max(1, n_layers // 5)
    x_ticks = np.arange(0, x_max + 1, x_tick_step)
    ax.set_xticks(x_ticks)
    
    # 稀疏的y轴刻度（最多4-5个刻度）
    y_ticks = np.linspace(y_min, y_max, 4)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

    # 大字体刻度标签
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 网格
    ax.grid(alpha=0.2, linestyle='-', linewidth=0.5)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 底部模型名称
    ax.set_xlabel(model_display, fontsize=13, fontweight='bold', labelpad=3)


def main():
    print("="*60)
    print("Exp2: Regenerate and Combine (Large Font, Sparse Ticks)")
    print("="*60 + "\n")

    # 创建4列2行的子图，减小纵向距离
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    axes = axes.flatten()

    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        # 加载数据
        data = load_exp2_data(model_key)
        
        if data is None:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Found',
                   ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            print(f"⚠ {model_display}: Data not found")
            continue

        # 绘制
        plot_single_model(data, ax, model_display)
        print(f"✓ {model_display}")

    # 紧凑布局
    plt.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.12,
                       hspace=0.25, wspace=0.18)

    # 保存
    output_png = OUTPUT_DIR / 'exp2_combined_large_font.png'
    output_pdf = OUTPUT_DIR / 'exp2_combined_large_font.pdf'

    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_png}")
    print(f"✅ Combined figure saved: {output_pdf}")

    plt.close()
    print("\n✅ All done!")


if __name__ == '__main__':
    main()
