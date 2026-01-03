#!/usr/bin/env python3
"""
Generate 8 individual Exp2 figures
- Consistent with combined figure style
- Fix x-axis label overlapping issue
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration - use relative paths from repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results/experiments/exp2'
OUTPUT_DIR = PROJECT_ROOT / 'results/plot_results/exp2_figures'

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
COLOR_BASELINE = '#2ECC71'    # 绿色 - Baseline
COLOR_ABLATED = '#E74C3C'     # 红色 - Ablated
COLOR_FILL = '#FFB6B0'        # 浅红色 - 填充区域


def load_exp2_data(model_key):
    """加载Exp2数据"""
    summary_file = RESULTS_DIR / model_key / 'summary.json'

    # 特殊情况：LLaMA2数据在另一个位置
    if not summary_file.exists() and 'llama' in model_key:
        alt_path = PROJECT_ROOT / 'results/archive/by_model/llama2_13b/exp2_llama2_13b/layer_3_results.json'
        if alt_path.exists():
            # 这个文件格式不同，需要转换
            with open(alt_path, 'r') as f:
                raw_data = json.load(f)
            # 转换为标准格式
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


def create_individual_figure(model_config):
    """为单个模型创建图"""
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
    ma_ablated = [ablation[str(layer)] for layer in layers]

    # 计算baseline
    baseline = max(ma_ablated)
    ma_baseline = [baseline] * len(layers)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制图形
    ax.plot(layers, ma_baseline, color=COLOR_BASELINE, linewidth=3,
           linestyle='--', label='Baseline', zorder=3)
    ax.plot(layers, ma_ablated, color=COLOR_ABLATED, linewidth=3,
           linestyle='-', label='Layer Disabled', zorder=4)

    # 填充区域
    ax.fill_between(layers, ma_baseline, ma_ablated,
                    color=COLOR_FILL, alpha=0.4, zorder=1)

    # 标记关键层
    if critical_layer < len(layers):
        ax.axvline(x=critical_layer, color='darkgray', linestyle=':',
                  linewidth=2, alpha=0.8, zorder=2)

    # 设置标题
    ax.set_title(model_display, fontsize=18, fontweight='bold', pad=15)

    # 设置坐标轴标签
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold', labelpad=8)
    ax.set_ylabel('MA Value', fontsize=14, fontweight='bold', labelpad=8)

    # 横坐标刻度 - 修复末尾数字重叠问题
    n_layers = len(layers)
    max_layer = layers[-1]
    
    if n_layers <= 12:
        tick_step = 3
    elif n_layers <= 30:
        tick_step = 6
    else:
        tick_step = 10

    tick_positions = list(range(0, max_layer + 1, tick_step))
    
    # 检查最后一个刻度与末尾是否太近，如果太近则移除最后一个刻度
    if tick_positions and max_layer not in tick_positions:
        if max_layer - tick_positions[-1] < tick_step * 0.6:
            tick_positions[-1] = max_layer
        else:
            tick_positions.append(max_layer)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=12, fontweight='bold')

    # 纵坐标字体
    ax.tick_params(axis='y', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='x', width=1.5, length=6)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=1, color='gray')

    # 边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # 调整布局
    plt.tight_layout()

    # 保存
    output_file_png = OUTPUT_DIR / f'exp2_{model_key}.png'
    output_file_pdf = OUTPUT_DIR / f'exp2_{model_key}.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"✓ {model_display} saved: {output_file_png.name}")

    plt.close()
    return True


def main():
    """主函数"""
    print("="*60)
    print("生成8个单独的Exp2小图")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for model_config in MODEL_CONFIGS:
        if create_individual_figure(model_config):
            success_count += 1

    print(f"\n✅ 完成! 成功生成 {success_count}/8 个图")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
