#!/usr/bin/env python3
"""
Exp3: 生成UV归因堆叠柱状图整图（8个模型合并）
- 保持原样式
- 增大坐标间隙、放大数字
- 去掉图例
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).resolve().parents[2] / 'results/experiments/exp3'
OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'results/plot_results/combined_figures_no_label'

# Model configuration
MODELS = ['gpt2', 'gptj_6b', 'bloom_7b1', 'falcon_7b', 'opt_6.7b', 'mistral_7b_v03', 'qwen2.5_7b', 'llama2_13b']
MODEL_DISPLAY_NAMES = {
    'gpt2': 'GPT-2',
    'gptj_6b': 'GPT-J-6B',
    'bloom_7b1': 'BLOOM-7B1',
    'falcon_7b': 'Falcon-7B',
    'opt_6.7b': 'OPT-7B',
    'mistral_7b_v03': 'Mistral-7B',
    'qwen2.5_7b': 'Qwen2.5-7B',
    'llama2_13b': 'LLaMA2-13B'
}


def load_all_data():
    """加载所有模型数据"""
    all_data = {}
    for model in MODELS:
        try:
            data_file = DATA_DIR / model / 'summary.json'
            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    baseline = data['attribution']['baseline']
                    if baseline is not None and not (isinstance(baseline, float) and np.isnan(baseline)):
                        all_data[model] = data
        except Exception as e:
            print(f"Warning: Cannot load {model}: {e}")
    return all_data


def create_stacked_bar_no_legend():
    """创建多模型归因占比堆叠柱状图（无图例、大字体、稀疏坐标）"""
    all_data = load_all_data()
    
    # 增大figsize确保PDF 100%缩放清晰
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

    # 准备数据
    models_with_data = [m for m in MODELS if m in all_data]
    model_names = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]

    u_proportions = []
    v_proportions = []
    inter_proportions = []

    for model in models_with_data:
        attribution = all_data[model]['attribution']

        # 计算归因占比
        u_abs = abs(attribution['u_attribution_pct'])
        v_abs = abs(attribution['v_attribution_pct'])
        inter_abs = abs(attribution['interaction_pct'])

        total_attribution = u_abs + v_abs + inter_abs

        if total_attribution > 0:
            u_prop = (u_abs / total_attribution) * 100
            v_prop = (v_abs / total_attribution) * 100
            inter_prop = (inter_abs / total_attribution) * 100
        else:
            u_prop = v_prop = inter_prop = 33.33

        u_proportions.append(u_prop)
        v_proportions.append(v_prop)
        inter_proportions.append(inter_prop)

    x = np.arange(len(models_with_data))
    width = 0.6

    # 绘制堆叠柱状图
    p1 = ax.bar(x, u_proportions, width,
                color='#3498db', alpha=0.85, edgecolor='white', linewidth=1.5)
    p2 = ax.bar(x, v_proportions, width, bottom=u_proportions,
                color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1.5)
    p3 = ax.bar(x, inter_proportions, width,
                bottom=np.array(u_proportions) + np.array(v_proportions),
                color='#f39c12', alpha=0.85, edgecolor='white', linewidth=1.5)

    # 添加百分比标签（放大字体）
    for i, (u, v, inter) in enumerate(zip(u_proportions, v_proportions, inter_proportions)):
        # U标签
        if u > 8:
            ax.text(i, u/2, f'{u:.1f}%', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='white'

        # V标签
        if v > 8:
            ax.text(i, u + v/2, f'{v:.1f}%', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='white'

        # 交互标签
        if inter > 8:
            ax.text(i, u + v + inter/2, f'{inter:.1f}%', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='white'

    # 添加模式标注（Independent vs Synergistic）
    for i, model in enumerate(models_with_data):
        interpretation = all_data[model]['attribution']['interpretation']
        if interpretation == 'independent':
            marker = '●'
            color = '#2ecc71'
            label_text = 'Indep'
        else:
            marker = '★'
            color = '#e67e22'
            label_text = 'Syner'

        ax.text(i, 102, marker, ha='center', fontsize=24, color=color, fontweight='bold'
        ax.text(i, 108, label_text, ha='center', fontsize=16, color=color, fontweight='bold'

    # 设置坐标轴（大字体、稀疏刻度）
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=20, fontweight='bold'
    ax.set_ylabel('Attribution Proportion (%)', fontsize=22, fontweight='bold'
    ax.set_ylim(0, 118)

    # 稀疏Y轴刻度
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='y', labelsize=20)

    # 添加100%参考线（不添加100%文字标签）
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.3)

    # 不添加图例

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--'
    ax.set_axisbelow(True)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    # 保存PNG和PDF
    output_png = OUTPUT_DIR / 'exp3_UV_Attribution_Stacked_Bar.png'
    output_pdf = OUTPUT_DIR / 'exp3_UV_Attribution_Stacked_Bar.pdf'
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white'
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white'
    plt.close()
    print(f"✓ 保存: {output_png.name}")
    print(f"✓ 保存: {output_pdf.name}")


def create_cross_model_comparison():
    """创建跨模型对比图（Baseline vs Ablated Average）- 无图例、大字体"""
    from scipy.interpolate import make_interp_spline
    
    all_data = load_all_data()
    
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

    # 准备数据
    models_with_data = [m for m in MODELS if m in all_data]
    model_names = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]
    x_pos = np.arange(len(models_with_data))

    # 两组数据
    baseline_values = [all_data[m]['attribution']['baseline'] for m in models_with_data]
    ablated_avg = [(all_data[m]['attribution']['ablate_u_mean'] +
                    all_data[m]['attribution']['ablate_v_mean'] +
                    all_data[m]['attribution']['ablate_both_mean']) / 3
                   for m in models_with_data]

    # 绘制柱状图（放大柱状宽度）
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, baseline_values, width,
                    color='#6495ED', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, ablated_avg, width,
                    color='#FFB6C1', alpha=0.8)

    # 绘制虚线曲线（加粗线条）
    x_smooth = np.linspace(x_pos[0], x_pos[-1], 500)

    if len(x_pos) > 3:
        spl_baseline = make_interp_spline(x_pos, baseline_values, k=3)
        baseline_smooth = spl_baseline(x_smooth)
        ax.plot(x_smooth, baseline_smooth, '--', color='#4169E1', linewidth=3.5, alpha=0.8)

        spl_ablated = make_interp_spline(x_pos, ablated_avg, k=3)
        ablated_smooth = spl_ablated(x_smooth)
        ax.plot(x_smooth, ablated_smooth, '--', color='#DC143C', linewidth=3.5, alpha=0.8)

    # 设置坐标轴（放大字体）
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=24, fontweight='bold'
    ax.set_ylabel('Massive Activation Value', fontsize=26, fontweight='bold'
    ax.tick_params(axis='y', labelsize=24)

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--'
    ax.set_ylim(bottom=0)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # 不添加图例

    plt.tight_layout()

    # 保存
    output_png = OUTPUT_DIR / 'exp3_Cross_Model_Comparison.png'
    output_pdf = OUTPUT_DIR / 'exp3_Cross_Model_Comparison.pdf'
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white'
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white'
    plt.close()
    print(f"✓ 保存: {output_png.name}")
    print(f"✓ 保存: {output_pdf.name}")


if __name__ == '__main__':
    print("="*60)
    print("Exp3: 生成整图（无图例、大字体）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成堆叠柱状图
    create_stacked_bar_no_legend()
    
    # 生成跨模型对比图
    create_cross_model_comparison()
    
    print("\n✅ All done!")
    print(f"保存位置: {OUTPUT_DIR}")
