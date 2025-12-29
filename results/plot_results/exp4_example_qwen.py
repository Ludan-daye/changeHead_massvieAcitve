#!/usr/bin/env python3
"""
Exp4 示例图表 - Qwen2.5-7B
展示Attention层SVD分析的可视化方案
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
EXP4_DIR = Path('PROJECT_ROOT/results/experiments/exp4')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp4_figures/qwen2.5_7b')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取数据
model = 'qwen2.5_7b'
with open(EXP4_DIR / model / 'svd_analysis.json', 'r') as f:
    data = json.load(f)

# MA生成层
ma_layer = 3

print("="*80)
print(f"生成 {model} 的示例图表...")
print(f"MA生成层: Layer {ma_layer}")
print("="*80)

# ========== 图1: 奇异值衰减曲线 ==========
def create_singular_value_decay():
    """对比不同层的奇异值衰减"""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # 选择几个有代表性的层
    layers_to_plot = ['0', '1', '2', '3', '26']
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']  # Layer 3用红色突出

    for i, layer_id in enumerate(layers_to_plot):
        if layer_id in data:
            sv = data[layer_id]['singular_values']
            x = np.arange(1, len(sv) + 1)

            # MA层使用更粗的线和特殊标记
            if int(layer_id) == ma_layer:
                ax.plot(x, sv, 'o-', color=colors[i], linewidth=3,
                       markersize=6, label=f'Layer {layer_id} (MA Layer)',
                       alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            else:
                ax.plot(x, sv, 'o-', color=colors[i], linewidth=2,
                       markersize=4, label=f'Layer {layer_id}', alpha=0.7)

    ax.set_xlabel('Singular Value Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Singular Value Magnitude', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(1, 21))

    plt.tight_layout()

    filename = f'{model}_Singular_Value_Decay_Comparison'
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 图1: {filename}")
    plt.close()

# ========== 图2: σ1/σ2比率对比 ==========
def create_ratio_comparison():
    """对比不同层的σ1/σ2比率"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    layers = []
    ratios = []
    colors_list = []

    for layer_id in ['0', '1', '2', '3', '26']:
        if layer_id in data:
            layers.append(f'Layer {layer_id}')
            ratios.append(data[layer_id]['ratio_s1_s2'])
            # MA层用红色，其他用蓝色
            if int(layer_id) == ma_layer:
                colors_list.append('#e74c3c')
            else:
                colors_list.append('#6495ED')

    x_pos = np.arange(len(layers))
    bars = ax.bar(x_pos, ratios, color=colors_list, alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.6)

    # 在柱子上方添加数值
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 添加参考线（ratio=1表示无主导方向）
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
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 图2: {filename}")
    plt.close()

# ========== 图3: 奇异值能量分布 ==========
def create_energy_distribution():
    """显示前k个奇异值的累积能量占比"""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    layers_to_plot = ['0', '1', '2', '3', '26']
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']

    for i, layer_id in enumerate(layers_to_plot):
        if layer_id in data:
            sv = np.array(data[layer_id]['singular_values'])
            # 计算累积能量占比
            energy = sv ** 2
            cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
            x = np.arange(1, len(cumulative_energy) + 1)

            if int(layer_id) == ma_layer:
                ax.plot(x, cumulative_energy, 'o-', color=colors[i], linewidth=3,
                       markersize=6, label=f'Layer {layer_id} (MA Layer)',
                       alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            else:
                ax.plot(x, cumulative_energy, 'o-', color=colors[i], linewidth=2,
                       markersize=4, label=f'Layer {layer_id}', alpha=0.7)

    # 添加参考线（90%能量）
    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='90% Energy')

    ax.set_xlabel('Number of Singular Values', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Energy (%)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(1, 21))
    ax.set_ylim(0, 105)

    plt.tight_layout()

    filename = f'{model}_Cumulative_Energy_Distribution'
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 图3: {filename}")
    plt.close()

# ========== 图4: MA层奇异值大小 ==========
def create_ma_layer_singular_values():
    """MA层的奇异值大小柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    layer_data = data['3']
    sv = np.array(layer_data['singular_values'])
    x = np.arange(1, len(sv) + 1)

    ax.bar(x, sv, color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注σ1和σ2
    ax.text(1, sv[0], f'σ1={sv[0]:.2f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='darkred')
    ax.text(2, sv[1], f'σ2={sv[1]:.2f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='darkred')

    plt.tight_layout()

    filename = f'{model}_Layer{ma_layer}_Singular_Values'
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 图4: {filename}")
    plt.close()

# ========== 图5: MA层归一化奇异值 ==========
def create_ma_layer_normalized():
    """MA层的归一化奇异值（相对于σ1）"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    layer_data = data['3']
    sv = np.array(layer_data['singular_values'])
    x = np.arange(1, len(sv) + 1)

    # 归一化奇异值（相对于σ1）
    sv_normalized = sv / sv[0]
    ax.plot(x, sv_normalized, 'o-', color='#3498db', linewidth=2.5,
            markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Magnitude (relative to σ1)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=1/layer_data['ratio_s1_s2'], color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'σ2/σ1 = {1/layer_data["ratio_s1_s2"]:.3f}')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    filename = f'{model}_Layer{ma_layer}_Normalized_Spectrum'
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 图5: {filename}")
    plt.close()

# 生成所有图表
print("\n生成图表:")
create_singular_value_decay()
create_ratio_comparison()
create_energy_distribution()
create_ma_layer_singular_values()
create_ma_layer_normalized()

print("\n" + "="*80)
print(f"✅ {model} 示例图表生成完成！")
print(f"共生成 5 个图表 × 2 格式 = 10 个文件")
print(f"保存位置: {OUTPUT_DIR}")
print("="*80)
