#!/usr/bin/env python3
"""
Exp3 风格化可视化 - 学习模板风格
1. 消融对比图：柱状图 + 虚线曲线风格
2. 雷达图：多模型组合显示
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# 设置样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
EXP3_DIR = Path('PROJECT_ROOT/results/experiments/exp3')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp3_figures')

# 所有模型列表
MODELS = ['gpt2', 'gptj_6b', 'bloom_7b1', 'falcon_7b', 'opt_7b', 'mistral_7b_v03', 'qwen2.5_7b', 'llama2_13b']
MODEL_DISPLAY_NAMES = {
    'gpt2': 'GPT-2',
    'gptj_6b': 'GPT-J-6B',
    'bloom_7b1': 'BLOOM-7B1',
    'falcon_7b': 'Falcon-7B',
    'opt_7b': 'OPT-7B',
    'mistral_7b_v03': 'Mistral-7B',
    'qwen2.5_7b': 'Qwen2.5-7B',
    'llama2_13b': 'LLaMA2-13B'
}

# 颜色方案（8种不同颜色）
COLORS = [
    '#1f77b4',  # 蓝色 - GPT-2
    '#ff7f0e',  # 橙色 - GPT-J
    '#2ca02c',  # 绿色 - BLOOM
    '#d62728',  # 红色 - Falcon
    '#9467bd',  # 紫色 - OPT
    '#8c564b',  # 棕色 - Mistral
    '#e377c2',  # 粉色 - Qwen
    '#7f7f7f',  # 灰色 - LLaMA2
]

# 读取所有模型数据
all_data = {}
for model in MODELS:
    try:
        with open(EXP3_DIR / model / 'summary.json', 'r') as f:
            all_data[model] = json.load(f)
    except Exception as e:
        print(f"警告: 无法读取 {model} 数据: {e}")

print(f"成功读取 {len(all_data)} 个模型的数据")

# ========== 图表1: 单个模型消融对比图（学习模板1风格）==========
def create_ablation_comparison_stylized(model):
    """创建风格化的消融对比图"""
    if model not in all_data:
        return

    data = all_data[model]
    attribution = data['attribution']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 数据准备
    conditions = ['Baseline', 'Ablate U', 'Ablate V', 'Ablate Both']
    ma_values = [
        attribution['baseline'],
        attribution['ablate_u_mean'],
        attribution['ablate_v_mean'],
        attribution['ablate_both_mean']
    ]

    # 计算变化百分比用于虚线
    changes = [0] + [(ma_values[i] - ma_values[0]) / ma_values[0] * 100 for i in range(1, 4)]

    x_pos = np.arange(len(conditions))

    # 绘制柱状图
    bars = ax.bar(x_pos, ma_values, width=0.6,
                   color=['#6495ED', '#FFB6C1', '#FFB6C1', '#FFB6C1'],
                   alpha=0.7, edgecolor='black', linewidth=1.2,
                   label='MA Value')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, ma_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 绘制虚线曲线（显示变化趋势）
    ax2 = ax.twinx()
    line = ax2.plot(x_pos, changes, 'r--o', linewidth=2.5, markersize=8,
                    label='Change %', alpha=0.8)

    # 在虚线上标注百分比
    for i, (x, y) in enumerate(zip(x_pos[1:], changes[1:], )):
        ax2.text(x, y + (5 if y > 0 else -5), f'{y:.1f}%',
                ha='center', va='bottom' if y > 0 else 'top',
                fontsize=10, fontweight='bold', color='red')

    # 设置坐标轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=11, fontweight='bold')
    ax.set_ylabel('Massive Activation Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Change from Baseline (%)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 标题
    interpretation = attribution['interpretation']
    ax.set_title(f'{MODEL_DISPLAY_NAMES[model]} - Ablation Experiment\n'
                 f'Layer {data["layer"]} | Mode: {interpretation.capitalize()}',
                 fontsize=13, fontweight='bold', pad=15)

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / model
    plt.savefig(output_path / f'{model}_ablation_stylized.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{model}_ablation_stylized.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ {model}: ablation_stylized")
    plt.close()

# 为所有模型生成消融对比图
print("\n生成单个模型消融对比图...")
for model in MODELS:
    create_ablation_comparison_stylized(model)

# ========== 图表2: 多模型组合雷达图（学习模板2风格）==========
print("\n生成多模型组合雷达图...")

fig = plt.figure(figsize=(12, 10), dpi=300)
ax = fig.add_subplot(111, projection='polar')

# 雷达图维度
categories = ['U Attribution', 'V Attribution', 'U×V Interaction']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 为每个模型绘制雷达图
for i, (model, color) in enumerate(zip(MODELS, COLORS)):
    if model not in all_data:
        continue

    attribution = all_data[model]['attribution']

    # 获取归因百分比（取绝对值用于显示）
    values = [
        abs(attribution['u_attribution_pct']),
        abs(attribution['v_attribution_pct']),
        abs(attribution['interaction_pct'])
    ]
    values += values[:1]  # 闭合

    # 绘制线条和填充
    ax.plot(angles, values, 'o-', linewidth=2, color=color,
            label=MODEL_DISPLAY_NAMES[model], markersize=6)
    ax.fill(angles, values, alpha=0.15, color=color)

# 设置雷达图标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

# 设置径向刻度
max_value = max([abs(all_data[m]['attribution'][k])
                 for m in all_data
                 for k in ['u_attribution_pct', 'v_attribution_pct', 'interaction_pct']])
max_value = max(max_value, 100)  # 至少100%

# 动态设置刻度
if max_value > 200:
    tick_values = [0, 50, 100, 150, 200]
elif max_value > 100:
    tick_values = [0, 25, 50, 75, 100, 125, 150]
else:
    tick_values = [0, 20, 40, 60, 80, 100]

ax.set_ylim(0, max_value * 1.1)
ax.set_yticks(tick_values)
ax.set_yticklabels([f'{v}%' for v in tick_values], fontsize=10)
ax.grid(True, alpha=0.3)

# 标题
plt.title('U×V Attribution Comparison Across Models\nExp3: Interaction Analysis',
          fontsize=14, fontweight='bold', pad=25)

# 图例（放在底部）
ax.legend(loc='upper left', bbox_to_anchor=(0.85, -0.05), ncol=2,
          fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'exp3_all_models_radar.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'exp3_all_models_radar.pdf', dpi=300, bbox_inches='tight')
print(f"✓ 多模型雷达图: exp3_all_models_radar")

plt.close()

# ========== 图表3: 跨模型归因对比柱状图 ==========
print("\n生成跨模型归因对比图...")

fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

# 准备数据
models_with_data = [m for m in MODELS if m in all_data]
u_attrs = [all_data[m]['attribution']['u_attribution_pct'] for m in models_with_data]
v_attrs = [all_data[m]['attribution']['v_attribution_pct'] for m in models_with_data]
inter_attrs = [all_data[m]['attribution']['interaction_pct'] for m in models_with_data]

x = np.arange(len(models_with_data))
width = 0.25

# 绘制分组柱状图
bars1 = ax.bar(x - width, u_attrs, width, label='U Attribution',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x, v_attrs, width, label='V Attribution',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
bars3 = ax.bar(x + width, inter_attrs, width, label='U×V Interaction',
               color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

# 添加0轴参考线
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

# 设置标签
ax.set_ylabel('Attribution Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('U×V Attribution Comparison Across All Models\nExp3: Independent vs Synergistic Modes',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in models_with_data],
                    rotation=15, ha='right', fontsize=11)
ax.legend(fontsize=11, loc='best')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加模式标注
for i, model in enumerate(models_with_data):
    interpretation = all_data[model]['attribution']['interpretation']
    color = '#2ecc71' if interpretation == 'independent' else '#e67e22'
    marker = '●' if interpretation == 'independent' else '★'
    ax.text(i, ax.get_ylim()[0] - 10, marker,
            ha='center', fontsize=12, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'exp3_all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'exp3_all_models_comparison.pdf', dpi=300, bbox_inches='tight')
print(f"✓ 跨模型对比图: exp3_all_models_comparison")

plt.close()

print(f"\n{'='*60}")
print(f"✅ Exp3 风格化可视化完成！")
print(f"单个模型图: {len(models_with_data)} × 1 = {len(models_with_data)}个文件")
print(f"组合图: 2个（雷达图 + 对比图）")
print(f"总计: {len(models_with_data) + 2}个主图表")
print(f"{'='*60}")
