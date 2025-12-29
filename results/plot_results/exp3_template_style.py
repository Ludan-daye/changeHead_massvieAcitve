#!/usr/bin/env python3
"""
Exp3 完全按照模板风格重新制作
模板特点：柱状图 + 虚线分布曲线叠加
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline

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

# 读取所有模型数据
all_data = {}
for model in MODELS:
    try:
        with open(EXP3_DIR / model / 'summary.json', 'r') as f:
            data = json.load(f)
            # 检查数据是否有效（不是NaN）
            baseline = data['attribution']['baseline']
            if baseline is None or (isinstance(baseline, float) and np.isnan(baseline)):
                print(f"警告: {model} 数据包含NaN，跳过")
                continue
            all_data[model] = data
    except Exception as e:
        print(f"警告: 无法读取 {model} 数据: {e}")

print(f"成功读取 {len(all_data)} 个有效模型的数据\n")

# ========== 单个模型图表 - 完全按照模板风格 ==========
def create_template_style_chart(model):
    """
    完全按照模板图风格创建图表
    模板特点：
    1. 柱状图（两组数据对比）
    2. 虚线曲线叠加显示分布趋势
    3. 图例在左上角
    4. 简洁清晰的坐标轴
    """
    if model not in all_data:
        return

    data = all_data[model]
    attribution = data['attribution']

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 准备数据
    # x轴：4种消融条件 - 去掉数学符号
    conditions = ['Baseline', 'Ablate U', 'Ablate V', 'Ablate Both']
    x_pos = np.arange(len(conditions))

    # y轴数据1：MA值
    ma_values = [
        attribution['baseline'],
        attribution['ablate_u_mean'],
        attribution['ablate_v_mean'],
        attribution['ablate_both_mean']
    ]

    # y轴数据2：归因贡献（用于对比）
    # 将归因百分比转换为绝对值用于显示
    u_contribution = attribution['u_main_effect']
    v_contribution = attribution['v_main_effect']
    interaction_contribution = attribution['interaction_effect']

    # 第二组数据：各种消融后的归因效应
    contributions = [
        0,  # Baseline作为参考点
        u_contribution,
        v_contribution,
        interaction_contribution
    ]

    # 检查数据有效性
    if any(np.isnan(v) or np.isinf(v) for v in ma_values):
        print(f"  警告: {model} MA值包含无效数据，跳过")
        return
    if any(np.isnan(v) or np.isinf(v) for v in contributions):
        print(f"  警告: {model} 归因值包含无效数据，跳过")
        return

    # 归一化contributions用于显示（映射到MA值的范围）
    if max(abs(c) for c in contributions) > 0:
        scale = max(ma_values) / max(abs(c) for c in contributions) * 0.8
        contributions_scaled = [c * scale for c in contributions]
    else:
        contributions_scaled = contributions

    # 绘制柱状图（两组数据）- 与跨模型图风格一致
    width = 0.25
    bars1 = ax.bar(x_pos - width/2, ma_values, width,
                    label=f'{MODEL_DISPLAY_NAMES[model]} MA Value',
                    color='#6495ED', alpha=0.7)  # 去掉边框

    bars2 = ax.bar(x_pos + width/2, [abs(c) for c in contributions_scaled], width,
                    label='Attribution Effect',
                    color='#FFB6C1', alpha=0.7)  # 改为粉色，去掉边框

    # 绘制虚线曲线（连接MA值）
    # 使用样条插值使曲线更平滑 - 增加插值点数量
    x_smooth = np.linspace(x_pos[0], x_pos[-1], 300)  # 从100增加到300

    # MA值曲线 - 使用3阶样条插值，更加平滑，与跨模型图风格一致
    if len(x_pos) > 3:
        spl_ma = make_interp_spline(x_pos, ma_values, k=3)
        ma_smooth = spl_ma(x_smooth)
        ax.plot(x_smooth, ma_smooth, '--', color='#4169E1', linewidth=2.5,
                label='MA Distribution', alpha=0.8)  # 简化标签
    elif len(x_pos) > 2:
        spl_ma = make_interp_spline(x_pos, ma_values, k=2)
        ma_smooth = spl_ma(x_smooth)
        ax.plot(x_smooth, ma_smooth, '--', color='#4169E1', linewidth=2.5,
                label='MA Distribution', alpha=0.8)
    else:
        ax.plot(x_pos, ma_values, '--', color='#4169E1', linewidth=2.5,
                label='MA Distribution', alpha=0.8)

    # 归因效应曲线 - 使用3阶样条插值，更加平滑，与跨模型图风格一致
    contrib_abs = [abs(c) for c in contributions_scaled]
    if len(x_pos) > 3:
        spl_contrib = make_interp_spline(x_pos, contrib_abs, k=3)
        contrib_smooth = spl_contrib(x_smooth)
        ax.plot(x_smooth, contrib_smooth, '--', color='#DC143C', linewidth=2.5,
                label='Attribution Distribution', alpha=0.8)
    elif len(x_pos) > 2:
        spl_contrib = make_interp_spline(x_pos, contrib_abs, k=2)
        contrib_smooth = spl_contrib(x_smooth)
        ax.plot(x_smooth, contrib_smooth, '--', color='#DC143C', linewidth=2.5,
                label='Attribution Distribution', alpha=0.8)
    else:
        ax.plot(x_pos, contrib_abs, '--', color='#DC143C', linewidth=2.5,
                label='Attribution Distribution', alpha=0.8)

    # 设置坐标轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel('Massive Activation Value', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ablation Conditions', fontsize=12, fontweight='bold')

    # 图例（左上角，按照模板位置）
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # 保存 - 用原标题作为文件名（去掉标题显示）
    interpretation = attribution['interpretation']
    title_text = f'{MODEL_DISPLAY_NAMES[model]} - UV Interaction Ablation Analysis Layer {data["layer"]} Mode {interpretation.capitalize()}'
    # 清理文件名中的特殊字符
    filename = title_text.replace(' ', '_').replace('×', '').replace('-', '_').replace(':', '')

    output_path = OUTPUT_DIR / model
    plt.savefig(output_path / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ {model}: {filename}")
    plt.close()

# ========== 跨模型对比图 - 模板风格 ==========
def create_cross_model_template_chart():
    """
    跨模型对比图 - 完全按照模板风格
    """
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    # 准备数据
    models_with_data = [m for m in MODELS if m in all_data]
    x_pos = np.arange(len(models_with_data))

    # 两组数据：
    # 1. Baseline MA值
    # 2. 消融后平均MA值
    baseline_values = [all_data[m]['attribution']['baseline'] for m in models_with_data]
    ablated_avg = [(all_data[m]['attribution']['ablate_u_mean'] +
                    all_data[m]['attribution']['ablate_v_mean'] +
                    all_data[m]['attribution']['ablate_both_mean']) / 3
                   for m in models_with_data]

    # 绘制柱状图 - 保持一致的风格
    width = 0.25
    bars1 = ax.bar(x_pos - width/2, baseline_values, width,
                    label='Baseline (Full W)',
                    color='#6495ED', alpha=0.7)  # 去掉边框

    bars2 = ax.bar(x_pos + width/2, ablated_avg, width,
                    label='Ablated Average',
                    color='#FFB6C1', alpha=0.7)  # 去掉边框

    # 绘制虚线曲线 - 更加平滑
    x_smooth = np.linspace(x_pos[0], x_pos[-1], 500)  # 从200增加到500

    # 使用3阶样条插值，曲线更加平滑
    if len(x_pos) > 3:
        spl_baseline = make_interp_spline(x_pos, baseline_values, k=3)  # k=3更平滑
        baseline_smooth = spl_baseline(x_smooth)
        ax.plot(x_smooth, baseline_smooth, '--', color='#4169E1', linewidth=2.5,
                label='Baseline Distribution', alpha=0.8)

        spl_ablated = make_interp_spline(x_pos, ablated_avg, k=3)  # k=3更平滑
        ablated_smooth = spl_ablated(x_smooth)
        ax.plot(x_smooth, ablated_smooth, '--', color='#DC143C', linewidth=2.5,
                label='Ablated Distribution', alpha=0.8)
    elif len(x_pos) > 2:
        spl_baseline = make_interp_spline(x_pos, baseline_values, k=2)
        baseline_smooth = spl_baseline(x_smooth)
        ax.plot(x_smooth, baseline_smooth, '--', color='#4169E1', linewidth=2.5,
                label='Baseline Distribution', alpha=0.8)

        spl_ablated = make_interp_spline(x_pos, ablated_avg, k=2)
        ablated_smooth = spl_ablated(x_smooth)
        ax.plot(x_smooth, ablated_smooth, '--', color='#DC143C', linewidth=2.5,
                label='Ablated Distribution', alpha=0.8)
    else:
        ax.plot(x_pos, baseline_values, '--', color='#4169E1', linewidth=2.5,
                label='Baseline Distribution', alpha=0.8)
        ax.plot(x_pos, ablated_avg, '--', color='#DC143C', linewidth=2.5,
                label='Ablated Distribution', alpha=0.8)

    # 设置坐标轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in models_with_data],
                        rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Massive Activation Value', fontsize=12, fontweight='bold')

    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # 保存 - 用原标题作为文件名（去掉标题显示）
    title_text = 'UV Interaction Ablation Cross_Model Comparison Exp3 Baseline vs Ablated Average'
    filename = title_text.replace(' ', '_').replace('×', '').replace('-', '_').replace(':', '')

    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 跨模型对比图: {filename}")
    plt.close()

# ========== 多模型堆叠柱状图 ==========
def create_multi_model_stacked_bar():
    """
    创建多模型归因占比堆叠柱状图
    使用堆叠柱状图展示U、V、交互效应的相对占比
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # 准备数据
    models_with_data = [m for m in MODELS if m in all_data]
    model_names = [MODEL_DISPLAY_NAMES[m] for m in models_with_data]

    u_proportions = []
    v_proportions = []
    inter_proportions = []

    for model in models_with_data:
        attribution = all_data[model]['attribution']

        # 计算归因占比（相对于总归因的比例）
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
    p1 = ax.bar(x, u_proportions, width, label='U Attribution',
                color='#3498db', alpha=0.85, edgecolor='white', linewidth=1.5)
    p2 = ax.bar(x, v_proportions, width, bottom=u_proportions,
                label='V Attribution', color='#e74c3c', alpha=0.85,
                edgecolor='white', linewidth=1.5)
    p3 = ax.bar(x, inter_proportions, width,
                bottom=np.array(u_proportions) + np.array(v_proportions),
                label='U×V Interaction', color='#f39c12', alpha=0.85,
                edgecolor='white', linewidth=1.5)

    # 添加百分比标签
    for i, (u, v, inter) in enumerate(zip(u_proportions, v_proportions, inter_proportions)):
        # U标签
        if u > 8:  # 只有足够大才显示
            ax.text(i, u/2, f'{u:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

        # V标签
        if v > 8:
            ax.text(i, u + v/2, f'{v:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

        # 交互标签
        if inter > 8:
            ax.text(i, u + v + inter/2, f'{inter:.1f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

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

        ax.text(i, 102, marker, ha='center', fontsize=14, color=color, fontweight='bold')
        ax.text(i, 107, label_text, ha='center', fontsize=8, color=color, fontweight='bold')

    # 设置标签和标题
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylabel('Attribution Proportion (%)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 115)

    # 添加100%参考线
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.text(len(models_with_data)-0.5, 100, '100%', fontsize=9, va='bottom')

    # 图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=3)

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # 保存
    filename = 'UV_Attribution_Stacked_Bar_All_Models'
    plt.savefig(OUTPUT_DIR / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 多模型堆叠柱状图: {filename}")
    plt.close()

# 执行生成
print("=" * 60)
print("开始生成模板风格图表...")
print("=" * 60)
print()

# 生成单个模型图表
print("1. 生成单个模型图表...")
for model in MODELS:
    if model in all_data:
        create_template_style_chart(model)

print()

# 生成跨模型对比图
print("2. 生成跨模型对比图...")
create_cross_model_template_chart()

print()

# 生成多模型堆叠柱状图
print("3. 生成多模型堆叠柱状图...")
create_multi_model_stacked_bar()

print()
print("=" * 60)
print("✅ 所有模板风格图表生成完成！")
print(f"单个模型图: {len(all_data)} 个")
print(f"跨模型图: 1 个")
print(f"堆叠柱状图: 1 个")
print(f"总计: {len(all_data) + 2} 个主图表 × 2 格式 = {(len(all_data) + 2) * 2} 个文件")
print("=" * 60)
