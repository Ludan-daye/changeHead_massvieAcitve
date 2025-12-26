#!/usr/bin/env python3
"""
分析MLP各层对MA的贡献度，确定起源层
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_layer_contribution(model_name, base_dir='results/models'):
    """分析各层贡献度"""

    # 加载数据
    summary_path = Path(base_dir) / model_name / 'exp2b_mlp_layer_ablation' / 'summary.json'
    baseline_path = Path(base_dir) / model_name / 'exp2b_mlp_layer_ablation' / 'baseline.json'

    with open(summary_path, 'r') as f:
        summary = json.load(f)
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    # 提取数据
    ablation = summary['ablation']
    baseline_results = baseline['results']

    layers = sorted([int(k) for k in ablation.keys()])

    # 获取baseline各层的MA值（逐层累积），过滤掉NaN值
    baseline_values = []
    valid_layers = []
    for layer in layers:
        mean_val = baseline_results[str(layer)]['mean']
        # 过滤掉NaN值的层
        if not np.isnan(mean_val):
            baseline_values.append(mean_val)
            valid_layers.append(layer)
        else:
            print(f"⚠️  跳过Layer {layer}（数据为NaN）")

    baseline_values = np.array(baseline_values)
    layers = np.array(valid_layers)

    # 获取ablation各层的最终MA值（禁用该层后的最终值），只保留有效层
    ablation_values = np.array([ablation[str(l)] for l in layers])

    # 确定baseline的最终值（过滤输出层异常）
    # 如果最后一层显著小于倒数第二层，说明是输出层，不使用
    if baseline_values[-1] < baseline_values[-2] * 0.5:
        baseline_final = baseline_values[-2]
        print(f"⚠️  检测到输出层异常（Layer {layers[-1]}: {baseline_values[-1]:.1f}），使用Layer {layers[-2]}作为最终值")
    else:
        baseline_final = baseline_values.max()

    print("\n" + "="*80)
    print(f"📊 {model_name.upper()} - MLP层贡献度分析")
    print("="*80)

    print(f"\n1️⃣  Baseline最终MA值: {baseline_final:.2f}")

    # 计算三种贡献度指标

    # 指标1: 绝对贡献 = baseline_final - ablation[i]
    # 含义: 禁用该层后MA下降了多少（正值=促进，负值=抑制）
    absolute_contribution = baseline_final - ablation_values

    # 指标2: 相对贡献百分比
    relative_contribution = (absolute_contribution / baseline_final) * 100

    # 指标3: 该层直接产出的MA值
    direct_output = baseline_values.copy()
    direct_output[1:] = baseline_values[1:] - baseline_values[:-1]  # 增量

    print("\n" + "="*80)
    print("2️⃣  各层贡献度分析")
    print("="*80)
    print(f"{'Layer':<8} {'Direct':<12} {'Ablation':<12} {'Absolute':<12} {'Relative':<12} {'Type':<15}")
    print(f"{'ID':<8} {'Output':<12} {'Final':<12} {'Contrib':<12} {'Contrib(%)':<12} {'':<15}")
    print("-"*80)

    for i, layer in enumerate(layers):
        contrib_type = "🟢 Promotion" if absolute_contribution[i] > 0 else "🔴 Suppression"
        print(f"{layer:<8} {direct_output[i]:<12.1f} {ablation_values[i]:<12.1f} "
              f"{absolute_contribution[i]:<12.1f} {relative_contribution[i]:<12.2f} {contrib_type:<15}")

    # 找出关键层
    print("\n" + "="*80)
    print("3️⃣  关键层识别")
    print("="*80)

    # Top 3 贡献层（绝对值）
    top_contrib_indices = np.argsort(absolute_contribution)[::-1][:3]
    print("\n🏆 Top 3 促进层（贡献最大）:")
    for rank, idx in enumerate(top_contrib_indices, 1):
        if absolute_contribution[idx] > 0:
            print(f"   {rank}. Layer {layers[idx]}: "
                  f"贡献 {absolute_contribution[idx]:.1f} ({relative_contribution[idx]:.2f}%)")

    # Top 3 抑制层
    bottom_contrib_indices = np.argsort(absolute_contribution)[:3]
    print("\n🔴 Top 3 抑制层:")
    for rank, idx in enumerate(bottom_contrib_indices, 1):
        if absolute_contribution[idx] < 0:
            print(f"   {rank}. Layer {layers[idx]}: "
                  f"抑制 {absolute_contribution[idx]:.1f} ({relative_contribution[idx]:.2f}%)")

    # 分析Layer 0的特殊性
    print("\n" + "="*80)
    print("4️⃣  Layer 0 作为起源层的证据")
    print("="*80)

    layer0_direct = direct_output[0]
    layer0_contrib = absolute_contribution[0]
    layer0_relative = relative_contribution[0]

    print(f"\n📌 Layer 0 分析:")
    print(f"   • 直接产出: {layer0_direct:.2f}")
    print(f"   • 禁用后最终MA: {ablation_values[0]:.2f}")
    print(f"   • 绝对贡献: {layer0_contrib:.2f} ({layer0_relative:.2f}%)")

    # 查找Layer 0在正贡献层中的排名
    layer0_rank_in_positive = np.where(top_contrib_indices == 0)[0]
    if len(layer0_rank_in_positive) > 0:
        print(f"   • 贡献排名: #{layer0_rank_in_positive[0] + 1}")
    else:
        print(f"   • 贡献排名: 不在Top促进层（Layer 0是抑制层）")

    # 判断是否为起源
    is_origin = False
    reasons = []

    # 证据1: Layer 0是第一个产生MA的层
    if layer0_direct > 0:
        reasons.append(f"✅ Layer 0是第一个产生MA的层（产出{layer0_direct:.1f}）")
        is_origin = True
    else:
        reasons.append(f"❌ Layer 0直接产出很小或为负（{layer0_direct:.1f}）")

    # 证据2: Layer 0的贡献在前3名
    if len(layer0_rank_in_positive) > 0 and layer0_rank_in_positive[0] < 3:
        rank = layer0_rank_in_positive[0] + 1
        reasons.append(f"✅ Layer 0贡献排名第{rank}（前3名）")
    elif layer0_contrib < 0:
        reasons.append(f"❌ Layer 0是抑制层（贡献{layer0_contrib:.1f}）")

    # 证据3: 禁用Layer 0导致显著下降（正贡献）
    if layer0_contrib > baseline_final * 0.03:  # 超过3%
        reasons.append(f"✅ 禁用Layer 0导致MA显著下降（{layer0_relative:.2f}%）")
    elif layer0_contrib < 0:
        reasons.append(f"❌ 禁用Layer 0导致MA上升（{layer0_relative:.2f}%），说明它抑制MA")

    # 证据4: 早期层（0-3）的累积贡献
    early_layers_contrib = absolute_contribution[:min(4, len(layers))].sum()
    early_layers_relative = (early_layers_contrib / baseline_final) * 100
    reasons.append(f"✅ 早期层(0-3)累积贡献{early_layers_contrib:.1f} ({early_layers_relative:.2f}%)")

    print("\n📊 判断依据:")
    for reason in reasons:
        print(f"   {reason}")

    if is_origin:
        print(f"\n✅ 结论: Layer 0 是MA的起源层")
    else:
        print(f"\n⚠️  结论: Layer 0 可能不是主要起源层")

    # 返回分析结果
    return {
        'model': model_name,
        'layers': layers,
        'baseline_final': baseline_final,
        'baseline_values': baseline_values.tolist(),
        'ablation_values': ablation_values.tolist(),
        'direct_output': direct_output.tolist(),
        'absolute_contribution': absolute_contribution.tolist(),
        'relative_contribution': relative_contribution.tolist(),
        'top_contrib_layers': [int(layers[i]) for i in top_contrib_indices],
        'early_layers_contrib_pct': float(early_layers_relative),
        'is_layer0_origin': is_origin
    }


def plot_contribution_comparison(analysis_result, outdir):
    """生成贡献度对比图（拆分为2个独立图）"""
    model = analysis_result['model']
    layers = np.array(analysis_result['layers'])
    direct_output = np.array(analysis_result['direct_output'])
    absolute_contrib = np.array(analysis_result['absolute_contribution'])

    # ========== 图1: 直接产出 vs 绝对贡献 ==========
    fig1, ax1 = plt.subplots(figsize=(10, 7), dpi=300)

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax1.bar(x - width/2, direct_output, width, label='Direct Output',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, absolute_contrib, width, label='Absolute Contribution',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.8)

    ax1.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
    ax1.set_ylabel('MA Value', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(0, color='black', linewidth=1.0)

    # 标注Layer 0
    ax1.annotate('Layer 0\n(Origin)', xy=(0, direct_output[0]),
                xytext=(0.5, direct_output[0] + 100),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    outfile1_png = Path(outdir) / f'{model}_direct_output_vs_contribution.png'
    outfile1_pdf = Path(outdir) / f'{model}_direct_output_vs_contribution.pdf'
    fig1.savefig(outfile1_png, dpi=400, bbox_inches='tight')
    fig1.savefig(outfile1_pdf, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Generated: {outfile1_png.name}")

    # ========== 图2: 累积贡献百分比 ==========
    fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=300)

    cumsum_contrib = np.cumsum(absolute_contrib)
    total_contrib = cumsum_contrib[-1]
    cumsum_pct = (cumsum_contrib / analysis_result['baseline_final']) * 100

    ax2.plot(layers, cumsum_pct, '-o', linewidth=3, markersize=8,
            color='#e74c3c', label='Cumulative Contribution %',
            markerfacecolor='white', markeredgewidth=2)

    # 根据数据正负值智能填充
    if cumsum_pct.min() >= 0:
        # 全部为正值，从0填充
        ax2.fill_between(layers, 0, cumsum_pct, alpha=0.25, color='#e74c3c')
    else:
        # 存在负值，从最小值填充到数据
        fill_baseline = min(0, cumsum_pct.min())
        ax2.fill_between(layers, fill_baseline, cumsum_pct, alpha=0.25, color='#e74c3c')

    # 标注关键阈值（根据数据正负自动调整）
    # 如果数据主要是负值，使用负阈值；否则使用正阈值
    data_is_negative = cumsum_pct.max() < 0  # 最大值都是负的，说明全是负数据

    if data_is_negative:
        # 负值数据，使用负阈值
        threshold_50 = -50
        threshold_80 = -80
        # 找到最接近阈值的层（绝对值意义上）
        idx_50 = np.where(cumsum_pct <= threshold_50)[0]
        idx_80 = np.where(cumsum_pct <= threshold_80)[0]
    else:
        # 正值数据，使用正阈值
        threshold_50 = 50
        threshold_80 = 80
        idx_50 = np.where(cumsum_pct >= threshold_50)[0]
        idx_80 = np.where(cumsum_pct >= threshold_80)[0]

    ax2.axhline(threshold_50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'{threshold_50}% threshold')
    ax2.axhline(threshold_80, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'{threshold_80}% threshold')

    # 标注达到阈值的层
    if len(idx_50) > 0:
        layer_50 = layers[idx_50[0]]
        ax2.axvline(layer_50, color='orange', linestyle=':', linewidth=1.0, alpha=0.5)
        ax2.text(layer_50, threshold_50, f'  Layer {layer_50}', fontsize=10, color='orange',
                fontweight='bold')

    if len(idx_80) > 0:
        layer_80 = layers[idx_80[0]]
        ax2.axvline(layer_80, color='green', linestyle=':', linewidth=1.0, alpha=0.5)
        ax2.text(layer_80, threshold_80, f'  Layer {layer_80}', fontsize=10, color='green',
                fontweight='bold')

    # 添加0轴参考线（如果数据跨越正负区域）
    if cumsum_pct.min() < 0:
        ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=2)

    ax2.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Contribution (%)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, framealpha=0.9, loc='best')
    ax2.grid(alpha=0.3, linestyle='--')

    # 动态设置y轴范围，确保所有数据可见（包括负值）
    y_min = min(cumsum_pct.min(), 0)
    y_max = max(cumsum_pct.max(), 105)
    # 为负值和正值都留出边距
    y_min_margin = y_min * 1.1 if y_min < 0 else y_min - abs(y_max) * 0.05
    y_max_margin = y_max * 1.05
    ax2.set_ylim(y_min_margin, y_max_margin)

    plt.tight_layout()

    outfile2_png = Path(outdir) / f'{model}_cumulative_contribution_analysis.png'
    outfile2_pdf = Path(outdir) / f'{model}_cumulative_contribution_analysis.pdf'
    fig2.savefig(outfile2_png, dpi=400, bbox_inches='tight')
    fig2.savefig(outfile2_pdf, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Generated: {outfile2_png.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze layer contribution')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--output-dir', type=str, default='results/plot_results/exp2_figures',
                       help='Output directory')
    args = parser.parse_args()

    # 分析
    result = analyze_layer_contribution(args.model)

    # 生成可视化
    outdir = Path(args.output_dir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    plot_contribution_comparison(result, outdir)

    # 保存分析结果
    analysis_file = outdir / f'{args.model}_contribution_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ 保存分析结果: {analysis_file}")

    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
