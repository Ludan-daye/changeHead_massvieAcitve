#!/usr/bin/env python3
"""
生成所有模型的Exp2 MLP层贡献度可视化
为每个模型生成5种图表类型，按模型分类存放
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def setup_style():
    """设置学术风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def load_summary(model_name, base_dir='results/models'):
    """加载模型的summary.json"""
    path = Path(base_dir) / model_name / 'exp2b_mlp_layer_ablation' / 'summary.json'
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    # 解析ablation数据
    ablation = data.get('ablation', {})
    layers = sorted([int(k) for k in ablation.keys()])
    values = [ablation[str(l)] for l in layers]

    return {
        'model': model_name,
        'layers': np.array(layers),
        'values': np.array(values),
        'baseline': data.get('baseline', {}),
        'contribution': data.get('contribution', {})
    }


def plot_3d_contribution(data, outdir):
    """生成3D MLP贡献图"""
    model = data['model']
    layers = data['layers']
    values = data['values']

    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # 创建网格
    x = layers
    y = np.zeros_like(layers)
    z = values

    # 绘制3D柱状图
    dx = 0.8
    dy = 0.5
    colors = plt.cm.viridis(z / z.max())

    ax.bar3d(x, y, np.zeros_like(z), dx, dy, z, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Layer', fontsize=12, labelpad=10)
    ax.set_ylabel('')
    ax.set_zlabel('MA Value (Top1)', fontsize=12, labelpad=10)
    ax.set_title(f'{model.upper()} - MLP Layer Contribution (3D)', fontsize=14, fontweight='bold')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    outfile_png = outdir / f'{model}_exp2_3d_mlp_contribution.png'
    outfile_pdf = outdir / f'{model}_exp2_3d_mlp_contribution.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated: {outfile_png.name}")


def plot_dual_bar_smooth(data, outdir):
    """生成双柱状平滑图"""
    model = data['model']
    layers = data['layers']
    values = data['values']

    # 平滑处理
    smoothed = gaussian_filter1d(values, sigma=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 左图：原始柱状图
    colors = plt.cm.coolwarm(values / values.max())
    ax1.bar(layers, values, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('MA Value (Top1)', fontsize=12)
    ax1.set_title(f'{model.upper()} - Raw Ablation Values', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 右图：平滑后的柱状图
    colors_smooth = plt.cm.coolwarm(smoothed / smoothed.max())
    ax2.bar(layers, smoothed, color=colors_smooth, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Smoothed MA Value', fontsize=12)
    ax2.set_title(f'{model.upper()} - Smoothed Ablation Values', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    outfile_png = outdir / f'{model}_exp2_dual_bar_smooth.png'
    outfile_pdf = outdir / f'{model}_exp2_dual_bar_smooth.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated: {outfile_png.name}")


def plot_line_smooth_v2(data, outdir):
    """生成平滑线图v2"""
    model = data['model']
    layers = data['layers']
    values = data['values']

    # 多级平滑
    smooth1 = gaussian_filter1d(values, sigma=0.8)
    smooth2 = gaussian_filter1d(values, sigma=2.0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 原始数据（散点）
    ax.scatter(layers, values, color='#e74c3c', s=60, alpha=0.6, label='Raw', zorder=3, edgecolors='black', linewidth=0.5)

    # 轻度平滑
    ax.plot(layers, smooth1, color='#3498db', linewidth=2.5, alpha=0.9, label='Smooth (σ=0.8)', zorder=2)

    # 重度平滑
    ax.plot(layers, smooth2, color='#2ecc71', linewidth=2.5, alpha=0.9, label='Smooth (σ=2.0)', linestyle='--', zorder=1)

    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('MA Value (Top1)', fontsize=13)
    ax.set_title(f'{model.upper()} - MLP Layer Ablation (Multi-level Smoothing)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    outfile_png = outdir / f'{model}_exp2_line_smooth_v2.png'
    outfile_pdf = outdir / f'{model}_exp2_line_smooth_v2.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated: {outfile_png.name}")


def plot_type3_percentage_change(data, outdir):
    """生成百分比变化图"""
    model = data['model']
    layers = data['layers']
    values = data['values']

    # 计算相对于最大值的百分比
    max_val = values.max()
    percentages = (values / max_val) * 100

    # 计算与前一层的变化率
    changes = np.zeros_like(percentages)
    changes[1:] = ((values[1:] - values[:-1]) / values[:-1]) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

    # 上图：相对百分比
    colors = ['#2ecc71' if p >= 95 else '#f39c12' if p >= 85 else '#e74c3c' for p in percentages]
    ax1.bar(layers, percentages, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax1.axhline(100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Max (100%)')
    ax1.axhline(95, color='orange', linestyle='--', linewidth=1.0, alpha=0.5, label='95%')
    ax1.set_ylabel('Percentage of Max (%)', fontsize=12)
    ax1.set_title(f'{model.upper()} - MA Value as % of Maximum', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 下图：逐层变化率
    pos_colors = ['#2ecc71' if c >= 0 else '#e74c3c' for c in changes]
    ax2.bar(layers, changes, color=pos_colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.0)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Change from Previous (%)', fontsize=12)
    ax2.set_title(f'{model.upper()} - Layer-to-Layer Change Rate', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    outfile_png = outdir / f'{model}_type3_percentage_change.png'
    outfile_pdf = outdir / f'{model}_type3_percentage_change.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated: {outfile_png.name}")


def plot_type4_heatmap_clean(data, outdir):
    """生成清洁热力图"""
    model = data['model']
    layers = data['layers']
    values = data['values']

    # 创建矩阵（1行N列）
    matrix = values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(14, 3), dpi=300)

    # 绘制热力图
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')

    # 设置刻度
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels(['MA Value'], fontsize=12)

    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_title(f'{model.upper()} - MLP Layer Ablation Heatmap', fontsize=14, fontweight='bold', pad=15)

    # 添加数值标注（每3个layer显示一个）
    for i in range(0, len(layers), 3):
        text = ax.text(i, 0, f'{values[i]:.0f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Activation Value', fontsize=11)

    plt.tight_layout()

    outfile_png = outdir / f'{model}_type4_heatmap_clean.png'
    outfile_pdf = outdir / f'{model}_type4_heatmap_clean.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Generated: {outfile_png.name}")


def process_model(model_name, base_results_dir, base_output_dir):
    """处理单个模型，生成所有图表"""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    # 创建模型专属输出目录
    outdir = Path(base_output_dir) / 'exp2_figures' / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # 加载数据
        data = load_summary(model_name, base_results_dir)

        # 生成5种图表
        plot_3d_contribution(data, outdir)
        plot_dual_bar_smooth(data, outdir)
        plot_line_smooth_v2(data, outdir)
        plot_type3_percentage_change(data, outdir)
        plot_type4_heatmap_clean(data, outdir)

        print(f"✅ {model_name}: All 5 figures generated successfully!")
        return True

    except Exception as e:
        print(f"❌ {model_name}: Error - {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Exp2 visualizations')
    parser.add_argument('--model', type=str, help='Generate for a single model (e.g., gpt2)')
    parser.add_argument('--all', action='store_true', help='Generate for all models')
    args = parser.parse_args()

    # 配置
    BASE_RESULTS_DIR = 'results/models'
    BASE_OUTPUT_DIR = 'results/plot_results'

    # 所有8个模型
    ALL_MODELS = [
        'gpt2',
        'gptj_6b',
        'bloom_7b1',
        'falcon_7b',
        'opt_7b',
        'mistral_7b_v03',
        'qwen2.5_7b',
        'llama2_13b',  # 新完成的模型
    ]

    # 确定要处理的模型列表
    if args.model:
        MODELS = [args.model]
        print(f"\n🎯 Mode: Single model ({args.model})")
    elif args.all:
        MODELS = ALL_MODELS
        print(f"\n🎯 Mode: All models")
    else:
        # 默认只生成gpt2作为样例
        MODELS = ['gpt2']
        print(f"\n🎯 Mode: Sample (gpt2 only)")
        print("💡 Use --all to generate all models, or --model <name> for specific model")

    setup_style()

    print("\n" + "="*60)
    print("🎨 Generating Exp2 Visualizations for All Models")
    print("="*60)
    print(f"Models to process: {len(MODELS)}")
    print(f"Figures per model: 5")
    print(f"Total figures: {len(MODELS) * 5}")
    print("="*60)

    success_count = 0
    for model in MODELS:
        if process_model(model, BASE_RESULTS_DIR, BASE_OUTPUT_DIR):
            success_count += 1

    print("\n" + "="*60)
    print(f"✅ Summary: {success_count}/{len(MODELS)} models processed successfully")
    print(f"📁 Output directory: {BASE_OUTPUT_DIR}/exp2_figures/")
    print("="*60)


if __name__ == '__main__':
    main()
