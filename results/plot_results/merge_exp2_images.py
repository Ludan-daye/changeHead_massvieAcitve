#!/usr/bin/env python3
"""
直接合并现有的Exp2 2D comparison图片为一个大图
使用PIL读取现有PNG图片并组合
布局：2行 × 4列
图例：底部中央统一放置
"""

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# 配置
FIGURES_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp2_figures')
OUTPUT_DIR = FIGURES_DIR

# 模型配置（按特定顺序排列）
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_7b', 'display': 'OPT-7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
]

def create_combined_figure_matplotlib():
    """使用matplotlib合并现有图片"""

    # 创建2行4列的子图（7个模型+1个空位）
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # 为每个模型加载并显示图片
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        # 查找对应的PNG文件
        img_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.png"

        if not img_file.exists():
            print(f"Warning: {img_file} not found")
            ax.text(0.5, 0.5, f'{model_display}\nImage Not Found',
                   ha='center', va='center', fontsize=16, color='red')
            ax.axis('off')
            continue

        # 读取并显示图片
        img = mpimg.imread(str(img_file))
        ax.imshow(img)
        ax.axis('off')

        # 添加模型名称作为标题
        ax.set_title(model_display, fontsize=18, fontweight='bold', pad=10)

    # 最后一个子图留空或添加说明
    ax = axes[7]
    ax.axis('off')
    ax.text(0.5, 0.5, 'Exp2: MLP Layer Ablation\n\nMassive Activation patterns\nacross model architectures',
           ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='black', linewidth=2))

    # 调整子图间距
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.08, hspace=0.15, wspace=0.05)

    # 添加总标题
    fig.suptitle('Exp2: MLP Layer-wise Ablation - 2D Heatmap Comparison Across Models',
                fontsize=22, fontweight='bold', y=0.98)

    # 添加说明文字（底部中央）
    fig.text(0.5, 0.02,
            'Figure: Each subplot shows layer-wise ablation effects. '
            'Darker regions indicate layers with significant MA suppression when ablated.',
            ha='center', fontsize=13, style='italic',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    # 保存
    output_file_png = OUTPUT_DIR / 'exp2_combined_all_models.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_all_models.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Combining existing Exp2 2D comparison figures...")
    create_combined_figure_matplotlib()
    print("Done!")
