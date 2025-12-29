#!/usr/bin/env python3
"""
Exp4: 组合8个模型的Cumulative_Energy_Distribution图为4×2图组
要求：去掉图例、大字体、稀疏坐标、小纵向距离、高DPI
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

# 配置
FIGURES_DIR = Path('PROJECT_ROOT/results/plot_results/exp4_figures')
OUTPUT_DIR = FIGURES_DIR

# 8个模型配置（注意exp4中opt是6.7b）
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]


def crop_image(img_array, crop_legend=True):
    """裁剪图片，去掉图例但保留坐标轴"""
    h, w = img_array.shape[:2]
    
    if crop_legend:
        # 只裁剪掉右侧图例区域，保留坐标轴
        left = int(w * 0.0)
        right = int(w * 0.82)  # 去掉右侧图例
        top = int(h * 0.02)
        bottom = int(h * 0.98)
        return img_array[top:bottom, left:right]
    return img_array


def main():
    print("="*60)
    print("Exp4: Combine Cumulative Energy Distribution (4×2)")
    print("="*60 + "\n")

    # 创建4列2行的子图
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    axes = axes.flatten()

    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        # 查找PNG文件
        png_file = FIGURES_DIR / model_key / f"{model_key}_Cumulative_Energy_Distribution.png"

        if not png_file.exists():
            ax.text(0.5, 0.5, f'{model_display}\nNot Found',
                   ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            print(f"⚠ {model_display}: {png_file.name} not found")
            continue

        # 读取并裁剪图片
        img = Image.open(png_file)
        img_array = np.array(img)
        img_cropped = crop_image(img_array, crop_legend=True)

        # 显示图片
        ax.imshow(img_cropped)
        ax.set_xticks([])
        ax.set_yticks([])

        # 底部添加模型名称
        ax.set_xlabel(model_display, fontsize=13, fontweight='bold', labelpad=2)

        # 边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        print(f"✓ {model_display}")

    # 紧凑布局
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.08,
                       hspace=0.12, wspace=0.05)

    # 保存（高DPI）
    output_png = OUTPUT_DIR / 'exp4_combined_energy.png'
    output_pdf = OUTPUT_DIR / 'exp4_combined_energy.pdf'

    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_png}")
    print(f"✅ Combined figure saved: {output_pdf}")

    plt.close()
    print("\n✅ All done!")


if __name__ == '__main__':
    main()
