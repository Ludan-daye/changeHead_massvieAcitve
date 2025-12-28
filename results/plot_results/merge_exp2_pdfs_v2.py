#!/usr/bin/env python3
"""
合并Exp2的PDF图为一个大图 - V2版本
- 8个模型完整展示
- 去掉每个小图的图例和标题（裁剪PDF）
- 模型名称作为子图标题
布局：2行 × 4列
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path
import numpy as np

# 配置
FIGURES_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/exp2_figures')
OUTPUT_DIR = FIGURES_DIR

# 模型配置（8个模型完整列表）
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

def pdf_to_image_cropped(pdf_path, dpi=200, crop_top=0.15, crop_bottom=0.1):
    """
    将PDF的第一页转换为PIL Image并裁剪掉标题和图例部分

    crop_top: 裁剪顶部比例（去掉标题）
    crop_bottom: 裁剪底部比例（去掉图例）
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]  # 获取第一页

        # 设置缩放以获得更高分辨率
        zoom = dpi / 72  # 默认72 DPI
        mat = fitz.Matrix(zoom, zoom)

        # 渲染页面为图像
        pix = page.get_pixmap(matrix=mat)

        # 转换为PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        doc.close()

        # 裁剪图像（去掉标题和图例）
        width, height = img.size
        crop_y_top = int(height * crop_top)
        crop_y_bottom = int(height * (1 - crop_bottom))

        # 裁剪
        img_cropped = img.crop((0, crop_y_top, width, crop_y_bottom))

        return np.array(img_cropped)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def create_combined_figure():
    """合并PDF图为大图"""

    # 创建2行4列的子图
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # 为每个模型加载并显示PDF
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        # 查找对应的PDF文件
        pdf_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.pdf"

        if not pdf_file.exists():
            print(f"Warning: {pdf_file} not found")
            ax.text(0.5, 0.5, f'{model_display}\nData Not Available',
                   ha='center', va='center', fontsize=16, color='red',
                   bbox=dict(boxstyle='round,pad=1', facecolor='#ffe6e6', edgecolor='red'))
            ax.axis('off')
            continue

        # 读取PDF并转换为图像（裁剪掉标题和图例）
        img = pdf_to_image_cropped(pdf_file, dpi=200, crop_top=0.15, crop_bottom=0.12)

        if img is None:
            ax.text(0.5, 0.5, f'{model_display}\nFailed to Load',
                   ha='center', va='center', fontsize=16, color='orange')
            ax.axis('off')
            continue

        # 显示图片
        ax.imshow(img)
        ax.axis('off')

        # 添加模型名称作为标题（在子图外部）
        ax.set_title(model_display, fontsize=16, fontweight='bold', pad=10)

        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        print(f"✓ Loaded {model_display}")

    # 调整布局
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                       hspace=0.25, wspace=0.08)

    # 添加总标题
    fig.suptitle('Exp2: MLP Layer-wise Ablation Analysis - 2D Heatmap Comparison',
                fontsize=24, fontweight='bold', y=0.98)

    # 保存
    output_file_png = OUTPUT_DIR / 'exp2_combined_8models.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_8models.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Combining Exp2 PDF figures (8 models, cropped version)...\n")
    try:
        create_combined_figure()
        print("\n✅ Done!")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install PyMuPDF: pip install PyMuPDF")
