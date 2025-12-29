#!/usr/bin/env python3
"""
合并Exp2的PDF图为一个大图 - 8个模型完整版
- 去掉模型名称标题
- 提高DPI让坐标轴数字更清晰
- 8个模型完整展示
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path
import numpy as np

# 配置
FIGURES_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')
OUTPUT_DIR = FIGURES_DIR

# 8个模型配置
MODEL_CONFIGS = [
    {'key': 'gpt2'},
    {'key': 'gptj_6b'},
    {'key': 'bloom_7b1'},
    {'key': 'falcon_7b'},
    {'key': 'opt_7b'},
    {'key': 'mistral_7b_v03'},
    {'key': 'qwen2.5_7b'},
    {'key': 'llama2_13b'},
]

def pdf_to_image_cropped(pdf_path, dpi=500):
    """
    将PDF转换为高清图像并裁剪掉标题和图例
    只保留核心图形和坐标轴
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]

        # 获取页面尺寸
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # 裁剪区域：去掉标题和图例，只保留中间核心图形
        crop_rect = fitz.Rect(
            page_width * 0.02,          # 左边界
            page_height * 0.15,         # 上边界（去掉标题）
            page_width * 0.70,          # 右边界（去掉图例）
            page_height * 0.75          # 下边界（去掉图例）
        )

        # 高DPI渲染以获得清晰的坐标轴数字
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # 渲染裁剪区域
        pix = page.get_pixmap(matrix=mat, clip=crop_rect)

        # 转换为PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        doc.close()
        return np.array(img)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def create_combined_figure():
    """创建8模型合并图"""

    # 创建2行4列的子图
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    axes = axes.flatten()

    # 为每个模型加载并显示PDF
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']

        # 查找PDF文件
        pdf_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.pdf"

        if not pdf_file.exists():
            print(f"Warning: {pdf_file} not found")
            ax.text(0.5, 0.5, f'Data Not Available',
                   ha='center', va='center', fontsize=18, color='red')
            ax.axis('off')
            continue

        # 读取并裁剪PDF
        img = pdf_to_image_cropped(pdf_file, dpi=500)

        if img is None:
            ax.text(0.5, 0.5, f'Failed to Load',
                   ha='center', va='center', fontsize=18, color='orange')
            ax.axis('off')
            continue

        # 显示图片
        ax.imshow(img)
        ax.axis('off')

        # 不添加标题（按要求去掉模型名称）

        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)

        print(f"✓ Loaded {model_key}")

    # 调整布局
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02,
                       hspace=0.12, wspace=0.08)

    # 保存
    output_file_png = OUTPUT_DIR / 'exp2_combined_all_models_final.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_all_models_final.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Creating combined Exp2 figure (8 models, no titles, high DPI)...\n")
    try:
        create_combined_figure()
        print("\n✅ Done!")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install PyMuPDF: pip install PyMuPDF")
