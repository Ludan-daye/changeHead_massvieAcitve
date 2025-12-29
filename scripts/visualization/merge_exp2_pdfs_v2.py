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
FIGURES_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')
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

def pdf_to_image_cropped(pdf_path, dpi=400):
    """
    将PDF的第一页转换为PIL Image并精确裁剪掉标题和图例部分
    使用PyMuPDF的坐标系统进行精确裁剪
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]  # 获取第一页

        # 获取原始页面尺寸
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # 根据分析，不同模型图例位置不同：
        # GPT-2: 底部右侧(Y=79%, X=73%)
        # GPT-J: 顶部右侧(Y=3%, X=73%)
        # BLOOM: 顶部左侧(Y=3%, X=12%)
        # 解决方案：只保留中间核心区域（去掉所有边缘）
        crop_rect = fitz.Rect(
            page_width * 0.02,          # 左边界
            page_height * 0.15,         # 上边界
            page_width * 0.70,          # 右边界
            page_height * 0.68          # 下边界（裁掉更多底部白边）
        )

        # 设置更高缩放以获得更清晰的坐标轴
        zoom = dpi / 72  # 默认72 DPI
        mat = fitz.Matrix(zoom, zoom)

        # 渲染裁剪后的区域
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
    """合并PDF图为大图"""

    # 创建2行4列的子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
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
        img = pdf_to_image_cropped(pdf_file, dpi=500)

        if img is None:
            ax.text(0.5, 0.5, f'{model_display}\nFailed to Load',
                   ha='center', va='center', fontsize=16, color='orange')
            ax.axis('off')
            continue

        # 显示图片
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # 在底部添加模型名称
        ax.set_xlabel(model_display, fontsize=11, fontweight='bold', labelpad=2)

        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        print(f"✓ Loaded {model_display}")

    # 调整布局（紧凑排列，减少留白）
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                       hspace=0.12, wspace=0.03)

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
