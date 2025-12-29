#!/usr/bin/env python3
"""
Exp2: 将每个模型的PDF转换为PNG，然后组合成4×2的图组
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
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_7b', 'display': 'OPT-7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]

def pdf_to_png(pdf_path, output_path, dpi=300, crop=True):
    """将PDF转换为PNG，可选择裁剪掉图例"""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        
        # 高DPI渲染
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        if crop:
            # 裁剪掉图例部分，只保留核心图形
            rect = page.rect
            page_width = rect.width
            page_height = rect.height
            crop_rect = fitz.Rect(
                page_width * 0.02,    # 左边界
                page_height * 0.12,   # 上边界（去掉标题）
                page_width * 0.72,    # 右边界（去掉图例）
                page_height * 0.78    # 下边界（去掉底部图例）
            )
            pix = page.get_pixmap(matrix=mat, clip=crop_rect)
        else:
            pix = page.get_pixmap(matrix=mat)
        
        # 保存为PNG
        pix.save(str(output_path))
        doc.close()
        return True
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return False

def step1_convert_pdfs_to_pngs():
    """步骤1: 将每个模型的PDF转换为PNG（裁剪掉图例）"""
    print("Step 1: Converting PDFs to PNGs (cropped, no legend)...\n")
    
    png_files = []
    for model_config in MODEL_CONFIGS:
        model_key = model_config['key']
        model_display = model_config['display']
        
        pdf_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.pdf"
        png_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison_cropped.png"
        
        if not pdf_file.exists():
            print(f"⚠ {model_display}: PDF not found")
            png_files.append(None)
            continue
        
        if pdf_to_png(pdf_file, png_file, dpi=400, crop=True):
            print(f"✓ {model_display}: {png_file.name}")
            png_files.append(png_file)
        else:
            png_files.append(None)
    
    return png_files

def step2_combine_pngs(png_files):
    """步骤2: 组合PNG为4×2图组"""
    print("\nStep 2: Combining PNGs into 4×2 grid...\n")
    
    # 创建4列2行的子图，减小纵向距离
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    axes = axes.flatten()
    
    for idx, (png_file, model_config) in enumerate(zip(png_files, MODEL_CONFIGS)):
        ax = axes[idx]
        model_display = model_config['display']
        
        if png_file is None or not png_file.exists():
            ax.text(0.5, 0.5, f'{model_display}\nNot Available',
                   ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            continue
        
        # 读取PNG
        img = Image.open(png_file)
        ax.imshow(np.array(img))
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 底部添加模型名称，字体变大
        ax.set_xlabel(model_display, fontsize=14, fontweight='bold', labelpad=2)
        
        # 边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        print(f"✓ Added {model_display}")
    
    # 紧凑布局，减小纵向距离
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06,
                       hspace=0.06, wspace=0.03)
    
    # 保存
    output_png = OUTPUT_DIR / 'exp2_combined_from_pngs.png'
    output_pdf = OUTPUT_DIR / 'exp2_combined_from_pngs.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_png}")
    print(f"✅ Combined figure saved: {output_pdf}")
    
    plt.close()

if __name__ == '__main__':
    print("="*60)
    print("Exp2: PDF to PNG conversion and combination")
    print("="*60 + "\n")
    
    # 步骤1: 转换PDF为PNG
    png_files = step1_convert_pdfs_to_pngs()
    
    # 步骤2: 组合PNG
    step2_combine_pngs(png_files)
    
    print("\n✅ All done!")
