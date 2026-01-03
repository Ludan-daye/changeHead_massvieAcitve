#!/usr/bin/env python3
"""
Merge Exp2 PDF figures into one large figure - 8 models complete version
- Remove model name titles
- Increase DPI to make axis numbers clearer
- Complete display of 8 models
"""

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path
import numpy as np

# Configuration
FIGURES_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')
OUTPUT_DIR = FIGURES_DIR

# 8 model configuration
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
    Convert PDF to high-resolution image and crop title and legend
    Only keep core graphics and axes
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]

        # Get page dimensions
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # Crop region: remove title and legend, keep only central core graphics
        crop_rect = fitz.Rect(
            page_width * 0.02,          # Left border
            page_height * 0.15,         # Top border（Remove title）
            page_width * 0.70,          # Right border（Remove legend）
            page_height * 0.75          # Bottom border（Remove legend）
        )

        # High DPI rendering to get clear axis numbers
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # Render crop region
        pix = page.get_pixmap(matrix=mat, clip=crop_rect)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        doc.close()
        return np.array(img)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def create_combined_figure():
    """Create 8-model combined figure"""

    # Create 2 rows × 4 columns subplots
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    axes = axes.flatten()

    # Load and display PDF for each model
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']

        # Find PDF file
        pdf_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.pdf"

        if not pdf_file.exists():
            print(f"Warning: {pdf_file} not found")
            ax.text(0.5, 0.5, f'Data Not Available',
                   ha='center', va='center', fontsize=18, color='red')
            ax.axis('off')
            continue

        # Read and crop PDF
        img = pdf_to_image_cropped(pdf_file, dpi=500)

        if img is None:
            ax.text(0.5, 0.5, f'Failed to Load',
                   ha='center', va='center', fontsize=18, color='orange')
            ax.axis('off')
            continue

        # Display image
        ax.imshow(img)
        ax.axis('off')

        # Do not add title (remove model name as requested)

        # Add borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)

        print(f"✓ Loaded {model_key}")

    # Adjust layout
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02,
                       hspace=0.12, wspace=0.08)

    # Save
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
