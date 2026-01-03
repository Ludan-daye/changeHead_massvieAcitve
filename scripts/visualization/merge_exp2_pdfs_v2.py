#!/usr/bin/env python3
"""
Merge Exp2 PDF figures into one large figure - V2 version
- Complete display of 8 models
- Remove legend and title from each subplot (crop PDF)
- Model names as subplot titles
Layout: 2 rows × 4 columns
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

# Model configuration(8 models complete list)
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
    Convert first page of PDF to PIL Image and precisely crop title and legend sections
    Use PyMuPDF coordinate system for precise cropping
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]  # Get first page

        # Get original page dimensions
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # According to analysis, legend positions differ for different models：
        # GPT-2: Bottom right(Y=79%, X=73%)
        # GPT-J: Top right(Y=3%, X=73%)
        # BLOOM: Top left(Y=3%, X=12%)
        # Solution: Keep only central core region (remove all edges)
        crop_rect = fitz.Rect(
            page_width * 0.02,          # Left border
            page_height * 0.15,         # Top border
            page_width * 0.70,          # Right border
            page_height * 0.68          # Bottom border (crop more bottom white space)
        )

        # Set higher zoom to get clearer axes
        zoom = dpi / 72  # Default 72 DPI
        mat = fitz.Matrix(zoom, zoom)

        # Render cropped region
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
    """Merge PDF figures into large figure"""

    # Create 2 rows × 4 columns subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    # Load and display PDF for each model
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        # Find corresponding PDF file
        pdf_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.pdf"

        if not pdf_file.exists():
            print(f"Warning: {pdf_file} not found")
            ax.text(0.5, 0.5, f'{model_display}\nData Not Available',
                   ha='center', va='center', fontsize=16, color='red',
                   bbox=dict(boxstyle='round,pad=1', facecolor='#ffe6e6', edgecolor='red'))
            ax.axis('off')
            continue

        # Read PDF and convert to image(crop title and legend)
        img = pdf_to_image_cropped(pdf_file, dpi=500)

        if img is None:
            ax.text(0.5, 0.5, f'{model_display}\nFailed to Load',
                   ha='center', va='center', fontsize=16, color='orange')
            ax.axis('off')
            continue

        # Display image
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add model name at bottom
        ax.set_xlabel(model_display, fontsize=11, fontweight='bold', labelpad=2)

        # Add borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        print(f"✓ Loaded {model_display}")

    # Adjust layout(compact arrangement, reduce white space)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                       hspace=0.12, wspace=0.03)

    # Save
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
