#!/usr/bin/env python3
"""
Merge Exp2 PDF figures into one large figure
Use PyMuPDF to read PDF and convert to images
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

# Model configuration (arranged in specific order)
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_7b', 'display': 'OPT-7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
]

def pdf_to_image(pdf_path, dpi=150):
    """Convert first page of PDF to PIL Image"""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]  # Get first page

        # Set zoom to get higher resolution
        zoom = dpi / 72  # Default 72 DPI
        mat = fitz.Matrix(zoom, zoom)

        # Render page as image
        pix = page.get_pixmap(matrix=mat)

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
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
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

        # Read PDF and convert to image
        img = pdf_to_image(pdf_file, dpi=150)

        if img is None:
            ax.text(0.5, 0.5, f'{model_display}\nFailed to Load',
                   ha='center', va='center', fontsize=16, color='orange')
            ax.axis('off')
            continue

        # Display image
        ax.imshow(img)
        ax.axis('off')

        # Add borders
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        print(f"✓ Loaded {model_display}")

    # Add description to last subplot
    ax = axes[7]
    ax.axis('off')

    # Add legend description
    legend_text = """
    Exp2 Analysis Summary

    • Each heatmap shows layer-wise
      MLP ablation effects

    • Color intensity indicates MA
      value changes when a layer
      is disabled

    • Critical layers show distinct
      patterns (typically one layer
      dominates MA generation)
    """

    ax.text(0.5, 0.5, legend_text,
           ha='center', va='center', fontsize=13,
           bbox=dict(boxstyle='round,pad=1.2', facecolor='#f0f8ff',
                    edgecolor='#4682b4', linewidth=2.5),
           family='monospace')

    # Adjust layout
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.08,
                       hspace=0.2, wspace=0.1)

    # Add main title
    fig.suptitle('Exp2: MLP Layer-wise Ablation Analysis - 2D Heatmap Comparison',
                fontsize=24, fontweight='bold', y=0.97)

    # Add bottom description
    caption = ('Figure: Layer-wise ablation heatmaps for 7 LLM architectures. '
              'Each subplot shows the Massive Activation (MA) values when individual MLP layers are ablated. '
              'Darker/cooler colors indicate larger MA suppression.')

    fig.text(0.5, 0.03, caption,
            ha='center', fontsize=12, style='italic', wrap=True,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                     alpha=0.9, edgecolor='gray'))

    # Save
    output_file_png = OUTPUT_DIR / 'exp2_combined_all_models_final.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_all_models_final.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Combining Exp2 PDF figures using PyMuPDF...\n")
    try:
        create_combined_figure()
        print("\n✅ Done!")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install PyMuPDF: pip install PyMuPDF")
