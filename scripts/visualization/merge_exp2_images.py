#!/usr/bin/env python3
"""
Directly merge existing Exp2 2D comparison images into one large figure
Use PIL to read existing PNG images and combine
Layout: 2 rows × 4 columns
Legend: Centered at bottom
"""

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# Configuration
FIGURES_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')
OUTPUT_DIR = FIGURES_DIR

# Model configuration(in specific order)
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
    """Use matplotlib to merge existing images"""

    # Create 2 rows × 4 columns subplots (7 models + 1 empty slot)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Load and display image for each model
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']

        # Find corresponding PNG file
        img_file = FIGURES_DIR / model_key / f"{model_key}_exp2_2d_comparison.png"

        if not img_file.exists():
            print(f"Warning: {img_file} not found")
            ax.text(0.5, 0.5, f'{model_display}\nImage Not Found',
                   ha='center', va='center', fontsize=16, color='red')
            ax.axis('off')
            continue

        # Read and display image
        img = mpimg.imread(str(img_file))
        ax.imshow(img)
        ax.axis('off')

        # Add model name as title
        ax.set_title(model_display, fontsize=18, fontweight='bold', pad=10)

    # Leave last subplot empty or add description
    ax = axes[7]
    ax.axis('off')
    ax.text(0.5, 0.5, 'Exp2: MLP Layer Ablation\n\nMassive Activation patterns\nacross model architectures',
           ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='black', linewidth=2))

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.08, hspace=0.15, wspace=0.05)

    # Add main title
    fig.suptitle('Exp2: MLP Layer-wise Ablation - 2D Heatmap Comparison Across Models',
                fontsize=22, fontweight='bold', y=0.98)

    # Add description text (centered at bottom)
    fig.text(0.5, 0.02,
            'Figure: Each subplot shows layer-wise ablation effects. '
            'Darker regions indicate layers with significant MA suppression when ablated.',
            ha='center', fontsize=13, style='italic',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    # Save
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
