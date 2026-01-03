#!/usr/bin/env python3
"""
Merge Exp2 8-model 2D comparison plots into one large figure
Layout: 2 rows × 4 columns
Legend: Centered at bottom
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Configuration
RESULTS_DIR = Path('PROJECT_ROOT/results/experiments/exp2')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')

# Model configuration (in specific order, including only 7 models with data)
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2', 'critical_layer': 3},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B', 'critical_layer': 22},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1', 'critical_layer': 28},
    {'key': 'falcon_7b', 'display': 'Falcon-7B', 'critical_layer': 3},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B', 'critical_layer': 3},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B', 'critical_layer': 31},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B', 'critical_layer': 3},
]

# Color configuration (reference from example figure)
COLOR_BASELINE = '#666666'  # Dark gray
COLOR_ABLATED = '#D97E73'   # Orange-red
COLOR_CRITICAL = '#7BA5D5'  # Blue

def load_exp2_data(model_key):
    """Load Exp2 data"""
    summary_file = RESULTS_DIR / model_key / 'summary.json'

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data

def create_combined_figure():
    """Create combined large figure"""

    # Create 2 rows × 4 columns subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.98, top=0.94, bottom=0.12)

    axes = []
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    # Draw subplot for each model
    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']
        critical_layer = model_config['critical_layer']

        # Load data
        data = load_exp2_data(model_key)
        if data is None:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Available',
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Extract ablation data
        ablation = data.get('ablation', {})
        if not ablation:
            ax.text(0.5, 0.5, f'{model_display}\nNo Data',
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Sort layers
        layers = sorted([int(k) for k in ablation.keys()])
        ma_values = [ablation[str(layer)] for layer in layers]

        # Calculate baseline (use maximum value as baseline reference)
        baseline = max(ma_values)

        # Draw heatmap-style 2D comparison
        # Display layers in groups
        n_layers = len(layers)

        # Create 2D grid
        grid_rows = int(np.ceil(np.sqrt(n_layers)))
        grid_cols = int(np.ceil(n_layers / grid_rows))

        # Draw square for each layer
        for i, (layer, ma) in enumerate(zip(layers, ma_values)):
            row = i // grid_cols
            col = i % grid_cols

            # Normalize color (relative to baseline)
            if layer == critical_layer:
                color = COLOR_CRITICAL
                alpha = 1.0
                edgecolor = 'black'
                linewidth = 3
            else:
                ratio = ma / baseline if baseline > 0 else 0
                if ratio > 0.8:
                    color = COLOR_BASELINE
                    alpha = 0.9
                else:
                    color = COLOR_ABLATED
                    alpha = 0.7
                edgecolor = 'gray'
                linewidth = 1

            rect = Rectangle((col, grid_rows - 1 - row), 1, 1,
                           facecolor=color, alpha=alpha,
                           edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)

            # Add layer number text
            if n_layers <= 32:  # Only display when number of layers is not too large
                ax.text(col + 0.5, grid_rows - 1 - row + 0.5, str(layer),
                       ha='center', va='center', fontsize=8, color='white',
                       fontweight='bold' if layer == critical_layer else 'normal')

        # Set axes
        ax.set_xlim(0, grid_cols)
        ax.set_ylim(0, grid_rows)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        ax.set_title(f'{model_display}\n(Critical Layer: {critical_layer})',
                    fontsize=14, fontweight='bold', pad=10)

        # Add borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # Add unified legend (centered at bottom)
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_BASELINE, alpha=0.9, edgecolor='gray',
                      label='Non-critical Layers (MA > 80% baseline)'),
        mpatches.Patch(facecolor=COLOR_ABLATED, alpha=0.7, edgecolor='gray',
                      label='Suppressed Layers (MA < 80% baseline)'),
        mpatches.Patch(facecolor=COLOR_CRITICAL, alpha=1.0, edgecolor='black', linewidth=3,
                      label='Critical Layer (MA source)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=13, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, 0.02))

    # Add main title
    fig.suptitle('Exp2: MLP Layer-wise Ablation Analysis - 2D Comparison Across 8 Models',
                fontsize=18, fontweight='bold', y=0.98)

    # Save
    output_file_png = OUTPUT_DIR / 'exp2_combined_2d_comparison.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_2d_comparison.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Creating combined Exp2 2D comparison figure...")
    create_combined_figure()
    print("Done!")
