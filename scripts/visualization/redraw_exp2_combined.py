#!/usr/bin/env python3
"""
Redraw Exp2 combined figure from data
- No main title
- Large font for x-axis
- Sparse ticks
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = Path('PROJECT_ROOT/results/experiments/exp2')
OUTPUT_DIR = Path('PROJECT_ROOT/results/plot_results/exp2_figures')

# Model configuration
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2', 'critical_layer': 3},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B', 'critical_layer': 22},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1', 'critical_layer': 28},
    {'key': 'falcon_7b', 'display': 'Falcon-7B', 'critical_layer': 3},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B', 'critical_layer': 3},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B', 'critical_layer': 31},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B', 'critical_layer': 3},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B', 'critical_layer': 30},
]

# Color configuration - using 2d_comparison color scheme
COLOR_BASELINE = '#2ECC71'    # Green - Baseline (All MLP Active)
COLOR_ABLATED = '#E74C3C'     # Red - Layer Disabled
COLOR_FILL = '#FFB6B0'        # Light red - Fill area

def load_exp2_data(model_key):
    """Load Exp2 data"""
    summary_file = RESULTS_DIR / model_key / 'summary.json'

    # Special case: LLaMA2 data is in another location
    if not summary_file.exists() and 'llama' in model_key:
        alt_path = Path('PROJECT_ROOT/results/models/llama2_13b/exp2b_mlp_layer_ablation/summary.json')
        if alt_path.exists():
            summary_file = alt_path

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data

def create_combined_figure():
    """Create combined figure"""

    # Create 2 rows × 4 columns subplots - increase size
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))
    axes = axes.flatten()

    for idx, model_config in enumerate(MODEL_CONFIGS):
        ax = axes[idx]
        model_key = model_config['key']
        model_display = model_config['display']
        critical_layer = model_config['critical_layer']

        # Load data
        data = load_exp2_data(model_key)
        if data is None:
            ax.text(0.5, 0.5, f'{model_display}\nData Not Available',
                   ha='center', va='center', fontsize=24, color='red',
                   fontweight='bold')
            ax.axis('off')
            continue

        # Extract ablation data
        ablation = data.get('ablation', {})
        if not ablation:
            ax.text(0.5, 0.5, f'{model_display}\nNo Data',
                   ha='center', va='center', fontsize=24, color='red',
                   fontweight='bold')
            ax.axis('off')
            continue

        # Sort layers and get MA values
        layers = sorted([int(k) for k in ablation.keys()])
        ma_ablated = [ablation[str(layer)] for layer in layers]

        # Calculate baseline (assume maximum MA value)
        baseline = max(ma_ablated)
        ma_baseline = [baseline] * len(layers)

        # Draw figure - bold lines, add scatter markers (mimic 2d_comparison style)
        ax.plot(layers, ma_baseline, color=COLOR_BASELINE, linewidth=4,
               linestyle='-', marker='o', markersize=8, markerfacecolor=COLOR_BASELINE,
               markeredgecolor=COLOR_BASELINE, markeredgewidth=2,
               label='Baseline', zorder=3)
        ax.plot(layers, ma_ablated, color=COLOR_ABLATED, linewidth=4,
               linestyle='-', marker='o', markersize=8, markerfacecolor=COLOR_ABLATED,
               markeredgecolor=COLOR_ABLATED, markeredgewidth=2,
               label='Ablated', zorder=4)

        # Fill area
        ax.fill_between(layers, ma_baseline, ma_ablated,
                        color=COLOR_FILL, alpha=0.4, zorder=1)

        # Mark critical layer
        if critical_layer < len(layers):
            ax.axvline(x=critical_layer, color='darkgray', linestyle=':',
                      linewidth=3, alpha=0.8, zorder=2)

        # Set axes labels - extra large font
        ax.set_xlabel('Layer Index', fontsize=26, fontweight='bold', labelpad=10)
        ax.set_ylabel('MA Value', fontsize=26, fontweight='bold', labelpad=10)

        # X-axis ticks - more sparse, larger font
        n_layers = len(layers)
        if n_layers <= 12:
            tick_step = 3
        elif n_layers <= 30:
            tick_step = 6
        else:
            tick_step = 10

        tick_positions = list(range(0, n_layers, tick_step))
        if (n_layers - 1) not in tick_positions:
            tick_positions.append(n_layers - 1)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=24, fontweight='bold')

        # Y-axis font - larger and bold
        ax.tick_params(axis='y', labelsize=22, width=2, length=8)
        ax.tick_params(axis='x', width=2, length=8)

        # Do not display title (remove model name as requested)
        # ax.set_title(model_display, fontsize=28, fontweight='bold', pad=20)

        # Strengthen grid
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.2, color='gray')

        # Bold borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

        print(f"✓ Plotted {model_display}")

    # Adjust layout - Remove title then reduce top margin
    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08,
                       hspace=0.25, wspace=0.3)

    # Save as exp2_combined_2d_heatmap (2d comparison style)
    output_file_png = OUTPUT_DIR / 'exp2_combined_2d_heatmap.png'
    output_file_pdf = OUTPUT_DIR / 'exp2_combined_2d_heatmap.pdf'

    plt.savefig(output_file_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined figure saved: {output_file_png}")
    print(f"✅ Combined figure saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Redrawing Exp2 combined figure from data...\n")
    create_combined_figure()
    print("\n✅ Done!")
