#!/usr/bin/env python3
"""
RQ5 Detailed Comparison Figures
Beautiful before-after V-ablation comparison for each model
Highlighting changes and importance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Enhanced academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.5,
    'patch.linewidth': 1.2,
})

sns.set_style("whitegrid", {
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Unified color scheme
MODEL_COLORS = {
    'gptj_6b': '#e74c3c',
    'bloom_7b1': '#3498db',
    'qwen2.5_7b': '#2ecc71',
    'falcon_7b': '#9b59b6',
    'mistral_7b_v03': '#f39c12',
    'gpt2': '#e67e22',
    'opt_6.7b': '#1abc9c'
}

SEMANTIC_COLORS = {
    'baseline': '#27ae60',
    'ablated': '#95a5a6',
    'change_positive': '#3498db',
    'change_negative': '#e74c3c',
    'strong': '#c0392b',
    'medium': '#e67e22',
    'weak': '#3498db'
}

MODEL_NAMES = {
    'gptj_6b': 'GPT-J',
    'bloom_7b1': 'BLOOM',
    'qwen2.5_7b': 'Qwen',
    'falcon_7b': 'Falcon',
    'mistral_7b_v03': 'Mistral',
    'gpt2': 'GPT-2',
    'opt_6.7b': 'OPT'
}

MODELS_7 = ['gptj_6b', 'bloom_7b1', 'qwen2.5_7b', 'falcon_7b', 'mistral_7b_v03', 'gpt2', 'opt_6.7b']

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'
VIS_DIR = BASE_DIR / 'visualizations'

DPI = 300
FIGURE_SIZE = (10, 6)


def load_model_rq5_data(model):
    """Load RQ5 data for a specific model"""
    json_path = RESULTS_DIR / 'models' / model / 'exp6' / 'v_ablation_simple.json'
    if not json_path.exists():
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    result = {}
    
    # Special handling for BLOOM - use Layer 0 (MA source layer)
    if model == 'bloom_7b1' and 'layer_comparison' in data:
        layer_0_data = data['layer_comparison']['layer_0']
        result['baseline_ma'] = layer_0_data['baseline_ma']
        result['ablated_ma'] = layer_0_data['ablated_ma']
        result['change_pct'] = layer_0_data['change_percent']
        # Note: Layer 0 doesn't have sample values in this format
        result['use_layer_0'] = True
    # Parse different JSON formats
    elif 'change_percentage' in data:
        result['baseline_ma'] = data.get('baseline_ma', 0)
        result['ablated_ma'] = data.get('ablated_ma', 0)
        result['change_pct'] = data['change_percentage']
    elif 'v_ablated' in data:
        result['baseline_ma'] = data['baseline'].get('ma_avg', 0)
        result['ablated_ma'] = data['v_ablated'].get('ma_avg', 0)
        result['change_pct'] = data['v_ablated'].get('change_percent', 0)
        
        # Get sample values if available
        if 'ma_values' in data['baseline']:
            result['baseline_samples'] = data['baseline']['ma_values']
        if 'ma_values' in data['v_ablated']:
            result['ablated_samples'] = data['v_ablated']['ma_values']
    else:
        return None
    
    result['change_abs'] = result['ablated_ma'] - result['baseline_ma']
    
    return result


def classify_dependency(change_pct):
    """Classify dependency strength"""
    abs_change = abs(change_pct)
    if abs_change > 80:
        return 'Strong', SEMANTIC_COLORS['strong']
    elif abs_change > 50:
        return 'Medium', SEMANTIC_COLORS['medium']
    else:
        return 'Weak', SEMANTIC_COLORS['weak']


def plot_model_comparison(model, data, output_dir):
    """Generate detailed comparison figure for one model"""
    print(f"Generating figure for {MODEL_NAMES[model]}...")
    
    model_name = MODEL_NAMES[model]
    model_color = MODEL_COLORS[model]
    
    baseline_ma = data['baseline_ma']
    ablated_ma = data['ablated_ma']
    change_pct = data['change_pct']
    change_abs = data['change_abs']
    
    dep_level, dep_color = classify_dependency(change_pct)
    
    # Create figure with simplified layout (no right panel)
    fig = plt.figure(figsize=(10, 7), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    
    # ============ Panel 1: Main Comparison Bar Chart ============
    ax1 = fig.add_subplot(gs[0])
    
    conditions = ['Baseline', 'V-Ablated']
    values = [baseline_ma, ablated_ma]
    colors = [SEMANTIC_COLORS['baseline'], SEMANTIC_COLORS['ablated']]
    
    bars = ax1.bar(conditions, values, color=colors, alpha=0.9, 
                   edgecolor='white', linewidth=3, width=0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='gray', alpha=0.9, linewidth=1.5))
    
    # Add connecting arrow showing change
    arrow_y = max(values) * 0.45
    arrow = FancyArrowPatch((0.3, arrow_y), (0.7, arrow_y),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=3, color=dep_color, alpha=0.7,
                           zorder=5)
    ax1.add_patch(arrow)
    
    # Change annotation on arrow
    ax1.text(0.5, arrow_y * 1.2, f'{change_pct:+.1f}%',
            ha='center', va='bottom', fontsize=17, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=dep_color, 
                     edgecolor='white', alpha=0.95, linewidth=3))
    
    # Add dependency level text
    dep_text = f'{dep_level} Dependency'
    ax1.text(0.5, arrow_y * 0.85, dep_text,
            ha='center', va='top', fontsize=13, fontweight='bold',
            color=dep_color, style='italic')
    
    ax1.set_ylabel('MA Value', fontweight='bold', fontsize=15)
    ax1.set_ylim(0, max(values) * 1.3)
    ax1.tick_params(axis='x', labelsize=14)
    
    # Add model name as title-like text (top center, avoid overlap)
    ax1.text(0.5, 1.08, model_name, transform=ax1.transAxes,
            fontsize=22, fontweight='bold', va='bottom', ha='center',
            color=model_color,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                     edgecolor=model_color, alpha=0.95, linewidth=3))
    
    # ============ Panel 2: Sample Distribution (if available) ============
    ax2 = fig.add_subplot(gs[1])
    
    if 'baseline_samples' in data and 'ablated_samples' in data:
        baseline_samples = data['baseline_samples']
        ablated_samples = data['ablated_samples']
        
        x = np.arange(len(baseline_samples))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, baseline_samples, width, 
                       label='Baseline', color=SEMANTIC_COLORS['baseline'],
                       alpha=0.85, edgecolor='white', linewidth=2)
        bars2 = ax2.bar(x + width/2, ablated_samples, width,
                       label='V-Ablated', color=SEMANTIC_COLORS['ablated'],
                       alpha=0.85, edgecolor='white', linewidth=2)
        
        ax2.set_xlabel('Sample Index', fontweight='bold', fontsize=14)
        ax2.set_ylabel('MA Value', fontweight='bold', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{i+1}' for i in x])
        ax2.legend(loc='upper right', frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.95, edgecolor='gray', fontsize=12)
        
        # Add mean lines
        ax2.axhline(baseline_ma, color=SEMANTIC_COLORS['baseline'], 
                   linestyle='--', linewidth=2, alpha=0.7, label='_nolegend_')
        ax2.axhline(ablated_ma, color=SEMANTIC_COLORS['ablated'], 
                   linestyle='--', linewidth=2, alpha=0.7, label='_nolegend_')
        
    else:
        # Show bar chart with absolute change
        conditions = ['Baseline', 'V-Ablated', 'Absolute\nChange']
        values_display = [baseline_ma, ablated_ma, abs(change_abs)]
        colors_display = [SEMANTIC_COLORS['baseline'], 
                         SEMANTIC_COLORS['ablated'],
                         dep_color]
        
        bars = ax2.bar(conditions, values_display, 
                      color=colors_display, alpha=0.85,
                      edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, values_display):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax2.set_ylabel('MA Value / Change', fontweight='bold', fontsize=14)
    
    # No suptitle - model name already shown in figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for model name at top
    output_path = output_dir / f'rq5_{model}_comparison.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return {
        'model': model_name,
        'baseline': baseline_ma,
        'ablated': ablated_ma,
        'change_pct': change_pct,
        'dependency': dep_level
    }


def create_summary_figure(all_results, output_dir):
    """Create a summary comparison figure for all models"""
    print("\nGenerating summary comparison figure...")
    
    models = [r['model'] for r in all_results]
    baselines = [r['baseline'] for r in all_results]
    ablateds = [r['ablated'] for r in all_results]
    changes = [r['change_pct'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
    
    # Panel 1: Baseline vs Ablated
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baselines, width, label='Baseline',
                   color=SEMANTIC_COLORS['baseline'], alpha=0.85,
                   edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x + width/2, ablateds, width, label='V-Ablated',
                   color=SEMANTIC_COLORS['ablated'], alpha=0.85,
                   edgecolor='white', linewidth=2)
    
    ax1.set_xlabel('Model', fontweight='bold', fontsize=14)
    ax1.set_ylabel('MA Value', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95, edgecolor='gray', fontsize=12)
    ax1.text(0.5, 0.98, '(a) Baseline vs V-Ablated Comparison',
            transform=ax1.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold')
    
    # Panel 2: Change percentage
    model_keys = list(MODEL_COLORS.keys())[:len(models)]
    colors = [MODEL_COLORS[k] for k in model_keys]
    
    bars = ax2.barh(models, changes, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=2)
    
    ax2.set_xlabel('MA Change (%)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Model', fontweight='bold', fontsize=14)
    ax2.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2)
    ax2.axvline(x=-50, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)
    ax2.axvline(x=-80, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)
    
    # Add value labels
    for bar, val in zip(bars, changes):
        width_val = bar.get_width()
        x_pos = width_val - abs(max(changes)) * 0.05 if width_val < 0 else width_val + abs(max(changes)) * 0.05
        ha = 'right' if width_val < 0 else 'left'
        color = 'white' if abs(width_val) > 60 else 'black'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%',
                ha=ha, va='center', fontsize=11, fontweight='bold', color=color)
    
    ax2.text(0.5, 0.98, '(b) V-Dependency Strength',
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold')
    
    plt.suptitle('RQ5: V-Matrix Ablation - All Models Summary',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = output_dir / 'rq5_all_models_summary.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("RQ5 Detailed Comparison Figures Generation")
    print("Highlighting V-Matrix Ablation Effects for Each Model")
    print("="*70 + "\n")
    
    output_dir = VIS_DIR / 'rq5'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Generate individual model figures
    for model in MODELS_7:
        data = load_model_rq5_data(model)
        if data is None:
            print(f"⚠ Skipping {MODEL_NAMES[model]}: No data available")
            continue
        
        result = plot_model_comparison(model, data, output_dir)
        all_results.append(result)
    
    # Generate summary figure
    if all_results:
        create_summary_figure(all_results, output_dir)
    
    print("\n" + "="*70)
    print(f"✓ RQ5 Comparison Figures Complete! ({len(all_results)} models)")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # Print summary table
    print("Summary Table:")
    print(f"{'Model':<12} {'Baseline':<10} {'Ablated':<10} {'Change %':<12} {'Dependency'}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<12} {r['baseline']:<10.1f} {r['ablated']:<10.1f} "
              f"{r['change_pct']:+11.1f}% {r['dependency']}")


if __name__ == '__main__':
    main()
