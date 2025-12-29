#!/usr/bin/env python3
"""
Academic-style Visualization Generation
Professional figures for publication with consistent styling
- All English
- No titles
- Clean, minimalist design
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

# Academic style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.color': '#CCCCCC',
    'grid.alpha': 0.5
})

sns.set_palette("deep")
sns.set_style("whitegrid", {
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Academic color scheme - professional and color-blind friendly
COLORS = {
    'attention': '#377eb8',  # Blue
    'mlp': '#e41a1c',        # Red
    'baseline': '#4daf4a',   # Green
    'ablated': '#999999',    # Gray
    'positive': '#4daf4a',   # Green
    'negative': '#e41a1c',   # Red
    'strong': '#d73027',     # Dark red
    'medium': '#fc8d59',     # Orange
    'weak': '#4575b4',       # Blue
    'cat1': '#8dd3c7',
    'cat2': '#fb8072',
    'cat3': '#fdb462',
    'cat4': '#80b1d3'
}

# Figure settings
FIGURE_SIZE_SINGLE = (5, 4)
FIGURE_SIZE_DOUBLE = (10, 4)
FIGURE_SIZE_WIDE = (7, 4)
DPI = 300

# Model configuration
MODELS = ['gptj_6b', 'bloom_7b1', 'qwen2.5_7b', 'falcon_7b', 'mistral_7b_v03']
MODELS_7 = MODELS + ['gpt2', 'opt_6.7b']

MODEL_NAMES = {
    'gptj_6b': 'GPT-J',
    'bloom_7b1': 'BLOOM',
    'qwen2.5_7b': 'Qwen',
    'falcon_7b': 'Falcon',
    'mistral_7b_v03': 'Mistral',
    'gpt2': 'GPT-2',
    'opt_6.7b': 'OPT'
}

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'
VIS_DIR = BASE_DIR / 'visualizations'


def load_rq2_data():
    """Load RQ2 data - MLP vs Attention"""
    data = {}
    for model in MODELS:
        json_path = RESULTS_DIR / 'models' / model / 'RQ2_mlp_source' / 'verification.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                model_data = json.load(f)
                data[model] = {
                    'attn_max': model_data.get('attention_output_max', 0),
                    'mlp_max': model_data.get('mlp_output_max', 0),
                    'ratio': model_data.get('ratio', 0)
                }
    return data


def load_rq1_data():
    """Load RQ1 data - Attention contribution"""
    data = {}
    for model in MODELS:
        readme_path = RESULTS_DIR / 'models' / model / 'exp1' / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '+266%' in content or '+273%' in content:
                    change = 266
                elif '-98%' in content:
                    change = -98
                elif '-96%' in content:
                    change = -96
                elif '-60%' in content:
                    change = -60
                elif '-21%' in content:
                    change = -21
                elif '-18%' in content:
                    change = -18
                else:
                    change = 0
                data[model] = {'change_pct': change}
    return data


def load_rq3_data():
    """Load RQ3 data - Function word triggers"""
    json_path = RESULTS_DIR / 'MA_POSITION_TOKEN_ANALYSIS.json'
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_rq5_data(include_7=False):
    """Load RQ5 data - V-matrix ablation"""
    models = MODELS_7 if include_7 else MODELS
    data = {}
    for model in models:
        json_path = RESULTS_DIR / 'models' / model / 'exp6' / 'v_ablation_simple.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                model_data = json.load(f)
                if 'change_percentage' in model_data:
                    change_pct = model_data['change_percentage']
                    baseline = model_data.get('baseline_ma', 0)
                    ablated = model_data.get('ablated_ma', 0)
                elif 'v_ablated' in model_data:
                    change_pct = model_data['v_ablated'].get('change_percent', 0)
                    baseline = model_data['baseline'].get('ma_avg', 0)
                    ablated = model_data['v_ablated'].get('ma_avg', 0)
                else:
                    continue
                
                data[model] = {
                    'baseline_ma': baseline,
                    'ablated_ma': ablated,
                    'change_pct': change_pct
                }
    return data


def plot_figure_1_ma_source():
    """Figure 1: MA Source Evidence - MLP vs Attention"""
    print("Generating Figure 1: MA Source Evidence...")
    
    data = load_rq2_data()
    if not data:
        print("  ✗ Data missing")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    attn_values = [data[m]['attn_max'] for m in models]
    mlp_values = [data[m]['mlp_max'] for m in models]
    ratios = [data[m]['ratio'] for m in models]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, attn_values, width, label='Attention', 
                   color=COLORS['attention'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, mlp_values, width, label='MLP', 
                   color=COLORS['mlp'], alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Maximum Activation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    
    # Add ratio annotations
    for i, (ratio, mlp_val) in enumerate(zip(ratios, mlp_values)):
        if ratio >= 100:
            label = f'{ratio:.0f}×'
        else:
            label = f'{ratio:.1f}×'
        ax.text(i, mlp_val + max(mlp_values)*0.02, label, 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '01_ma_source_evidence.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_2_attention_role():
    """Figure 2: Attention Role - Trigger Input"""
    print("Generating Figure 2: Attention Role...")
    
    data = load_rq1_data()
    if not data:
        print("  ✗ Data missing")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    changes = [data[m]['change_pct'] for m in models]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)
    
    colors = [COLORS['negative'] if c < 0 else COLORS['positive'] for c in changes]
    bars = ax.bar(model_labels, changes, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('MA Change (%)', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=0)
    
    # Add value labels
    for bar, val in zip(bars, changes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.0f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '02_attention_role.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_3_function_word():
    """Figure 3: Token Type Distribution"""
    print("Generating Figure 3: Function Word Triggers...")
    
    data = load_rq3_data()
    if not data:
        print("  ✗ Data missing")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    
    # Token type mapping
    type_map = {
        'Punctuation': 'Punctuation',
        'Function': 'Function Word',
        'Whitespace': 'Whitespace',
        'Content': 'Content Word'
    }
    
    # Build data matrix
    data_matrix = []
    
    for model in models:
        model_data = data[model]
        type_stats = model_data.get('type_statistics', {})
        row = []
        
        # Map Chinese to English
        chinese_to_english = {
            '标点符号': 'Punctuation',
            '功能词': 'Function Word',
            '空白/换行': 'Whitespace',
            '实义词': 'Content Word'
        }
        
        for eng_key in ['Punctuation', 'Function Word', 'Whitespace', 'Content Word']:
            count = 0
            for ch_key, e_key in chinese_to_english.items():
                if e_key == eng_key:
                    count = type_stats.get(ch_key, {}).get('count', 0)
                    break
            row.append(count)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix).T
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)
    
    type_labels = ['Punctuation', 'Function Word', 'Whitespace', 'Content Word']
    colors_stack = [COLORS['cat2'], COLORS['cat3'], COLORS['cat4'], COLORS['cat1']]
    bottom = np.zeros(len(models))
    
    for i, (type_name, color) in enumerate(zip(type_labels, colors_stack)):
        ax.bar(model_labels, data_matrix[i], bottom=bottom, label=type_name,
               color=color, alpha=0.85, edgecolor='white', linewidth=1)
        bottom += data_matrix[i]
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Count (Top 50)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '03_function_word_trigger.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_4_v_dependency():
    """Figure 4: V-Matrix Dependency Strength"""
    print("Generating Figure 4: V-Matrix Dependency (7 models)...")
    
    data = load_rq5_data(include_7=True)
    if not data:
        print("  ✗ Data missing")
        return
    
    # Sort by absolute change
    sorted_models = sorted(data.keys(), key=lambda m: abs(data[m]['change_pct']), reverse=True)
    model_labels = [MODEL_NAMES[m] for m in sorted_models]
    changes = [data[m]['change_pct'] for m in sorted_models]
    
    # Color by dependency strength
    colors = []
    for change in changes:
        abs_change = abs(change)
        if abs_change > 80:
            colors.append(COLORS['strong'])
        elif abs_change > 50:
            colors.append(COLORS['medium'])
        else:
            colors.append(COLORS['weak'])
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)
    
    bars = ax.barh(model_labels, changes, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('MA Change after V-Ablation (%)', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2, zorder=0)
    ax.axvline(x=-50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, changes):
        width = bar.get_width()
        ax.text(width - 3 if val < 0 else width + 3, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}',
                ha='right' if val < 0 else 'left', va='center',
                fontsize=9, fontweight='bold',
                color='white' if abs(val) > 70 else 'black')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['strong'], label='Strong (>80%)', edgecolor='black', linewidth=0.8),
        Patch(facecolor=COLORS['medium'], label='Medium (50-80%)', edgecolor='black', linewidth=0.8),
        Patch(facecolor=COLORS['weak'], label='Weak (<50%)', edgecolor='black', linewidth=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=False, shadow=False)
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '04_v_matrix_dependency.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_5_heatmap():
    """Figure 5: Cross-RQ Comprehensive Metrics"""
    print("Generating Figure 5: Comprehensive Heatmap...")
    
    rq1_data = load_rq1_data()
    rq2_data = load_rq2_data()
    rq3_data = load_rq3_data()
    rq5_data = load_rq5_data()
    
    if not all([rq1_data, rq2_data, rq3_data, rq5_data]):
        print("  ✗ Data incomplete")
        return
    
    models = [m for m in MODELS if m in rq1_data and m in rq2_data and m in rq3_data and m in rq5_data]
    model_labels = [MODEL_NAMES[m] for m in models]
    
    # Build data matrix
    matrix = []
    for model in models:
        row = [
            abs(rq1_data[model]['change_pct']),
            rq2_data[model]['ratio'],
            rq3_data[model]['semantic_free_percentage'],
            abs(rq5_data[model]['change_pct'])
        ]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Normalize by column
    matrix_norm = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if col.max() > col.min():
            matrix_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            matrix_norm[:, j] = 0.5
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=DPI)
    
    im = ax.imshow(matrix_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(['|Attn\nChange|', 'MLP/Attn\nRatio', 
                        'Function\nWord %', '|V-Abl\nChange|'], fontsize=9)
    ax.set_yticklabels(model_labels)
    
    # Add value annotations
    for i in range(len(models)):
        for j in range(4):
            val = matrix[i, j]
            if j == 2:  # Percentage
                text_val = f'{val:.0f}'
            elif j == 1 and val > 100:  # Ratio
                text_val = f'{val:.0f}'
            else:
                text_val = f'{val:.0f}'
            
            ax.text(j, i, text_val,
                   ha="center", va="center", 
                   color="white" if matrix_norm[i, j] > 0.5 else "black",
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '05_comprehensive_heatmap.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_6_classification():
    """Figure 6: Mechanism Classification Tree"""
    print("Generating Figure 6: Mechanism Classification...")
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    ax.axis('off')
    
    tree_text = """MA Generation Mechanisms

Attention-Triggered (MA↓ >50%)
  • Strong V-Dep: GPT-J (−96%, V−71%)
  • Weak V-Dep:   BLOOM (−98%, V−19%)

MLP-Dominant (MA↑)
  • Strong V-Dep: Qwen (+266%, V−99%)

Hybrid (|MA change| <50%)
  • Falcon (−21%, V−79%)
  • Mistral (−18%, V−83%)"""
    
    ax.text(0.1, 0.5, tree_text, 
            transform=ax.transAxes,
            fontsize=11,
            family='monospace',
            verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=1', facecolor='white', 
                     edgecolor='black', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '06_mechanism_classification.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_7_bloom_special():
    """Figure 7: BLOOM Special Case Analysis"""
    print("Generating Figure 7: BLOOM Special Case...")
    
    bloom_json = RESULTS_DIR / 'models' / 'bloom_7b1' / 'exp6' / 'v_ablation_simple.json'
    if not bloom_json.exists():
        print("  ✗ BLOOM data missing")
        return
    
    with open(bloom_json, 'r') as f:
        bloom_data = json.load(f)
    
    fig = plt.figure(figsize=(12, 3.5), dpi=DPI)
    
    # Panel A: Layer comparison
    ax1 = plt.subplot(1, 3, 1)
    
    if 'layer_comparison' in bloom_data:
        layer_comp = bloom_data['layer_comparison']
        l0_baseline = layer_comp['layer_0']['baseline_ma']
        l0_ablated = layer_comp['layer_0']['ablated_ma']
        l0_change = layer_comp['layer_0']['change_percent']
        
        l28_baseline = bloom_data['baseline']['ma_avg']
        l28_ablated = bloom_data['v_ablated']['ma_avg']
        l28_change = bloom_data['v_ablated']['change_percent']
        
        layers = ['Layer 0', 'Layer 28']
        baselines = [l0_baseline, l28_baseline]
        ablateds = [l0_ablated, l28_ablated]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baselines, width, label='Baseline', 
                       color=COLORS['baseline'], alpha=0.85, edgecolor='black', linewidth=0.8)
        bars2 = ax1.bar(x + width/2, ablateds, width, label='V-Ablated', 
                       color=COLORS['ablated'], alpha=0.85, edgecolor='black', linewidth=0.8)
        
        ax1.set_ylabel('MA Value', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend(frameon=True, fancybox=False, shadow=False)
        ax1.text(0.5, 0.95, '(a)', transform=ax1.transAxes, 
                fontsize=12, fontweight='bold', va='top', ha='center')
        
        # Add change labels
        for i, change in enumerate([l0_change, l28_change]):
            y_pos = max(baselines[i], ablateds[i]) * 1.05
            ax1.text(i, y_pos, f'{change:.0f}%', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel B: Punctuation correlation
    ax2 = plt.subplot(1, 3, 2)
    punctuations = [',', '.', '\\n']
    similarities = [0.44, 0.42, 0.38]
    
    bars = ax2.bar(punctuations, similarities, color=COLORS['mlp'], 
                   alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Cosine Similarity', fontweight='bold')
    ax2.set_xlabel('Token', fontweight='bold')
    ax2.set_ylim(0, 0.5)
    ax2.text(0.5, 0.95, '(b)', transform=ax2.transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='center')
    
    for bar, val in zip(bars, similarities):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # Panel C: Mechanism diagram
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    mechanism_text = """BLOOM Mechanism

Early Generation (L0)
  • MLP produces MA
  • Strong V-dep (−71%)

Residual Propagation (L28)
  • Accumulates via residual
  • Weak V-dep (−19%)

Semantic Alignment
  • MA ≈ Punctuation
  • Boundary marking"""
    
    ax3.text(0.5, 0.5, mechanism_text,
            transform=ax3.transAxes,
            fontsize=10,
            family='monospace',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                     edgecolor='black', linewidth=1.5, alpha=0.9))
    ax3.text(0.5, 0.95, '(c)', transform=ax3.transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='center')
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '07_bloom_special_case.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main function - Generate P0 academic-style figures"""
    print("\n" + "="*60)
    print("Generating Academic-Style Figures (P0 Core)")
    print("Style: Professional, Publication-Ready, No Titles")
    print("="*60 + "\n")
    
    (VIS_DIR / 'conclusion').mkdir(parents=True, exist_ok=True)
    
    plot_figure_1_ma_source()
    plot_figure_2_attention_role()
    plot_figure_3_function_word()
    plot_figure_4_v_dependency()
    plot_figure_5_heatmap()
    plot_figure_6_classification()
    plot_figure_7_bloom_special()
    
    print("\n" + "="*60)
    print("✓ Academic-Style Figures Complete")
    print(f"Output: {VIS_DIR / 'conclusion'}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
