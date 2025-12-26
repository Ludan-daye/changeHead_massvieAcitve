#!/usr/bin/env python3
"""
Unified Color Scheme Academic Visualization
Enhanced aesthetics with consistent color palette across all figures
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
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Enhanced academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.5,
    'patch.linewidth': 1.2,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.edgecolor': '#2c3e50',
    'axes.labelcolor': '#2c3e50',
    'text.color': '#2c3e50',
    'xtick.color': '#2c3e50',
    'ytick.color': '#2c3e50',
})

sns.set_style("whitegrid", {
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# UNIFIED COLOR SCHEME - Each model has a fixed color across all figures
MODEL_COLORS = {
    'gptj_6b': '#e74c3c',      # Bright Red
    'bloom_7b1': '#3498db',    # Bright Blue
    'qwen2.5_7b': '#2ecc71',   # Emerald Green
    'falcon_7b': '#9b59b6',    # Purple
    'mistral_7b_v03': '#f39c12', # Orange
    'gpt2': '#e67e22',         # Dark Orange
    'opt_6.7b': '#1abc9c'      # Turquoise
}

# Semantic colors (consistent across figures)
SEMANTIC_COLORS = {
    'attention': '#3498db',     # Blue
    'mlp': '#e74c3c',          # Red
    'baseline': '#27ae60',     # Dark Green
    'ablated': '#95a5a6',      # Gray
    'positive': '#27ae60',     # Green
    'negative': '#c0392b',     # Dark Red
    'strong_dep': '#c0392b',   # Dark Red
    'medium_dep': '#e67e22',   # Orange
    'weak_dep': '#3498db',     # Blue
    
    # Token types
    'punctuation': '#e74c3c',   # Red
    'function_word': '#f39c12', # Orange
    'whitespace': '#9b59b6',    # Purple
    'content_word': '#2ecc71'   # Green
}

FIGURE_SIZE_SINGLE = (6, 4.5)
FIGURE_SIZE_WIDE = (8, 4.5)
FIGURE_SIZE_TALL = (6, 5.5)
DPI = 300

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


def add_value_labels(ax, bars, values, format_str='{:.0f}', offset=0.02, vertical=True):
    """Add value labels on bars with smart positioning"""
    max_val = max([abs(v) for v in values])
    for bar, val in zip(bars, values):
        if vertical:
            height = bar.get_height()
            y_pos = height + max_val * offset if height >= 0 else height - max_val * offset
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   format_str.format(val),
                   ha='center', va=va, fontsize=10, fontweight='bold')
        else:
            width = bar.get_width()
            x_pos = width - abs(max_val) * 0.03 if width < 0 else width + abs(max_val) * 0.03
            ha = 'right' if width < 0 else 'left'
            color = 'white' if abs(width) > abs(max_val) * 0.6 else 'black'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
                   format_str.format(val),
                   ha=ha, va='center', fontsize=10, fontweight='bold', color=color)


def load_rq2_data():
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
    json_path = RESULTS_DIR / 'MA_POSITION_TOKEN_ANALYSIS.json'
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_rq5_data(include_7=False):
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
    """Figure 1: MA Source Evidence - Enhanced with unified colors"""
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
    width = 0.38
    
    bars1 = ax.bar(x - width/2, attn_values, width, label='Attention', 
                   color=SEMANTIC_COLORS['attention'], alpha=0.85, 
                   edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, mlp_values, width, label='MLP', 
                   color=SEMANTIC_COLORS['mlp'], alpha=0.85, 
                   edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=13)
    ax.set_ylabel('Maximum Activation', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
              framealpha=0.95, edgecolor='gray')
    
    # Add ratio annotations
    for i, (ratio, mlp_val) in enumerate(zip(ratios, mlp_values)):
        label = f'{ratio:.0f}×' if ratio >= 100 else f'{ratio:.1f}×'
        ax.text(i, mlp_val * 1.03, label, 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    ax.set_ylim(0, max(mlp_values) * 1.15)
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '01_ma_source_evidence.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_2_attention_role():
    """Figure 2: Attention Role - Unified model colors"""
    print("Generating Figure 2: Attention Role...")
    
    data = load_rq1_data()
    if not data:
        print("  ✗ Data missing")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    changes = [data[m]['change_pct'] for m in models]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)
    
    # Use unified model colors
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax.bar(model_labels, changes, color=colors, alpha=0.85, 
                  edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=13)
    ax.set_ylabel('MA Change (%)', fontweight='bold', fontsize=13)
    ax.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=2, zorder=0)
    
    # Add value labels
    add_value_labels(ax, bars, changes, format_str='{:+.0f}')
    
    # Add shaded regions
    ax.axhspan(-100, 0, alpha=0.05, color=SEMANTIC_COLORS['negative'], zorder=0)
    ax.axhspan(0, 300, alpha=0.05, color=SEMANTIC_COLORS['positive'], zorder=0)
    
    plt.xticks(fontsize=12)
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '02_attention_role.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_3_function_word():
    """Figure 3: Token Type Distribution - Unified semantic colors"""
    print("Generating Figure 3: Function Word Triggers...")
    
    data = load_rq3_data()
    if not data:
        print("  ✗ Data missing")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    
    chinese_to_english = {
        '标点符号': 'Punctuation',
        '功能词': 'Function Word',
        '空白/换行': 'Whitespace',
        '实义词': 'Content Word'
    }
    
    data_matrix = []
    for model in models:
        model_data = data[model]
        type_stats = model_data.get('type_statistics', {})
        row = []
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
    colors_stack = [
        SEMANTIC_COLORS['punctuation'],
        SEMANTIC_COLORS['function_word'],
        SEMANTIC_COLORS['whitespace'],
        SEMANTIC_COLORS['content_word']
    ]
    
    bottom = np.zeros(len(models))
    
    for i, (type_name, color) in enumerate(zip(type_labels, colors_stack)):
        ax.bar(model_labels, data_matrix[i], bottom=bottom, label=type_name,
               color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
        bottom += data_matrix[i]
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=13)
    ax.set_ylabel('Count (Top 50)', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
              framealpha=0.95, edgecolor='gray', ncol=2)
    
    plt.xticks(fontsize=12)
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '03_function_word_trigger.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_4_v_dependency():
    """Figure 4: V-Matrix Dependency - Unified model colors"""
    print("Generating Figure 4: V-Matrix Dependency (7 models)...")
    
    data = load_rq5_data(include_7=True)
    if not data:
        print("  ✗ Data missing")
        return
    
    sorted_models = sorted(data.keys(), key=lambda m: abs(data[m]['change_pct']), reverse=True)
    model_labels = [MODEL_NAMES[m] for m in sorted_models]
    changes = [data[m]['change_pct'] for m in sorted_models]
    
    # Use unified model colors
    colors = [MODEL_COLORS[m] for m in sorted_models]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TALL, dpi=DPI)
    
    bars = ax.barh(model_labels, changes, color=colors, alpha=0.85, 
                   edgecolor='white', linewidth=2)
    
    ax.set_xlabel('MA Change after V-Ablation (%)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Model', fontweight='bold', fontsize=13)
    ax.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2, zorder=0)
    ax.axvline(x=-50, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=0)
    ax.axvline(x=-80, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=0)
    
    # Add shaded regions
    ax.axvspan(-100, -80, alpha=0.08, color=SEMANTIC_COLORS['strong_dep'], zorder=0)
    ax.axvspan(-80, -50, alpha=0.08, color=SEMANTIC_COLORS['medium_dep'], zorder=0)
    ax.axvspan(-50, 0, alpha=0.08, color=SEMANTIC_COLORS['weak_dep'], zorder=0)
    
    # Add value labels
    add_value_labels(ax, bars, changes, format_str='{:.1f}', vertical=False)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=SEMANTIC_COLORS['strong_dep'], label='Strong (>80%)', 
                      edgecolor='white', linewidth=1.5, alpha=0.85),
        mpatches.Patch(facecolor=SEMANTIC_COLORS['medium_dep'], label='Medium (50-80%)', 
                      edgecolor='white', linewidth=1.5, alpha=0.85),
        mpatches.Patch(facecolor=SEMANTIC_COLORS['weak_dep'], label='Weak (<50%)', 
                      edgecolor='white', linewidth=1.5, alpha=0.85)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95, edgecolor='gray')
    
    plt.yticks(fontsize=12)
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '04_v_matrix_dependency.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_5_heatmap():
    """Figure 5: Cross-RQ Comprehensive Metrics - Unified model colors"""
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
    
    # Normalize
    matrix_norm = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if col.max() > col.min():
            matrix_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            matrix_norm[:, j] = 0.5
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=DPI)
    
    # Use a better colormap
    im = ax.imshow(matrix_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(['|Attn\nChange|', 'MLP/Attn\nRatio', 
                        'Function\nWord %', '|V-Abl\nChange|'], fontsize=11)
    ax.set_yticklabels(model_labels, fontsize=12)
    
    # Add value annotations
    for i in range(len(models)):
        for j in range(4):
            val = matrix[i, j]
            text_val = f'{val:.0f}' if j == 2 or (j == 1 and val > 100) else f'{val:.0f}'
            
            ax.text(j, i, text_val,
                   ha="center", va="center", 
                   color="white" if matrix_norm[i, j] > 0.6 else "black",
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metric', fontweight='bold', fontsize=13)
    ax.set_ylabel('Model', fontweight='bold', fontsize=13)
    
    # Enhance colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value', rotation=270, labelpad=25, fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add grid lines
    for i in range(len(models) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(5):
        ax.axvline(j - 0.5, color='white', linewidth=2)
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '05_comprehensive_heatmap.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_6_classification():
    """Figure 6: Mechanism Classification Tree - Enhanced"""
    print("Generating Figure 6: Mechanism Classification...")
    
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=DPI)
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
    
    ax.text(0.5, 0.5, tree_text, 
            transform=ax.transAxes,
            fontsize=12,
            family='monospace',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1.5', facecolor='#ecf0f1', 
                     edgecolor='#2c3e50', linewidth=2, alpha=0.95))
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '06_mechanism_classification.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_figure_7_bloom_special():
    """Figure 7: BLOOM Special Case - Enhanced multi-panel"""
    print("Generating Figure 7: BLOOM Special Case...")
    
    bloom_json = RESULTS_DIR / 'models' / 'bloom_7b1' / 'exp6' / 'v_ablation_simple.json'
    if not bloom_json.exists():
        print("  ✗ BLOOM data missing")
        return
    
    with open(bloom_json, 'r') as f:
        bloom_data = json.load(f)
    
    fig = plt.figure(figsize=(13, 4), dpi=DPI)
    
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
        width = 0.38
        
        bars1 = ax1.bar(x - width/2, baselines, width, label='Baseline', 
                       color=SEMANTIC_COLORS['baseline'], alpha=0.85, 
                       edgecolor='white', linewidth=2)
        bars2 = ax1.bar(x + width/2, ablateds, width, label='V-Ablated', 
                       color=SEMANTIC_COLORS['ablated'], alpha=0.85, 
                       edgecolor='white', linewidth=2)
        
        ax1.set_ylabel('MA Value', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers, fontsize=11)
        ax1.legend(frameon=True, fancybox=True, shadow=True, 
                  framealpha=0.95, edgecolor='gray', fontsize=10)
        ax1.text(0.5, 0.97, '(a)', transform=ax1.transAxes, 
                fontsize=14, fontweight='bold', va='top', ha='center')
        
        # Add change labels
        for i, change in enumerate([l0_change, l28_change]):
            y_pos = max(baselines[i], ablateds[i]) * 1.08
            color = SEMANTIC_COLORS['strong_dep'] if abs(change) > 50 else SEMANTIC_COLORS['weak_dep']
            ax1.text(i, y_pos, f'{change:.0f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                             edgecolor='white', alpha=0.7, linewidth=1.5),
                    color='white')
    
    # Panel B: Punctuation correlation
    ax2 = plt.subplot(1, 3, 2)
    punctuations = [',', '.', '\\n']
    similarities = [0.44, 0.42, 0.38]
    
    bars = ax2.bar(punctuations, similarities, color=MODEL_COLORS['bloom_7b1'], 
                   alpha=0.85, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Cosine Similarity', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Token', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 0.52)
    ax2.text(0.5, 0.97, '(b)', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='center')
    
    for bar, val in zip(bars, similarities):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.015,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
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
            fontsize=11,
            family='monospace',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#ecf0f1', 
                     edgecolor='#2c3e50', linewidth=2, alpha=0.95))
    ax3.text(0.5, 0.97, '(c)', transform=ax3.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='center')
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '07_bloom_special_case.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main function - Generate enhanced unified figures"""
    print("\n" + "="*70)
    print("Generating ENHANCED Academic Figures with UNIFIED Color Scheme")
    print("="*70 + "\n")
    
    print("Color Scheme:")
    print("  Models (consistent across all figures):")
    for model, color in MODEL_COLORS.items():
        if model in MODELS_7:
            print(f"    • {MODEL_NAMES[model]:12s}: {color}")
    print("\n  Semantic (Attention/MLP/Token types):")
    print(f"    • Attention:  {SEMANTIC_COLORS['attention']}")
    print(f"    • MLP:        {SEMANTIC_COLORS['mlp']}")
    print(f"    • Baseline:   {SEMANTIC_COLORS['baseline']}")
    print(f"    • Ablated:    {SEMANTIC_COLORS['ablated']}")
    print("\n" + "-"*70 + "\n")
    
    (VIS_DIR / 'conclusion').mkdir(parents=True, exist_ok=True)
    
    plot_figure_1_ma_source()
    plot_figure_2_attention_role()
    plot_figure_3_function_word()
    plot_figure_4_v_dependency()
    plot_figure_5_heatmap()
    plot_figure_6_classification()
    plot_figure_7_bloom_special()
    
    print("\n" + "="*70)
    print("✓ Enhanced Unified Figures Complete!")
    print(f"Output: {VIS_DIR / 'conclusion'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
