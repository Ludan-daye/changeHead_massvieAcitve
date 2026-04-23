#!/usr/bin/env python3
"""
Experiment 2A: MLP Feasibility Test - Do MLP Layers Generate Massive Activations?

This experiment tests whether MLP layers are responsible for generating massive activations
by comparing:
  1. Baseline: Normal model (all components active)
  2. All MLP Disabled: All 12 MLP layers zeroed out

If massive activations disappear when all MLPs are disabled, it proves MLPs generate them.
If they remain unchanged, the source must be elsewhere (LayerNorm, embeddings, residual).
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib
import monkey_patch as mp


class MLPDisableHook:
    """
    Hook to disable MLP layers by zeroing their output
    """
    def __init__(self, layer_id, mode='disable_all'):
        """
        Args:
            layer_id: Which layer this hook is for
            mode: 'disable_all' (zero out output)
        """
        self.layer_id = layer_id
        self.mode = mode

    def __call__(self, module, input, output):
        """
        Hook function that zeros out MLP output
        """
        if self.mode == 'disable_all':
            # Zero out entire MLP output
            return torch.zeros_like(output)
        else:
            return output


def run_experiment(args, mode='baseline'):
    """
    Run one configuration of the experiment

    Args:
        args: Command line arguments
        mode: 'baseline' or 'all_mlp_disabled'

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"Running Experiment: {mode.upper()}")
    print(f"{'='*80}\n")

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture for all layers
    print("Enabling feature capture for all layers...")
    for layer_id in range(len(layers)):
        lib.enable_custom_block(args.model, layers[layer_id], layer_id)

    # Register hooks for MLP disabling if not baseline
    hooks = []
    if mode != 'baseline':
        print(f"\nRegistering hooks for mode: {mode}")
        for layer_id in range(len(layers)):
            layer = layers[layer_id]

            if mode == 'all_mlp_disabled':
                # Disable all MLP layers
                hook = MLPDisableHook(layer_id, mode='disable_all')
                print(f"  Layer {layer_id}: Disabling MLP")

                # Register hook on the MLP module
                handle = layer.mlp.register_forward_hook(hook)
                hooks.append(handle)

    # Load data
    print("\nLoading dataset...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Register hooks to capture MLP and Attention outputs separately (for ratio calculation)
    mlp_output_store = {}
    attn_output_store = {}

    def make_output_hook(store, layer_id):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            store[layer_id] = out.detach().abs().max().item()
        return hook_fn

    ratio_hooks = []
    if mode == 'baseline':
        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            # MLP module
            mlp_module = getattr(layer, 'mlp', None)
            if mlp_module:
                h = mlp_module.register_forward_hook(make_output_hook(mlp_output_store, layer_id))
                ratio_hooks.append(h)
            # Attention module
            attn_module = getattr(layer, 'attn', None) or getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)
            if attn_module:
                h = attn_module.register_forward_hook(make_output_hook(attn_output_store, layer_id))
                ratio_hooks.append(h)

    # Storage for analysis
    n_layers = len(layers)
    layer_stats = {}
    ma_dims = None  # Dynamically detected

    for layer_id in range(n_layers):
        layer_stats[layer_id] = {
            'top1': [],
            'top2': [],
            'top3': [],
            'median': [],
            'ma_dim0': [],
            'ma_dim1': [],
            'mlp_max': [],
            'attn_max': [],
        }

    print(f"\nProcessing {len(testseq_list)} samples...")

    # Process samples
    with torch.no_grad():
        for idx, testseq in enumerate(tqdm(testseq_list, desc=f"Processing ({mode})")):
            # Forward pass
            _ = model(testseq)

            # Analyze each layer
            for layer_id in range(n_layers):
                layer = layers[layer_id]

                if not hasattr(layer, 'feat') or layer.feat is None:
                    continue

                feat_abs = layer.feat.abs()
                if len(feat_abs.shape) == 3:
                    feat_abs = feat_abs.view(-1, feat_abs.shape[-1])

                sorted_vals, _ = torch.sort(feat_abs.flatten(), descending=True)

                layer_stats[layer_id]['top1'].append(sorted_vals[0].item())
                layer_stats[layer_id]['top2'].append(sorted_vals[1].item())
                layer_stats[layer_id]['top3'].append(sorted_vals[2].item())
                layer_stats[layer_id]['median'].append(torch.median(feat_abs).item())

                # Detect MA dimensions dynamically
                if ma_dims is None and layer_id == 0:
                    detected = lib.detect_ma_dimensions(layer.feat, top_k=2)
                    ma_dims = [d[0] for d in detected]
                    print(f"  Detected MA dimensions: {ma_dims}")

                if ma_dims and feat_abs.shape[1] > max(ma_dims):
                    layer_stats[layer_id]['ma_dim0'].append(torch.max(feat_abs[:, ma_dims[0]]).item())
                    layer_stats[layer_id]['ma_dim1'].append(torch.max(feat_abs[:, ma_dims[1]]).item())

                # Record MLP/Attention output maxima
                if layer_id in mlp_output_store:
                    layer_stats[layer_id]['mlp_max'].append(mlp_output_store[layer_id])
                if layer_id in attn_output_store:
                    layer_stats[layer_id]['attn_max'].append(attn_output_store[layer_id])

    # Clean up hooks
    for handle in hooks:
        handle.remove()
    for handle in ratio_hooks:
        handle.remove()

    # Compute statistics
    print("\nComputing statistics...")
    results = {}

    for layer_id in range(n_layers):
        mlp_vals = layer_stats[layer_id]['mlp_max']
        attn_vals = layer_stats[layer_id]['attn_max']
        mlp_mean = float(np.mean(mlp_vals)) if mlp_vals else 0
        attn_mean = float(np.mean(attn_vals)) if attn_vals else 0

        results[layer_id] = {
            'top1_mean': np.mean(layer_stats[layer_id]['top1']),
            'top1_std': np.std(layer_stats[layer_id]['top1']),
            'top2_mean': np.mean(layer_stats[layer_id]['top2']),
            'top3_mean': np.mean(layer_stats[layer_id]['top3']),
            'median_mean': np.mean(layer_stats[layer_id]['median']),
            'ma_dim0_mean': np.mean(layer_stats[layer_id]['ma_dim0']) if layer_stats[layer_id]['ma_dim0'] else 0,
            'ma_dim1_mean': np.mean(layer_stats[layer_id]['ma_dim1']) if layer_stats[layer_id]['ma_dim1'] else 0,
            'ma_dim0_max': np.max(layer_stats[layer_id]['ma_dim0']) if layer_stats[layer_id]['ma_dim0'] else 0,
            'ma_dim1_max': np.max(layer_stats[layer_id]['ma_dim1']) if layer_stats[layer_id]['ma_dim1'] else 0,
            'mlp_output_mean': mlp_mean,
            'attn_output_mean': attn_mean,
            'mlp_attn_ratio': mlp_mean / attn_mean if attn_mean > 0 else 0,
        }

        key_layers = sorted(set([0, n_layers//6, n_layers//3, n_layers-2, n_layers-1]))
        if layer_id in key_layers:
            ratio_str = f", MLP/Attn ratio: {results[layer_id]['mlp_attn_ratio']:.1f}×" if results[layer_id]['mlp_attn_ratio'] > 0 else ""
            print(f"\nLayer {layer_id}:")
            print(f"  Top1: {results[layer_id]['top1_mean']:.2f} ± {results[layer_id]['top1_std']:.2f}")
            print(f"  Median: {results[layer_id]['median_mean']:.2f}{ratio_str}")

    return results, layer_stats


def generate_visualizations(baseline_results, disabled_results, baseline_stats, disabled_stats, savedir):
    """
    Generate comprehensive comparison visualizations
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    n_layers = len(baseline_results)

    # Create output directories
    comparison_dir = os.path.join(savedir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # ===== Figure 1: Top1 Activation Comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    layers = list(range(n_layers))
    baseline_top1 = [baseline_results[i]['top1_mean'] for i in layers]
    disabled_top1 = [disabled_results[i]['top1_mean'] for i in layers]

    # Plot 1: Side-by-side comparison
    ax = axes[0, 0]
    x = np.arange(n_layers)
    width = 0.35
    ax.bar(x - width/2, baseline_top1, width, label='Baseline (Normal)', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, disabled_top1, width, label='All MLP Disabled', alpha=0.8, color='#F18F01')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top1 Activation Value', fontsize=12, fontweight='bold')
    ax.set_title('Top1 Activation: Baseline vs All MLP Disabled', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Difference (absolute)
    ax = axes[0, 1]
    diff = np.array(baseline_top1) - np.array(disabled_top1)
    colors = ['red' if d > 0 else 'green' for d in diff]
    ax.bar(layers, diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Difference (Baseline - Disabled)', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Difference in Top1 Activation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: MA Primary Dimension (Most important)
    ax = axes[1, 0]
    baseline_ma1 = [baseline_results[i]['ma_dim1_mean'] for i in layers]
    disabled_ma1 = [disabled_results[i]['ma_dim1_mean'] for i in layers]
    ax.plot(layers, baseline_ma1, 'o-', linewidth=3, markersize=8, label='Baseline', color='#2E86AB')
    ax.plot(layers, disabled_ma1, 's-', linewidth=3, markersize=8, label='All MLP Disabled', color='#F18F01')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('MA Primary Dimension Activation', fontsize=12, fontweight='bold')
    ax.set_title('MA Primary Dimension: Primary Massive Activation Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    # Plot 4: MA Secondary Dimension
    ax = axes[1, 1]
    baseline_ma0 = [baseline_results[i]['ma_dim0_mean'] for i in layers]
    disabled_ma0 = [disabled_results[i]['ma_dim0_mean'] for i in layers]
    ax.plot(layers, baseline_ma0, 'o-', linewidth=3, markersize=8, label='Baseline', color='#2E86AB')
    ax.plot(layers, disabled_ma0, 's-', linewidth=3, markersize=8, label='All MLP Disabled', color='#F18F01')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('MA Secondary Dimension Activation', fontsize=12, fontweight='bold')
    ax.set_title('MA Secondary Dimension: Secondary Massive Activation Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp2a_top1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp2a_top1_comparison.png")

    # ===== Figure 2: Percentage Change Heatmap =====
    fig, ax = plt.subplots(figsize=(16, 6))

    metrics = ['top1_mean', 'top2_mean', 'top3_mean', 'median_mean', 'ma_dim0_mean', 'ma_dim1_mean']
    metric_labels = ['Top 1', 'Top 2', 'Top 3', 'Median', 'MA Dim 0', 'MA Dim 1']

    change_matrix = np.zeros((len(metrics), n_layers))

    for i, metric in enumerate(metrics):
        for layer_id in range(n_layers):
            baseline_val = baseline_results[layer_id][metric]
            disabled_val = disabled_results[layer_id][metric]

            if baseline_val > 0:
                pct_change = 100 * (disabled_val - baseline_val) / baseline_val
            else:
                pct_change = 0

            change_matrix[i, layer_id] = pct_change

    sns.heatmap(change_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0,
                xticklabels=layers, yticklabels=metric_labels,
                cbar_kws={'label': 'Percentage Change (%)'}, ax=ax)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title('Percentage Change: (All MLP Disabled - Baseline) / Baseline × 100%',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp2a_percentage_change_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp2a_percentage_change_heatmap.png")

    # ===== Figure 3: Detailed Layer-wise Breakdown =====
    fig = plt.figure(figsize=(20, 12))

    for i, layer_id in enumerate([0, 2, 5, 7, 10, 11]):
        ax = plt.subplot(2, 3, i+1)

        metrics_to_plot = ['top1_mean', 'top2_mean', 'top3_mean', 'median_mean']
        labels = ['Top 1', 'Top 2', 'Top 3', 'Median']

        baseline_vals = [baseline_results[layer_id][m] for m in metrics_to_plot]
        disabled_vals = [disabled_results[layer_id][m] for m in metrics_to_plot]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='#2E86AB')
        ax.bar(x + width/2, disabled_vals, width, label='All MLP Disabled', alpha=0.8, color='#F18F01')

        ax.set_ylabel('Activation Value', fontsize=10, fontweight='bold')
        ax.set_title(f'Layer {layer_id}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Layer-wise Detailed Comparison: Key Layers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp2a_layerwise_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp2a_layerwise_breakdown.png")

    # ===== Figure 4: Critical Dimensions Focus =====
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # MA Primary Dimension with error bars
    ax = axes[0]
    baseline_ma1_mean = [baseline_results[i]['ma_dim1_mean'] for i in layers]
    disabled_ma1_mean = [disabled_results[i]['ma_dim1_mean'] for i in layers]
    baseline_ma1_max = [baseline_results[i]['ma_dim1_max'] for i in layers]
    disabled_ma1_max = [disabled_results[i]['ma_dim1_max'] for i in layers]

    ax.plot(layers, baseline_ma1_mean, 'o-', linewidth=3, markersize=10,
            label='Baseline (Mean)', color='#2E86AB')
    ax.plot(layers, disabled_ma1_mean, 's-', linewidth=3, markersize=10,
            label='All MLP Disabled (Mean)', color='#F18F01')
    ax.fill_between(layers, baseline_ma1_mean, baseline_ma1_max, alpha=0.2, color='#2E86AB')
    ax.fill_between(layers, disabled_ma1_mean, disabled_ma1_max, alpha=0.2, color='#F18F01')

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation Value (MA Primary Dimension)', fontsize=12, fontweight='bold')
    ax.set_title('MA Primary Dimension: Most Critical Massive Activation Dimension',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    # MA Secondary Dimension with error bars
    ax = axes[1]
    baseline_ma0_mean = [baseline_results[i]['ma_dim0_mean'] for i in layers]
    disabled_ma0_mean = [disabled_results[i]['ma_dim0_mean'] for i in layers]
    baseline_ma0_max = [baseline_results[i]['ma_dim0_max'] for i in layers]
    disabled_ma0_max = [disabled_results[i]['ma_dim0_max'] for i in layers]

    ax.plot(layers, baseline_ma0_mean, 'o-', linewidth=3, markersize=10,
            label='Baseline (Mean)', color='#2E86AB')
    ax.plot(layers, disabled_ma0_mean, 's-', linewidth=3, markersize=10,
            label='All MLP Disabled (Mean)', color='#F18F01')
    ax.fill_between(layers, baseline_ma0_mean, baseline_ma0_max, alpha=0.2, color='#2E86AB')
    ax.fill_between(layers, disabled_ma0_mean, disabled_ma0_max, alpha=0.2, color='#F18F01')

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation Value (MA Secondary Dimension)', fontsize=12, fontweight='bold')
    ax.set_title('MA Secondary Dimension: Secondary Massive Activation Dimension',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp2a_critical_dimensions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp2a_critical_dimensions.png")

    print("\n✅ All visualizations generated successfully!")


def generate_summary_report(baseline_results, disabled_results, savedir):
    """
    Generate a comprehensive text summary report
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EXPERIMENT 2A: MLP FEASIBILITY TEST - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("RESEARCH QUESTION:")
    report_lines.append("  Do MLP layers generate massive activations?")
    report_lines.append("")
    report_lines.append("METHODOLOGY:")
    report_lines.append("  1. Baseline: Run model with all MLP layers active")
    report_lines.append("  2. All Disabled: Run model with all 12 MLP layers zeroed out")
    report_lines.append("  3. Compare massive activation magnitudes")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")

    # Calculate key metrics
    n_layers = len(baseline_results)

    # MA Primary Dimension (most important)
    ma1_baseline_peak = max([baseline_results[i]['ma_dim1_mean'] for i in range(n_layers)])
    ma1_disabled_peak = max([disabled_results[i]['ma_dim1_mean'] for i in range(n_layers)])
    ma1_peak_layer = max(range(n_layers), key=lambda i: baseline_results[i]['ma_dim1_mean'])
    ma1_change_pct = 100 * (ma1_disabled_peak - ma1_baseline_peak) / ma1_baseline_peak if ma1_baseline_peak > 0 else 0

    # MA Secondary Dimension (secondary)
    ma0_baseline_peak = max([baseline_results[i]['ma_dim0_mean'] for i in range(n_layers)])
    ma0_disabled_peak = max([disabled_results[i]['ma_dim0_mean'] for i in range(n_layers)])
    ma0_peak_layer = max(range(n_layers), key=lambda i: baseline_results[i]['ma_dim0_mean'])
    ma0_change_pct = 100 * (ma0_disabled_peak - ma0_baseline_peak) / ma0_baseline_peak if ma0_baseline_peak > 0 else 0

    # Top1 overall
    top1_baseline_peak = max([baseline_results[i]['top1_mean'] for i in range(n_layers)])
    top1_disabled_peak = max([disabled_results[i]['top1_mean'] for i in range(n_layers)])
    top1_peak_layer = max(range(n_layers), key=lambda i: baseline_results[i]['top1_mean'])
    top1_change_pct = 100 * (top1_disabled_peak - top1_baseline_peak) / top1_baseline_peak if top1_baseline_peak > 0 else 0

    report_lines.append("1️⃣ MA PRIMARY DIMENSION (Primary Massive Activation Dimension)")
    report_lines.append("-" * 70)
    report_lines.append(f"  Baseline Peak Value:       {ma1_baseline_peak:.2f} (Layer {ma1_peak_layer})")
    report_lines.append(f"  All MLP Disabled Value:    {ma1_disabled_peak:.2f}")
    report_lines.append(f"  Absolute Change:           {ma1_disabled_peak - ma1_baseline_peak:.2f}")
    report_lines.append(f"  Percentage Change:         {ma1_change_pct:.2f}%")
    report_lines.append("")

    if ma1_change_pct < -50:
        conclusion_ma = "✅ MASSIVE DECREASE - MLP layers ARE the primary source!"
    elif ma1_change_pct < -10:
        conclusion_ma = "⚠️ SIGNIFICANT DECREASE - MLP layers contribute substantially"
    elif abs(ma1_change_pct) < 10:
        conclusion_ma = "❌ MINIMAL IMPACT - MLP layers are NOT the source"
    else:
        conclusion_ma = "🔍 UNEXPECTED INCREASE - Requires investigation"

    report_lines.append(f"  Conclusion: {conclusion_ma}")
    report_lines.append("")

    report_lines.append("2️⃣ DIMENSION 138 (Secondary Massive Activation Dimension)")
    report_lines.append("-" * 70)
    report_lines.append(f"  Baseline Peak Value:       {ma0_baseline_peak:.2f} (Layer {ma0_peak_layer})")
    report_lines.append(f"  All MLP Disabled Value:    {ma0_disabled_peak:.2f}")
    report_lines.append(f"  Absolute Change:           {ma0_disabled_peak - ma0_baseline_peak:.2f}")
    report_lines.append(f"  Percentage Change:         {ma0_change_pct:.2f}%")
    report_lines.append("")

    if ma0_change_pct < -50:
        conclusion_138 = "✅ MASSIVE DECREASE - MLP layers ARE the primary source!"
    elif ma0_change_pct < -10:
        conclusion_138 = "⚠️ SIGNIFICANT DECREASE - MLP layers contribute substantially"
    elif abs(ma0_change_pct) < 10:
        conclusion_138 = "❌ MINIMAL IMPACT - MLP layers are NOT the source"
    else:
        conclusion_138 = "🔍 UNEXPECTED INCREASE - Requires investigation"

    report_lines.append(f"  Conclusion: {conclusion_138}")
    report_lines.append("")

    report_lines.append("3️⃣ OVERALL TOP1 ACTIVATION")
    report_lines.append("-" * 70)
    report_lines.append(f"  Baseline Peak Value:       {top1_baseline_peak:.2f} (Layer {top1_peak_layer})")
    report_lines.append(f"  All MLP Disabled Value:    {top1_disabled_peak:.2f}")
    report_lines.append(f"  Absolute Change:           {top1_disabled_peak - top1_baseline_peak:.2f}")
    report_lines.append(f"  Percentage Change:         {top1_change_pct:.2f}%")
    report_lines.append("")

    # Layer-wise breakdown
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("LAYER-WISE DETAILED ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")

    report_lines.append(f"{'Layer':<7} {'Baseline Top1':<15} {'Disabled Top1':<15} {'Change':<12} {'% Change':<12}")
    report_lines.append("-" * 70)

    for layer_id in range(n_layers):
        baseline_val = baseline_results[layer_id]['top1_mean']
        disabled_val = disabled_results[layer_id]['top1_mean']
        change = disabled_val - baseline_val
        pct_change = 100 * change / baseline_val if baseline_val > 0 else 0

        report_lines.append(f"{layer_id:<7} {baseline_val:<15.2f} {disabled_val:<15.2f} {change:<12.2f} {pct_change:<12.2f}%")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("OVERALL CONCLUSION")
    report_lines.append("="*80)
    report_lines.append("")

    # Determine overall conclusion
    avg_change = np.mean([100 * (disabled_results[i]['top1_mean'] - baseline_results[i]['top1_mean']) / baseline_results[i]['top1_mean']
                          for i in range(n_layers) if baseline_results[i]['top1_mean'] > 0])

    if avg_change < -50:
        overall_conclusion = """
✅ MLP LAYERS ARE THE PRIMARY SOURCE OF MASSIVE ACTIVATIONS!

The experiment shows that disabling all 12 MLP layers causes massive activations
to decrease by >50% on average. This is DEFINITIVE PROOF that:

  1. MLP layers GENERATE massive activations (unlike attention heads)
  2. The source has been identified!
  3. This aligns perfectly with Experiment 1 which showed attention heads are readers

NEXT STEPS:
  → Proceed to Experiment 2B: Identify which specific MLP layers are most critical
  → Focus on Layer 2 (where MA dimension shows extreme amplification)
  → Investigate MLP internal mechanisms in Experiment 2C
"""
    elif avg_change < -10:
        overall_conclusion = """
⚠️ MLP LAYERS SIGNIFICANTLY CONTRIBUTE TO MASSIVE ACTIVATIONS

The experiment shows moderate decreases (10-50%) when disabling MLP layers.
This suggests:

  1. MLP layers are a major contributor to massive activations
  2. Other components (LayerNorm, residual) may also play a role
  3. Combined effect from multiple sources

NEXT STEPS:
  → Proceed to Experiment 2B to identify critical MLP layers
  → Also investigate LayerNorm and residual connections
  → May need combined ablation studies
"""
    elif abs(avg_change) < 10:
        overall_conclusion = """
❌ MLP LAYERS DO NOT GENERATE MASSIVE ACTIVATIONS

The experiment shows that disabling all MLP layers has minimal impact (<10%)
on massive activations. This is surprising but suggests:

  1. Neither attention heads NOR MLPs generate massive activations
  2. The source must be LayerNorm, embeddings, or residual connections
  3. Or possibly an emergent property from layer interactions

NEXT STEPS:
  → Skip Experiment 2B and 2C (MLP internal analysis)
  → Test LayerNorm parameters and effects
  → Investigate embedding layer and residual connections
  → Consider positional encodings
"""
    else:
        overall_conclusion = """
🔍 UNEXPECTED INCREASE IN ACTIVATIONS

Disabling MLP layers INCREASED activations, which is counterintuitive.
Possible explanations:

  1. Residual connections amplifying pre-MLP values
  2. LayerNorm compensation effects
  3. Numerical instability in the disabled configuration

NEXT STEPS:
  → Investigate why disabling MLPs increases activations
  → Check if residual connections are causing accumulation
  → Verify implementation correctness
"""

    report_lines.append(overall_conclusion)
    report_lines.append("")
    report_lines.append("="*80)

    # Save report
    report_path = os.path.join(savedir, 'comparison', 'EXPERIMENT_2A_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Also print to console
    print('\n'.join(report_lines))

    print(f"\n✅ Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2A: Test if MLP layers generate massive activations'
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--access_token', type=str, default='type in your access token here',
                        help='Hugging Face access token')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'c4', 'RedPajama'], help='Dataset name')
    parser.add_argument('--nsamples', type=int, default=30,
                        help='Number of samples to analyze')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Output arguments
    parser.add_argument('--savedir', type=str, default='results/exp2a_mlp_feasibility_test/',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create directory structure
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'baseline'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'all_mlp_disabled'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'comparison'), exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENT 2A: MLP FEASIBILITY TEST")
    print("="*80)
    print("\nResearch Question:")
    print("  Do MLP layers generate massive activations?")
    print("\nMethod:")
    print("  1. Run baseline (all components active)")
    print("  2. Run with all 12 MLP layers disabled")
    print("  3. Compare activation magnitudes")
    print("\n" + "="*80)

    # Run baseline
    print("\n🔵 PHASE 1: Running Baseline Experiment")
    baseline_results, baseline_stats = run_experiment(args, mode='baseline')

    # Save baseline results
    with open(os.path.join(args.savedir, 'baseline', 'results.json'), 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                   for k, v in baseline_results.items()}, f, indent=2)

    # Run all MLP disabled
    print("\n🟠 PHASE 2: Running All MLP Disabled Experiment")
    disabled_results, disabled_stats = run_experiment(args, mode='all_mlp_disabled')

    # Save disabled results
    with open(os.path.join(args.savedir, 'all_mlp_disabled', 'results.json'), 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                   for k, v in disabled_results.items()}, f, indent=2)

    # Generate visualizations
    print("\n🎨 PHASE 3: Generating Visualizations")
    generate_visualizations(baseline_results, disabled_results,
                          baseline_stats, disabled_stats, args.savedir)

    # Generate summary report
    print("\n📊 PHASE 4: Generating Summary Report")
    generate_summary_report(baseline_results, disabled_results, args.savedir)

    print("\n" + "="*80)
    print("✅ EXPERIMENT 2A COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("\nGenerated files:")
    print("  📁 baseline/")
    print("     └─ results.json")
    print("  📁 all_mlp_disabled/")
    print("     └─ results.json")
    print("  📁 comparison/")
    print("     ├─ exp2a_top1_comparison.png")
    print("     ├─ exp2a_percentage_change_heatmap.png")
    print("     ├─ exp2a_layerwise_breakdown.png")
    print("     ├─ exp2a_critical_dimensions.png")
    print("     └─ EXPERIMENT_2A_SUMMARY.txt")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
