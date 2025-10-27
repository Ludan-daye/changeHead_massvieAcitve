#!/usr/bin/env python3
"""
Experiment 1: Feasibility Test - Do Attention Heads Generate Massive Activations?

This experiment tests whether attention heads are responsible for generating massive activations
by comparing:
  1. Baseline: Normal model (no pruning)
  2. All Heads Disabled: All 144 attention heads set to zero

If massive activations disappear when all heads are disabled, it proves heads generate them.
If they remain unchanged, massive activations come from MLP layers or other sources.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp


class HeadDisableHook:
    """
    Hook to disable all attention heads by zeroing their output
    """
    def __init__(self, layer_id, num_heads, mode='disable_all', enable_heads=None):
        """
        Args:
            layer_id: Which layer this hook is for
            num_heads: Number of attention heads in the layer
            mode: 'disable_all' or 'disable_except'
            enable_heads: List of head indices to keep enabled (only for 'disable_except')
        """
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.mode = mode
        self.enable_heads = enable_heads if enable_heads is not None else []
        self.hook_registered = False

    def __call__(self, module, input, output):
        """
        Hook function that zeros out attention head outputs
        """
        # output[0] is the attention output, shape: [batch, seq_len, hidden_dim]
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads

        # Reshape to [batch, seq_len, num_heads, head_dim]
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)

        if self.mode == 'disable_all':
            # Zero out all heads
            attn_output_reshaped[:, :, :, :] = 0
        elif self.mode == 'disable_except':
            # Zero out all heads except those in enable_heads
            for head_idx in range(self.num_heads):
                if head_idx not in self.enable_heads:
                    attn_output_reshaped[:, :, head_idx, :] = 0

        # Reshape back to [batch, seq_len, hidden_dim]
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)

        return (modified_output,) + output[1:]


def run_experiment(args, mode='baseline', enable_heads_dict=None):
    """
    Run one configuration of the experiment

    Args:
        args: Command line arguments
        mode: 'baseline' or 'all_disabled' or 'partial_restore'
        enable_heads_dict: Dict mapping layer_id -> list of enabled head indices

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
        mp.enable_gpt2_custom_block(layers[layer_id], layer_id)

    # Register hooks for head disabling if not baseline
    hooks = []
    if mode != 'baseline':
        print(f"\nRegistering hooks for mode: {mode}")
        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            num_heads = model.config.n_head

            if mode == 'all_disabled':
                # Disable all heads
                hook = HeadDisableHook(layer_id, num_heads, mode='disable_all')
                print(f"  Layer {layer_id}: Disabling all {num_heads} heads")
            elif mode == 'partial_restore' and enable_heads_dict:
                # Disable all except specified heads
                enable_heads = enable_heads_dict.get(layer_id, [])
                hook = HeadDisableHook(layer_id, num_heads, mode='disable_except', enable_heads=enable_heads)
                print(f"  Layer {layer_id}: Enabling heads {enable_heads}, disabling others")

            handle = layer.attn.register_forward_hook(hook)
            hooks.append(handle)

    # Load data
    print("\nLoading dataset...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Storage for analysis
    n_layers = len(layers)
    layer_stats = {}

    for layer_id in range(n_layers):
        layer_stats[layer_id] = {
            'top1': [],
            'top2': [],
            'top3': [],
            'median': [],
            'dim138': [],  # Track dimension 138 specifically
            'dim447': []   # Track dimension 447 specifically
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

                # Get features: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
                feat_abs = layer.feat.abs()

                # Flatten to [total_tokens, hidden_dim]
                if len(feat_abs.shape) == 3:
                    feat_abs = feat_abs.view(-1, feat_abs.shape[-1])

                # Get top-k values
                sorted_vals, _ = torch.sort(feat_abs.flatten(), descending=True)

                layer_stats[layer_id]['top1'].append(sorted_vals[0].item())
                layer_stats[layer_id]['top2'].append(sorted_vals[1].item())
                layer_stats[layer_id]['top3'].append(sorted_vals[2].item())
                layer_stats[layer_id]['median'].append(torch.median(feat_abs).item())

                # Track specific dimensions (dim 138 and 447)
                if feat_abs.shape[1] > 447:
                    layer_stats[layer_id]['dim138'].append(torch.max(feat_abs[:, 138]).item())
                    layer_stats[layer_id]['dim447'].append(torch.max(feat_abs[:, 447]).item())

    # Clean up hooks
    for handle in hooks:
        handle.remove()

    # Compute statistics
    print("\nComputing statistics...")
    results = {}

    for layer_id in range(n_layers):
        results[layer_id] = {
            'top1_mean': np.mean(layer_stats[layer_id]['top1']),
            'top1_std': np.std(layer_stats[layer_id]['top1']),
            'top2_mean': np.mean(layer_stats[layer_id]['top2']),
            'top3_mean': np.mean(layer_stats[layer_id]['top3']),
            'median_mean': np.mean(layer_stats[layer_id]['median']),
            'dim138_mean': np.mean(layer_stats[layer_id]['dim138']) if layer_stats[layer_id]['dim138'] else 0,
            'dim447_mean': np.mean(layer_stats[layer_id]['dim447']) if layer_stats[layer_id]['dim447'] else 0,
            'dim138_max': np.max(layer_stats[layer_id]['dim138']) if layer_stats[layer_id]['dim138'] else 0,
            'dim447_max': np.max(layer_stats[layer_id]['dim447']) if layer_stats[layer_id]['dim447'] else 0,
        }

        # Print summary for key layers
        if layer_id in [0, 2, 5, 10, 11]:
            print(f"\nLayer {layer_id}:")
            print(f"  Top1: {results[layer_id]['top1_mean']:.2f} ± {results[layer_id]['top1_std']:.2f}")
            print(f"  Median: {results[layer_id]['median_mean']:.2f}")
            print(f"  Dim 138: {results[layer_id]['dim138_mean']:.2f} (max: {results[layer_id]['dim138_max']:.2f})")
            print(f"  Dim 447: {results[layer_id]['dim447_mean']:.2f} (max: {results[layer_id]['dim447_max']:.2f})")

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
    ax.bar(x - width/2, baseline_top1, width, label='Baseline (No Pruning)', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, disabled_top1, width, label='All Heads Disabled', alpha=0.8, color='#A23B72')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top1 Activation Value', fontsize=12, fontweight='bold')
    ax.set_title('Top1 Activation: Baseline vs All Heads Disabled', fontsize=14, fontweight='bold')
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

    # Plot 3: Dimension 447 (Most important)
    ax = axes[1, 0]
    baseline_447 = [baseline_results[i]['dim447_mean'] for i in layers]
    disabled_447 = [disabled_results[i]['dim447_mean'] for i in layers]
    ax.plot(layers, baseline_447, 'o-', linewidth=3, markersize=8, label='Baseline', color='#2E86AB')
    ax.plot(layers, disabled_447, 's-', linewidth=3, markersize=8, label='All Heads Disabled', color='#A23B72')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 447 Activation', fontsize=12, fontweight='bold')
    ax.set_title('Dimension 447: Primary Massive Activation Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    # Plot 4: Dimension 138
    ax = axes[1, 1]
    baseline_138 = [baseline_results[i]['dim138_mean'] for i in layers]
    disabled_138 = [disabled_results[i]['dim138_mean'] for i in layers]
    ax.plot(layers, baseline_138, 'o-', linewidth=3, markersize=8, label='Baseline', color='#2E86AB')
    ax.plot(layers, disabled_138, 's-', linewidth=3, markersize=8, label='All Heads Disabled', color='#A23B72')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 138 Activation', fontsize=12, fontweight='bold')
    ax.set_title('Dimension 138: Secondary Massive Activation Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp1_top1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp1_top1_comparison.png")

    # ===== Figure 2: Percentage Change Heatmap =====
    fig, ax = plt.subplots(figsize=(16, 6))

    metrics = ['top1_mean', 'top2_mean', 'top3_mean', 'median_mean', 'dim138_mean', 'dim447_mean']
    metric_labels = ['Top 1', 'Top 2', 'Top 3', 'Median', 'Dim 138', 'Dim 447']

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
    ax.set_title('Percentage Change: (All Heads Disabled - Baseline) / Baseline × 100%',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp1_percentage_change_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp1_percentage_change_heatmap.png")

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
        ax.bar(x + width/2, disabled_vals, width, label='All Heads Disabled', alpha=0.8, color='#A23B72')

        ax.set_ylabel('Activation Value', fontsize=10, fontweight='bold')
        ax.set_title(f'Layer {layer_id}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Layer-wise Detailed Comparison: Key Layers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp1_layerwise_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp1_layerwise_breakdown.png")

    # ===== Figure 4: Critical Dimensions Focus =====
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # Dimension 447 with error bars
    ax = axes[0]
    baseline_447_mean = [baseline_results[i]['dim447_mean'] for i in layers]
    disabled_447_mean = [disabled_results[i]['dim447_mean'] for i in layers]
    baseline_447_max = [baseline_results[i]['dim447_max'] for i in layers]
    disabled_447_max = [disabled_results[i]['dim447_max'] for i in layers]

    ax.plot(layers, baseline_447_mean, 'o-', linewidth=3, markersize=10,
            label='Baseline (Mean)', color='#2E86AB')
    ax.plot(layers, disabled_447_mean, 's-', linewidth=3, markersize=10,
            label='All Heads Disabled (Mean)', color='#A23B72')
    ax.fill_between(layers, baseline_447_mean, baseline_447_max, alpha=0.2, color='#2E86AB')
    ax.fill_between(layers, disabled_447_mean, disabled_447_max, alpha=0.2, color='#A23B72')

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation Value (Dimension 447)', fontsize=12, fontweight='bold')
    ax.set_title('🔥 Dimension 447: Most Critical Massive Activation Dimension',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    # Dimension 138 with error bars
    ax = axes[1]
    baseline_138_mean = [baseline_results[i]['dim138_mean'] for i in layers]
    disabled_138_mean = [disabled_results[i]['dim138_mean'] for i in layers]
    baseline_138_max = [baseline_results[i]['dim138_max'] for i in layers]
    disabled_138_max = [disabled_results[i]['dim138_max'] for i in layers]

    ax.plot(layers, baseline_138_mean, 'o-', linewidth=3, markersize=10,
            label='Baseline (Mean)', color='#2E86AB')
    ax.plot(layers, disabled_138_mean, 's-', linewidth=3, markersize=10,
            label='All Heads Disabled (Mean)', color='#A23B72')
    ax.fill_between(layers, baseline_138_mean, baseline_138_max, alpha=0.2, color='#2E86AB')
    ax.fill_between(layers, disabled_138_mean, disabled_138_max, alpha=0.2, color='#A23B72')

    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Activation Value (Dimension 138)', fontsize=12, fontweight='bold')
    ax.set_title('⭐ Dimension 138: Secondary Massive Activation Dimension',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'exp1_critical_dimensions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Generated: exp1_critical_dimensions.png")

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
    report_lines.append("EXPERIMENT 1: FEASIBILITY TEST - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("RESEARCH QUESTION:")
    report_lines.append("  Do attention heads generate massive activations?")
    report_lines.append("")
    report_lines.append("METHODOLOGY:")
    report_lines.append("  1. Baseline: Run model with all attention heads active")
    report_lines.append("  2. All Disabled: Run model with all 144 attention heads zeroed out")
    report_lines.append("  3. Compare massive activation magnitudes")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")

    # Calculate key metrics
    n_layers = len(baseline_results)

    # Dimension 447 (most important)
    dim447_baseline_peak = max([baseline_results[i]['dim447_mean'] for i in range(n_layers)])
    dim447_disabled_peak = max([disabled_results[i]['dim447_mean'] for i in range(n_layers)])
    dim447_peak_layer = max(range(n_layers), key=lambda i: baseline_results[i]['dim447_mean'])
    dim447_change_pct = 100 * (dim447_disabled_peak - dim447_baseline_peak) / dim447_baseline_peak if dim447_baseline_peak > 0 else 0

    # Dimension 138 (secondary)
    dim138_baseline_peak = max([baseline_results[i]['dim138_mean'] for i in range(n_layers)])
    dim138_disabled_peak = max([disabled_results[i]['dim138_mean'] for i in range(n_layers)])
    dim138_peak_layer = max(range(n_layers), key=lambda i: baseline_results[i]['dim138_mean'])
    dim138_change_pct = 100 * (dim138_disabled_peak - dim138_baseline_peak) / dim138_baseline_peak if dim138_baseline_peak > 0 else 0

    # Top1 overall
    top1_baseline_peak = max([baseline_results[i]['top1_mean'] for i in range(n_layers)])
    top1_disabled_peak = max([disabled_results[i]['top1_mean'] for i in range(n_layers)])
    top1_peak_layer = max(range(n_layers), key=lambda i: baseline_results[i]['top1_mean'])
    top1_change_pct = 100 * (top1_disabled_peak - top1_baseline_peak) / top1_baseline_peak if top1_baseline_peak > 0 else 0

    report_lines.append("1️⃣ DIMENSION 447 (Primary Massive Activation Dimension)")
    report_lines.append("-" * 70)
    report_lines.append(f"  Baseline Peak Value:       {dim447_baseline_peak:.2f} (Layer {dim447_peak_layer})")
    report_lines.append(f"  All Heads Disabled Value:  {dim447_disabled_peak:.2f}")
    report_lines.append(f"  Absolute Change:           {dim447_disabled_peak - dim447_baseline_peak:.2f}")
    report_lines.append(f"  Percentage Change:         {dim447_change_pct:.2f}%")
    report_lines.append("")

    if abs(dim447_change_pct) < 5:
        conclusion_447 = "✅ MINIMAL IMPACT - Massive activations persist without heads"
    elif dim447_change_pct < -20:
        conclusion_447 = "⚠️ SIGNIFICANT DECREASE - Heads may contribute to generation"
    else:
        conclusion_447 = "🔍 MODERATE CHANGE - Partial contribution from heads"

    report_lines.append(f"  Conclusion: {conclusion_447}")
    report_lines.append("")

    report_lines.append("2️⃣ DIMENSION 138 (Secondary Massive Activation Dimension)")
    report_lines.append("-" * 70)
    report_lines.append(f"  Baseline Peak Value:       {dim138_baseline_peak:.2f} (Layer {dim138_peak_layer})")
    report_lines.append(f"  All Heads Disabled Value:  {dim138_disabled_peak:.2f}")
    report_lines.append(f"  Absolute Change:           {dim138_disabled_peak - dim138_baseline_peak:.2f}")
    report_lines.append(f"  Percentage Change:         {dim138_change_pct:.2f}%")
    report_lines.append("")

    if abs(dim138_change_pct) < 5:
        conclusion_138 = "✅ MINIMAL IMPACT - Massive activations persist without heads"
    elif dim138_change_pct < -20:
        conclusion_138 = "⚠️ SIGNIFICANT DECREASE - Heads may contribute to generation"
    else:
        conclusion_138 = "🔍 MODERATE CHANGE - Partial contribution from heads"

    report_lines.append(f"  Conclusion: {conclusion_138}")
    report_lines.append("")

    report_lines.append("3️⃣ OVERALL TOP1 ACTIVATION")
    report_lines.append("-" * 70)
    report_lines.append(f"  Baseline Peak Value:       {top1_baseline_peak:.2f} (Layer {top1_peak_layer})")
    report_lines.append(f"  All Heads Disabled Value:  {top1_disabled_peak:.2f}")
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
    avg_change = np.mean([abs(100 * (disabled_results[i]['top1_mean'] - baseline_results[i]['top1_mean']) / baseline_results[i]['top1_mean'])
                          for i in range(n_layers) if baseline_results[i]['top1_mean'] > 0])

    if avg_change < 5:
        overall_conclusion = """
✅ ATTENTION HEADS DO NOT GENERATE MASSIVE ACTIVATIONS

The experiment shows that disabling all 144 attention heads has minimal impact
(<5% average change) on massive activation magnitudes. This strongly suggests:

  1. Massive activations originate from MLP layers, not attention mechanisms
  2. The attention heads we previously identified are "READERS" not "GENERATORS"
  3. They attend to massive activations but do not create them

NEXT STEPS:
  → Skip single-head restoration experiments (would not be informative)
  → Focus investigation on MLP layer contributions
  → Analyze layer normalization and residual connection effects
"""
    elif avg_change > 20:
        overall_conclusion = """
⚠️ ATTENTION HEADS SIGNIFICANTLY CONTRIBUTE TO MASSIVE ACTIVATIONS

The experiment shows that disabling all attention heads reduces massive
activations by >20% on average. This suggests:

  1. Attention heads DO participate in generating massive activations
  2. Further experiments needed to identify which specific heads are responsible
  3. Possible multi-head cooperation effects

NEXT STEPS:
  → Proceed to Experiment 2: Single-layer restoration
  → Identify critical layers
  → Then proceed to Experiment 3: Single-head restoration within critical layers
"""
    else:
        overall_conclusion = """
🔍 MIXED RESULTS - PARTIAL CONTRIBUTION FROM ATTENTION HEADS

The experiment shows moderate changes (5-20%) when disabling attention heads.
This suggests:

  1. Attention heads have partial contribution to massive activations
  2. MLP layers likely also play a significant role
  3. Complex interaction between attention and MLP components

NEXT STEPS:
  → Conduct layer-by-layer analysis to identify contribution sources
  → Investigate MLP layer effects separately
  → Consider interaction effects between components
"""

    report_lines.append(overall_conclusion)
    report_lines.append("")
    report_lines.append("="*80)

    # Save report
    report_path = os.path.join(savedir, 'comparison', 'EXPERIMENT_1_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Also print to console
    print('\n'.join(report_lines))

    print(f"\n✅ Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Test if attention heads generate massive activations'
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
    parser.add_argument('--savedir', type=str, default='results/exp1_feasibility_test/',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create directory structure
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'baseline'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'all_heads_disabled'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'comparison'), exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENT 1: FEASIBILITY TEST")
    print("="*80)
    print("\nResearch Question:")
    print("  Do attention heads generate massive activations?")
    print("\nMethod:")
    print("  1. Run baseline (no pruning)")
    print("  2. Run with all 144 attention heads disabled")
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

    # Run all heads disabled
    print("\n🔴 PHASE 2: Running All Heads Disabled Experiment")
    disabled_results, disabled_stats = run_experiment(args, mode='all_disabled')

    # Save disabled results
    with open(os.path.join(args.savedir, 'all_heads_disabled', 'results.json'), 'w') as f:
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
    print("✅ EXPERIMENT 1 COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("\nGenerated files:")
    print("  📁 baseline/")
    print("     └─ results.json")
    print("  📁 all_heads_disabled/")
    print("     └─ results.json")
    print("  📁 comparison/")
    print("     ├─ exp1_top1_comparison.png")
    print("     ├─ exp1_percentage_change_heatmap.png")
    print("     ├─ exp1_layerwise_breakdown.png")
    print("     ├─ exp1_critical_dimensions.png")
    print("     └─ EXPERIMENT_1_SUMMARY.txt")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
