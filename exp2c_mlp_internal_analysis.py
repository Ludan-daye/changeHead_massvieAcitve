#!/usr/bin/env python3
"""
Experiment 2C: MLP Internal Analysis - Tracking Massive Activation Generation

This experiment tracks Layer 2 MLP's internal processing to identify EXACTLY where
massive activations are generated:

GPT-2 MLP Structure:
  Input (768-dim)
    ↓
  Linear1 (c_fc): 768 → 3072
    ↓
  GELU activation
    ↓
  Linear2 (c_proj): 3072 → 768
    ↓
  Output (768-dim)

We track 4 checkpoints to pinpoint the generation mechanism.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp


class MLPInternalTracker:
    """
    Hook to track MLP internal activations at 4 checkpoints
    """
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.checkpoints = {
            'mlp_input': [],
            'after_linear1': [],
            'after_gelu': [],
            'mlp_output': []
        }

    def track_input(self, module, input, output):
        """Track MLP input"""
        mlp_input = input[0].detach().cpu().double()
        self.checkpoints['mlp_input'].append(mlp_input)

    def track_linear1(self, module, input, output):
        """Track after Linear1 (before GELU)"""
        after_linear1 = output.detach().cpu().double()
        self.checkpoints['after_linear1'].append(after_linear1)

    def track_gelu(self, module, input, output):
        """Track after GELU"""
        after_gelu = output.detach().cpu().double()
        self.checkpoints['after_gelu'].append(after_gelu)

    def track_output(self, module, input, output):
        """Track MLP output (after Linear2)"""
        mlp_output = output.detach().cpu().double()
        self.checkpoints['mlp_output'].append(mlp_output)


def run_internal_tracking(args):
    """
    Run MLP internal tracking experiment
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2C: MLP INTERNAL ANALYSIS - LAYER {args.layer_id}")
    print(f"{'='*80}\n")

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture for target layer
    target_layer = layers[args.layer_id]
    mp.enable_gpt2_custom_block(target_layer, args.layer_id)

    # Create tracker
    tracker = MLPInternalTracker(args.layer_id)

    # Register hooks at 4 checkpoints
    print(f"Registering hooks for Layer {args.layer_id} MLP internal tracking...")

    # Checkpoint 1: MLP Input
    handle1 = target_layer.mlp.register_forward_hook(tracker.track_input)

    # Checkpoint 2: After Linear1 (c_fc)
    handle2 = target_layer.mlp.c_fc.register_forward_hook(tracker.track_linear1)

    # Checkpoint 3: After GELU (act)
    handle3 = target_layer.mlp.act.register_forward_hook(tracker.track_gelu)

    # Checkpoint 4: MLP Output (after c_proj)
    handle4 = target_layer.mlp.c_proj.register_forward_hook(tracker.track_output)

    print("✓ Checkpoint 1: MLP Input")
    print("✓ Checkpoint 2: After Linear1 (768 → 3072)")
    print("✓ Checkpoint 3: After GELU")
    print("✓ Checkpoint 4: After Linear2 (MLP Output)")

    # Load data
    print("\nLoading dataset...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    print(f"\nProcessing {len(testseq_list)} samples...")

    # Process samples
    with torch.no_grad():
        for idx, testseq in enumerate(tqdm(testseq_list, desc="Tracking MLP internals")):
            # Forward pass
            _ = model(testseq)

    # Clean up hooks
    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()

    print("\n" + "="*80)
    print("ANALYZING CHECKPOINTS")
    print("="*80)

    # Analyze each checkpoint
    results = analyze_checkpoints(tracker, args)

    # Analyze weights
    weight_analysis = analyze_weights(target_layer, args)

    # Generate visualizations
    generate_visualizations(results, weight_analysis, args)

    # Generate report
    generate_report(results, weight_analysis, args)

    return results, weight_analysis


def analyze_checkpoints(tracker, args):
    """
    Analyze activations at each checkpoint
    """
    results = {}

    checkpoint_names = [
        ('mlp_input', 'MLP Input', 768),
        ('after_linear1', 'After Linear1', 3072),
        ('after_gelu', 'After GELU', 3072),
        ('mlp_output', 'MLP Output', 768)
    ]

    for cp_key, cp_name, expected_dim in checkpoint_names:
        print(f"\n{'─'*60}")
        print(f"Checkpoint: {cp_name}")
        print(f"{'─'*60}")

        data_list = tracker.checkpoints[cp_key]

        if not data_list:
            print("⚠️  No data captured!")
            continue

        # Concatenate all samples
        all_data = torch.cat(data_list, dim=0)  # [total_samples, seq_len, dim]

        # Flatten to [total_tokens, dim]
        if len(all_data.shape) == 3:
            all_data_flat = all_data.view(-1, all_data.shape[-1])
        else:
            all_data_flat = all_data

        # Get absolute values
        all_data_abs = all_data_flat.abs()

        # Compute statistics
        top1_val = torch.max(all_data_abs).item()
        top10_vals = torch.topk(all_data_abs.flatten(), k=10).values.numpy()
        median_val = torch.median(all_data_abs).item()
        mean_val = torch.mean(all_data_abs).item()

        # Top dimensions (for 3072-dim intermediate representations)
        if all_data_abs.shape[1] > 768:
            # For 3072-dim: find which dimensions are largest
            max_per_dim = torch.max(all_data_abs, dim=0).values  # [3072]
            top_dims_indices = torch.topk(max_per_dim, k=20).indices.numpy()
            top_dims_values = torch.topk(max_per_dim, k=20).values.numpy()
        else:
            # For 768-dim: check dim 447 and 138 specifically
            if all_data_abs.shape[1] > 447:
                dim447_max = torch.max(all_data_abs[:, 447]).item()
                dim138_max = torch.max(all_data_abs[:, 138]).item()
            else:
                dim447_max = 0
                dim138_max = 0
            top_dims_indices = None
            top_dims_values = None

        results[cp_key] = {
            'name': cp_name,
            'dim': all_data_abs.shape[1],
            'top1': top1_val,
            'top10': top10_vals.tolist(),
            'median': median_val,
            'mean': mean_val,
            'ratio': top1_val / median_val if median_val > 0 else 0,
            'top_dims_indices': top_dims_indices.tolist() if top_dims_indices is not None else None,
            'top_dims_values': top_dims_values.tolist() if top_dims_values is not None else None,
            'dim447_max': dim447_max if 'dim447_max' in locals() else None,
            'dim138_max': dim138_max if 'dim138_max' in locals() else None,
        }

        print(f"Dimension: {all_data_abs.shape[1]}")
        print(f"Top 1: {top1_val:.2f}")
        print(f"Median: {median_val:.2f}")
        print(f"Top1/Median: {top1_val/median_val if median_val > 0 else 0:.2f}×")

        if cp_key == 'mlp_output':
            print(f"Dim 447 max: {dim447_max:.2f}")
            print(f"Dim 138 max: {dim138_max:.2f}")

        if top_dims_indices is not None:
            print(f"\nTop 10 dimensions (out of {all_data_abs.shape[1]}):")
            for i in range(10):
                print(f"  Dim {top_dims_indices[i]:4d}: {top_dims_values[i]:.2f}")

    return results


def analyze_weights(layer, args):
    """
    Analyze MLP weight matrices
    """
    print(f"\n{'='*80}")
    print("ANALYZING WEIGHT MATRICES")
    print(f"{'='*80}")

    # Linear1 (c_fc): [3072, 768]
    W1 = layer.mlp.c_fc.weight.detach().cpu().numpy()  # [3072, 768]
    b1 = layer.mlp.c_fc.bias.detach().cpu().numpy() if layer.mlp.c_fc.bias is not None else None

    # Linear2 (c_proj): [768, 3072]
    W2 = layer.mlp.c_proj.weight.detach().cpu().numpy()  # [768, 3072]
    b2 = layer.mlp.c_proj.bias.detach().cpu().numpy() if layer.mlp.c_proj.bias is not None else None

    print(f"\nLinear1 (c_fc) weight shape: {W1.shape}")
    print(f"Linear2 (c_proj) weight shape: {W2.shape}")

    # Analyze Linear2 weights for dim 447 and 138
    w_dim447 = W2[447, :]  # [3072] - weights that produce dim 447
    w_dim138 = W2[138, :]  # [3072] - weights that produce dim 138

    # Find which intermediate dimensions contribute most to dim 447
    top_k = 20
    top_contributors_447 = np.argsort(np.abs(w_dim447))[::-1][:top_k]
    top_weights_447 = w_dim447[top_contributors_447]

    top_contributors_138 = np.argsort(np.abs(w_dim138))[::-1][:top_k]
    top_weights_138 = w_dim138[top_contributors_138]

    print(f"\nTop 10 intermediate dims contributing to Dim 447:")
    for i in range(10):
        dim_idx = top_contributors_447[i]
        weight = top_weights_447[i]
        print(f"  Intermediate dim {dim_idx:4d}: weight = {weight:+.4f}")

    print(f"\nTop 10 intermediate dims contributing to Dim 138:")
    for i in range(10):
        dim_idx = top_contributors_138[i]
        weight = top_weights_138[i]
        print(f"  Intermediate dim {dim_idx:4d}: weight = {weight:+.4f}")

    # Overall weight statistics
    W1_max = np.max(np.abs(W1))
    W1_mean = np.mean(np.abs(W1))
    W2_max = np.max(np.abs(W2))
    W2_mean = np.mean(np.abs(W2))

    print(f"\nWeight Statistics:")
    print(f"  Linear1: max = {W1_max:.4f}, mean = {W1_mean:.4f}")
    print(f"  Linear2: max = {W2_max:.4f}, mean = {W2_mean:.4f}")

    return {
        'W1_shape': W1.shape,
        'W2_shape': W2.shape,
        'W1_max': float(W1_max),
        'W1_mean': float(W1_mean),
        'W2_max': float(W2_max),
        'W2_mean': float(W2_mean),
        'top_contributors_447': top_contributors_447.tolist(),
        'top_weights_447': top_weights_447.tolist(),
        'top_contributors_138': top_contributors_138.tolist(),
        'top_weights_138': top_weights_138.tolist(),
        'w_dim447_full': w_dim447.tolist(),
        'w_dim138_full': w_dim138.tolist(),
    }


def generate_visualizations(results, weight_analysis, args):
    """
    Generate comprehensive visualizations
    """
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    # ===== Figure 1: Activation Flow Through MLP =====
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    checkpoint_keys = ['mlp_input', 'after_linear1', 'after_gelu', 'mlp_output']
    checkpoint_names = ['MLP Input\n(768-dim)', 'After Linear1\n(3072-dim)', 'After GELU\n(3072-dim)', 'MLP Output\n(768-dim)']

    # Plot 1: Top1 values progression
    ax1 = fig.add_subplot(gs[0, :])
    top1_vals = [results[k]['top1'] for k in checkpoint_keys]
    colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E']

    bars = ax1.bar(range(4), top1_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(checkpoint_names, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Maximum Activation Value', fontsize=14, fontweight='bold')
    ax1.set_title(f'Layer {args.layer_id} MLP: Activation Flow Through 4 Checkpoints',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Annotate bars with values
    for i, (bar, val) in enumerate(zip(bars, top1_vals)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add arrows between stages
    for i in range(3):
        ax1.annotate('', xy=(i+1, top1_vals[i+1]/2), xytext=(i, top1_vals[i]/2),
                    arrowprops=dict(arrowstyle='->', lw=3, color='gray', alpha=0.5))

    # Plot 2: Top1/Median ratio
    ax2 = fig.add_subplot(gs[1, 0])
    ratios = [results[k]['ratio'] for k in checkpoint_keys]
    ax2.plot(range(4), ratios, 'o-', linewidth=3, markersize=12, color='#C73E1D')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['Input', 'Linear1', 'GELU', 'Output'], rotation=45)
    ax2.set_ylabel('Top1 / Median Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Activation Distribution Skewness', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for i, ratio in enumerate(ratios):
        ax2.text(i, ratio, f'{ratio:.1f}×', ha='center', va='bottom', fontsize=10)

    # Plot 3: Median values
    ax3 = fig.add_subplot(gs[1, 1])
    medians = [results[k]['median'] for k in checkpoint_keys]
    ax3.bar(range(4), medians, color=colors, alpha=0.6)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(['Input', 'Linear1', 'GELU', 'Output'], rotation=45)
    ax3.set_ylabel('Median Activation', fontsize=12, fontweight='bold')
    ax3.set_title('Median Activation Values', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Top 10 values comparison
    ax4 = fig.add_subplot(gs[2, :])
    for i, (key, name) in enumerate(zip(checkpoint_keys, checkpoint_names)):
        top10 = results[key]['top10']
        ax4.plot(range(1, 11), top10, 'o-', linewidth=2, markersize=8,
                label=name.replace('\n', ' '), color=colors[i])

    ax4.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Activation Value', fontsize=12, fontweight='bold')
    ax4.set_title('Top 10 Activation Values at Each Checkpoint', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 11))

    plt.savefig(os.path.join(savedir, 'exp2c_activation_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: exp2c_activation_flow.png")

    # ===== Figure 2: Dimension Analysis for Intermediate Layers =====
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Top dimensions after Linear1
    ax = axes[0, 0]
    if results['after_linear1']['top_dims_indices']:
        top_dims = results['after_linear1']['top_dims_indices'][:15]
        top_vals = results['after_linear1']['top_dims_values'][:15]
        ax.barh(range(len(top_dims)), top_vals, color='#F18F01', alpha=0.8)
        ax.set_yticks(range(len(top_dims)))
        ax.set_yticklabels([f'Dim {d}' for d in top_dims], fontsize=9)
        ax.set_xlabel('Max Activation Value', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Dimensions After Linear1 (3072-dim)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    # Top dimensions after GELU
    ax = axes[0, 1]
    if results['after_gelu']['top_dims_indices']:
        top_dims = results['after_gelu']['top_dims_indices'][:15]
        top_vals = results['after_gelu']['top_dims_values'][:15]
        ax.barh(range(len(top_dims)), top_vals, color='#C73E1D', alpha=0.8)
        ax.set_yticks(range(len(top_dims)))
        ax.set_yticklabels([f'Dim {d}' for d in top_dims], fontsize=9)
        ax.set_xlabel('Max Activation Value', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Dimensions After GELU (3072-dim)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    # Weight contribution to Dim 447
    ax = axes[1, 0]
    top_contributors = weight_analysis['top_contributors_447'][:15]
    top_weights = weight_analysis['top_weights_447'][:15]
    colors_weights = ['green' if w > 0 else 'red' for w in top_weights]
    ax.barh(range(len(top_contributors)), [abs(w) for w in top_weights],
           color=colors_weights, alpha=0.7)
    ax.set_yticks(range(len(top_contributors)))
    ax.set_yticklabels([f'Dim {d}' for d in top_contributors], fontsize=9)
    ax.set_xlabel('|Weight| Value', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 Weight Contributors to Dim 447\n(Green=Positive, Red=Negative)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Weight contribution to Dim 138
    ax = axes[1, 1]
    top_contributors = weight_analysis['top_contributors_138'][:15]
    top_weights = weight_analysis['top_weights_138'][:15]
    colors_weights = ['green' if w > 0 else 'red' for w in top_weights]
    ax.barh(range(len(top_contributors)), [abs(w) for w in top_weights],
           color=colors_weights, alpha=0.7)
    ax.set_yticks(range(len(top_contributors)))
    ax.set_yticklabels([f'Dim {d}' for d in top_contributors], fontsize=9)
    ax.set_xlabel('|Weight| Value', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 Weight Contributors to Dim 138\n(Green=Positive, Red=Negative)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'exp2c_dimension_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: exp2c_dimension_analysis.png")

    # ===== Figure 3: Critical Analysis - GELU's Role =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Compare before and after GELU
    ax = axes[0, 0]
    stages = ['Before GELU\n(Linear1)', 'After GELU']
    before_gelu = results['after_linear1']['top1']
    after_gelu = results['after_gelu']['top1']
    values = [before_gelu, after_gelu]
    change_pct = 100 * (after_gelu - before_gelu) / before_gelu if before_gelu > 0 else 0

    bars = ax.bar([0, 1], values, color=['#F18F01', '#C73E1D'], alpha=0.8, width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(stages, fontsize=12, fontweight='bold')
    ax.set_ylabel('Maximum Activation', fontsize=12, fontweight='bold')
    ax.set_title(f'GELU Impact: {change_pct:+.1f}% Change', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # GELU amplification per dimension
    ax = axes[0, 1]
    if results['after_linear1']['top_dims_indices']:
        linear1_dims = results['after_linear1']['top_dims_indices'][:10]
        linear1_vals = results['after_linear1']['top_dims_values'][:10]
        gelu_dims = results['after_gelu']['top_dims_indices'][:10]
        gelu_vals = results['after_gelu']['top_dims_values'][:10]

        # For common dimensions, show amplification
        x = np.arange(10)
        width = 0.35
        ax.bar(x - width/2, linear1_vals, width, label='Before GELU', color='#F18F01', alpha=0.7)
        ax.bar(x + width/2, gelu_vals, width, label='After GELU', color='#C73E1D', alpha=0.7)
        ax.set_xlabel('Top Dimension Rank', fontsize=11, fontweight='bold')
        ax.set_ylabel('Activation Value', fontsize=11, fontweight='bold')
        ax.set_title('Top 10 Dimensions: Before vs After GELU', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Full pipeline summary
    ax = axes[1, :]
    ax = plt.subplot(2, 1, 2)

    stages_full = ['Input', 'Linear1', 'GELU', 'Output']
    top1_progression = [results[k]['top1'] for k in checkpoint_keys]

    ax.plot(stages_full, top1_progression, 'o-', linewidth=4, markersize=15,
           color='#C73E1D', label='Max Activation')

    # Highlight the explosion point
    explosion_idx = np.argmax(top1_progression)
    ax.scatter([explosion_idx], [top1_progression[explosion_idx]],
              s=500, color='red', marker='*', zorder=10,
              label=f'Explosion Point: {stages_full[explosion_idx]}')

    ax.set_xlabel('MLP Processing Stage', fontsize=14, fontweight='bold')
    ax.set_ylabel('Maximum Activation Value', fontsize=14, fontweight='bold')
    ax.set_title('🔥 MASSIVE ACTIVATION GENERATION POINT 🔥', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotate percentage changes
    for i in range(len(stages_full) - 1):
        val1, val2 = top1_progression[i], top1_progression[i+1]
        change = 100 * (val2 - val1) / val1 if val1 > 0 else 0
        mid_x = i + 0.5
        mid_y = (val1 + val2) / 2
        ax.annotate(f'{change:+.1f}%', xy=(mid_x, mid_y), fontsize=11,
                   ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'exp2c_gelu_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: exp2c_gelu_impact.png")

    print("\n✅ All visualizations generated!")


def generate_report(results, weight_analysis, args):
    """
    Generate comprehensive text report
    """
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*80}")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"EXPERIMENT 2C: MLP INTERNAL ANALYSIS - LAYER {args.layer_id}")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("RESEARCH QUESTION:")
    report_lines.append("  Where exactly in the MLP do massive activations originate?")
    report_lines.append("")
    report_lines.append("METHODOLOGY:")
    report_lines.append(f"  Track Layer {args.layer_id} MLP at 4 checkpoints:")
    report_lines.append("    1. MLP Input (768-dim)")
    report_lines.append("    2. After Linear1 (3072-dim)")
    report_lines.append("    3. After GELU (3072-dim)")
    report_lines.append("    4. MLP Output (768-dim)")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)

    checkpoint_keys = ['mlp_input', 'after_linear1', 'after_gelu', 'mlp_output']

    # Find explosion point
    top1_vals = [results[k]['top1'] for k in checkpoint_keys]
    explosion_idx = np.argmax(top1_vals)
    explosion_stage = ['Input', 'Linear1', 'GELU', 'Output'][explosion_idx]

    report_lines.append("")
    report_lines.append(f"🔥 EXPLOSION POINT: {explosion_stage}")
    report_lines.append("")

    # Checkpoint-by-checkpoint analysis
    for i, key in enumerate(checkpoint_keys):
        stage_name = ['MLP Input', 'After Linear1', 'After GELU', 'MLP Output'][i]
        report_lines.append(f"\n{'─'*70}")
        report_lines.append(f"Checkpoint {i+1}: {stage_name}")
        report_lines.append(f"{'─'*70}")
        report_lines.append(f"  Dimensions:       {results[key]['dim']}")
        report_lines.append(f"  Max activation:   {results[key]['top1']:.2f}")
        report_lines.append(f"  Median:           {results[key]['median']:.2f}")
        report_lines.append(f"  Top1/Median:      {results[key]['ratio']:.2f}×")

        if key == 'mlp_output':
            report_lines.append(f"  Dim 447 max:      {results[key]['dim447_max']:.2f}")
            report_lines.append(f"  Dim 138 max:      {results[key]['dim138_max']:.2f}")

        if i > 0:
            prev_key = checkpoint_keys[i-1]
            change = results[key]['top1'] - results[prev_key]['top1']
            change_pct = 100 * change / results[prev_key]['top1'] if results[prev_key]['top1'] > 0 else 0
            report_lines.append(f"  Change from prev: {change:+.2f} ({change_pct:+.1f}%)")

    # GELU analysis
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("GELU IMPACT ANALYSIS")
    report_lines.append("="*80)

    before_gelu = results['after_linear1']['top1']
    after_gelu = results['after_gelu']['top1']
    gelu_change = after_gelu - before_gelu
    gelu_change_pct = 100 * gelu_change / before_gelu if before_gelu > 0 else 0

    report_lines.append(f"\nBefore GELU: {before_gelu:.2f}")
    report_lines.append(f"After GELU:  {after_gelu:.2f}")
    report_lines.append(f"Change:      {gelu_change:+.2f} ({gelu_change_pct:+.1f}%)")

    if gelu_change_pct > 50:
        conclusion_gelu = "✅ GELU SIGNIFICANTLY AMPLIFIES activations!"
    elif gelu_change_pct > 10:
        conclusion_gelu = "⚠️ GELU moderately amplifies activations"
    elif gelu_change_pct < -10:
        conclusion_gelu = "❌ GELU SUPPRESSES activations (unexpected)"
    else:
        conclusion_gelu = "➖ GELU has minimal impact"

    report_lines.append(f"\n{conclusion_gelu}")

    # Weight analysis
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("WEIGHT MATRIX ANALYSIS")
    report_lines.append("="*80)
    report_lines.append(f"\nLinear1 weights: max={weight_analysis['W1_max']:.4f}, mean={weight_analysis['W1_mean']:.4f}")
    report_lines.append(f"Linear2 weights: max={weight_analysis['W2_max']:.4f}, mean={weight_analysis['W2_mean']:.4f}")

    report_lines.append(f"\nTop 5 intermediate dimensions contributing to Dim 447:")
    for i in range(5):
        dim_idx = weight_analysis['top_contributors_447'][i]
        weight = weight_analysis['top_weights_447'][i]
        report_lines.append(f"  Intermediate dim {dim_idx:4d}: weight = {weight:+.4f}")

    # Overall conclusion
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("OVERALL CONCLUSION")
    report_lines.append("="*80)

    if explosion_idx == 2:  # After GELU
        overall_conclusion = f"""
✅ MASSIVE ACTIVATIONS ARE GENERATED AFTER GELU!

The experiment definitively shows:
  1. MLP Input: Low activations ({results['mlp_input']['top1']:.2f})
  2. After Linear1: Medium activations ({results['after_linear1']['top1']:.2f})
  3. After GELU: 🔥 EXPLOSION to {results['after_gelu']['top1']:.2f} ({gelu_change_pct:+.1f}%)
  4. MLP Output: Dim 447 reaches {results['mlp_output']['dim447_max']:.2f}

MECHANISM IDENTIFIED:
  → Linear1 (768→3072) creates intermediate activations
  → GELU non-linearity AMPLIFIES large values while suppressing small ones
  → Certain intermediate dimensions explode after GELU
  → Linear2 maps these explosive dimensions to output dim 447

This is the "perfect storm" for massive activations:
  - Wide intermediate layer (3072 dims) creates opportunity
  - GELU's non-linear amplification effect
  - Specific weight patterns in Linear2 concentrate the effect
"""
    elif explosion_idx == 1:  # After Linear1
        overall_conclusion = f"""
✅ MASSIVE ACTIVATIONS ARE GENERATED BY LINEAR1!

The experiment shows:
  1. MLP Input: Low ({results['mlp_input']['top1']:.2f})
  2. After Linear1: 🔥 EXPLOSION to {results['after_linear1']['top1']:.2f}
  3. After GELU: Maintained at {results['after_gelu']['top1']:.2f}
  4. Output: Dim 447 reaches {results['mlp_output']['dim447_max']:.2f}

MECHANISM: Linear1 weight matrix (768→3072) has certain weights that
produce extremely large intermediate activations even before GELU.
"""
    else:
        overall_conclusion = f"""
Explosion point: {explosion_stage}
Further investigation needed to understand the mechanism.
"""

    report_lines.append(overall_conclusion)
    report_lines.append("="*80)

    # Save report
    report_path = os.path.join(args.savedir, 'EXPERIMENT_2C_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Also print to console
    print('\n'.join(report_lines))

    # Save detailed JSON
    json_path = os.path.join(args.savedir, 'exp2c_detailed_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'results': {k: {kk: vv if not isinstance(vv, (np.ndarray, np.floating, np.integer))
                           else float(vv) if isinstance(vv, (np.floating, np.integer))
                           else vv.tolist() if isinstance(vv, np.ndarray)
                           else vv
                           for kk, vv in v.items()}
                       for k, v in results.items()},
            'weight_analysis': weight_analysis
        }, f, indent=2)

    print(f"\n✅ Summary report saved to: {report_path}")
    print(f"✅ Detailed results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2C: MLP internal analysis - track activation generation'
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--access_token', type=str, default='type in your access token here',
                        help='Hugging Face access token')

    # Experiment arguments
    parser.add_argument('--layer_id', type=int, default=2,
                        help='Which layer to analyze (default: 2, the explosion layer)')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'c4', 'RedPajama'], help='Dataset name')
    parser.add_argument('--nsamples', type=int, default=30,
                        help='Number of samples to analyze')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Output arguments
    parser.add_argument('--savedir', type=str, default='results/exp2c_mlp_internal/',
                        help='Directory to save results')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "="*80)
    print(f"EXPERIMENT 2C: MLP INTERNAL ANALYSIS - LAYER {args.layer_id}")
    print("="*80)
    print("\nResearch Question:")
    print("  Where exactly in the MLP are massive activations generated?")
    print("\nMethod:")
    print("  Track 4 checkpoints in Layer 2 MLP:")
    print("    1. MLP Input (768-dim)")
    print("    2. After Linear1: 768 → 3072")
    print("    3. After GELU activation")
    print("    4. After Linear2: 3072 → 768 (output)")
    print("\n" + "="*80)

    # Run experiment
    results, weight_analysis = run_internal_tracking(args)

    print("\n" + "="*80)
    print("✅ EXPERIMENT 2C COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("\nGenerated files:")
    print("  📊 exp2c_activation_flow.png - Activation progression through MLP")
    print("  📊 exp2c_dimension_analysis.png - Top dimensions and weight analysis")
    print("  📊 exp2c_gelu_impact.png - GELU's amplification effect")
    print("  📄 EXPERIMENT_2C_SUMMARY.txt - Detailed text report")
    print("  📄 exp2c_detailed_results.json - Full numerical results")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
