"""
Test how head pruning affects massive activations.
Compare massive activation magnitudes before and after pruning different heads.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

import lib
import monkey_patch as mp


class HeadPruningHook:
    """Hook to prune specific attention heads."""

    def __init__(self, heads_to_prune, num_heads, strategy='zero'):
        self.heads_to_prune = set(heads_to_prune)
        self.num_heads = num_heads
        self.strategy = strategy

    def __call__(self, module, input, output):
        """Apply pruning during forward pass."""
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads

        # Reshape to separate heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)

        # Apply pruning
        for head_idx in self.heads_to_prune:
            if self.strategy == 'zero':
                attn_output_reshaped[:, :, head_idx, :] = 0
            elif self.strategy == 'mean':
                other_heads = [h for h in range(self.num_heads) if h not in self.heads_to_prune]
                if other_heads:
                    mean_output = attn_output_reshaped[:, :, other_heads, :].mean(dim=2, keepdim=True)
                    attn_output_reshaped[:, :, head_idx, :] = mean_output.squeeze(2)

        # Reshape back
        attn_output_pruned = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)

        if isinstance(output, tuple):
            return (attn_output_pruned,) + output[1:]
        else:
            return attn_output_pruned


def measure_massive_activations(model, layers, tokenizer, device, seq_len, nsamples=10):
    """Measure massive activation statistics."""

    # Enable feature capture for all layers
    for layer_id, layer in enumerate(layers):
        if hasattr(layer, 'attn'):  # GPT-2 style
            mp.enable_gpt2_custom_block(layer, layer_id)

    # Get test data
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seq_len, device=device)

    # Collect statistics
    layer_stats = []

    for sample_idx, test_seq in enumerate(testseq_list[:nsamples]):
        if sample_idx % 5 == 0:
            print(f"  Processing sample {sample_idx}/{nsamples}")

        with torch.no_grad():
            model(test_seq)

        sample_stats = []
        for layer_id, layer in enumerate(layers):
            if hasattr(layer, 'feat'):
                feat_abs = layer.feat.abs()

                # Get top 3 activation values
                flat_feat = feat_abs.flatten()
                top_values, _ = torch.topk(flat_feat, k=3)
                median_value = torch.median(flat_feat)

                sample_stats.append({
                    'top1': top_values[0].item(),
                    'top2': top_values[1].item(),
                    'top3': top_values[2].item(),
                    'median': median_value.item(),
                })
            else:
                sample_stats.append({
                    'top1': 0, 'top2': 0, 'top3': 0, 'median': 0
                })

        layer_stats.append(sample_stats)

    # Average across samples
    num_layers = len(layers)
    avg_stats = []
    for layer_id in range(num_layers):
        layer_data = [s[layer_id] for s in layer_stats]
        avg_stats.append({
            'top1': np.mean([d['top1'] for d in layer_data]),
            'top2': np.mean([d['top2'] for d in layer_data]),
            'top3': np.mean([d['top3'] for d in layer_data]),
            'median': np.mean([d['median'] for d in layer_data]),
        })

    return avg_stats


def run_experiment(args, heads_to_prune_config, experiment_name):
    """Run one pruning experiment."""

    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)

    # Apply pruning hooks
    hooks = []
    for layer_id, heads in heads_to_prune_config.items():
        if layer_id >= len(layers):
            continue

        layer = layers[layer_id]

        if hasattr(layer, 'attn'):  # GPT-2
            num_heads = layer.attn.num_heads
            attn_module = layer.attn
        else:
            print(f"  Warning: Could not find attention in layer {layer_id}")
            continue

        if len(heads) > 0:
            print(f"  Layer {layer_id}: Pruning heads {heads}")
            pruning_hook = HeadPruningHook(
                heads_to_prune=heads,
                num_heads=num_heads,
                strategy=args.prune_method
            )
            handle = attn_module.register_forward_hook(pruning_hook)
            hooks.append(handle)

    # Measure massive activations
    print(f"\nMeasuring massive activations...")
    stats = measure_massive_activations(
        model, layers, tokenizer, device, seq_len,
        nsamples=args.nsamples
    )

    # Cleanup hooks
    for handle in hooks:
        handle.remove()

    return stats


def compare_results(baseline_stats, pruned_stats, experiment_name, args):
    """Compare baseline vs pruned statistics."""

    num_layers = len(baseline_stats)

    # Extract data for plotting
    baseline_top1 = np.array([s['top1'] for s in baseline_stats])
    pruned_top1 = np.array([s['top1'] for s in pruned_stats])

    baseline_median = np.array([s['median'] for s in baseline_stats])
    pruned_median = np.array([s['median'] for s in pruned_stats])

    # Calculate change
    top1_change = ((pruned_top1 - baseline_top1) / (baseline_top1 + 1e-8)) * 100
    median_change = ((pruned_median - baseline_median) / (baseline_median + 1e-8)) * 100

    # Print summary
    print(f"\n{experiment_name} Results:")
    print(f"{'Layer':<8} {'Baseline Top1':<15} {'Pruned Top1':<15} {'Change %':<12}")
    print("-" * 60)
    for i in range(num_layers):
        print(f"{i:<8} {baseline_top1[i]:<15.2f} {pruned_top1[i]:<15.2f} {top1_change[i]:<12.2f}%")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Top1 activations
    ax = axes[0, 0]
    x = np.arange(num_layers)
    width = 0.35
    ax.bar(x - width/2, baseline_top1, width, label='Baseline', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, pruned_top1, width, label='Pruned', alpha=0.8, color='coral')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 1 Activation Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Massive Activation Magnitude Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Top-right: Percent change
    ax = axes[0, 1]
    colors = ['red' if c < 0 else 'green' for c in top1_change]
    ax.bar(x, top1_change, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Change in Top1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Massive Activation Change After Pruning', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Median comparison
    ax = axes[1, 0]
    ax.plot(x, baseline_median, marker='o', label='Baseline', linewidth=2, markersize=8)
    ax.plot(x, pruned_median, marker='s', label='Pruned', linewidth=2, markersize=8)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Activation', fontsize=12, fontweight='bold')
    ax.set_title('Median Activation Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Ratio (Top1 / Median)
    ax = axes[1, 1]
    baseline_ratio = baseline_top1 / (baseline_median + 1e-8)
    pruned_ratio = pruned_top1 / (pruned_median + 1e-8)
    ax.bar(x - width/2, baseline_ratio, width, label='Baseline', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, pruned_ratio, width, label='Pruned', alpha=0.8, color='coral')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top1 / Median Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Massive Activation Concentration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = os.path.join(args.savedir, f'{experiment_name}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()

    return {
        'baseline_top1': baseline_top1,
        'pruned_top1': pruned_top1,
        'top1_change_pct': top1_change,
        'baseline_ratio': baseline_ratio,
        'pruned_ratio': pruned_ratio,
    }


def main(args):
    """Main experiment runner."""

    print("="*80)
    print("Testing Head Pruning Effects on Massive Activations")
    print("="*80)

    # Experiment 1: Baseline (no pruning)
    print("\n" + "="*80)
    print("BASELINE: No Pruning")
    print("="*80)

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    baseline_stats = measure_massive_activations(
        model, layers, tokenizer, device, seq_len,
        nsamples=args.nsamples
    )

    del model
    torch.cuda.empty_cache()

    # Experiment 2: Prune TOP heads (most related to massive activations)
    # Based on our analysis: Layer 5 Head 1, Layer 2 Head 7, Layer 6 Head 1
    top_heads_config = {
        2: [7],      # Layer 2, Head 7 (score: 0.568)
        5: [1],      # Layer 5, Head 1 (score: 0.828)
        6: [1],      # Layer 6, Head 1 (score: 0.796)
    }

    top_pruned_stats = run_experiment(args, top_heads_config, "Prune_TOP_Heads")
    results_top = compare_results(baseline_stats, top_pruned_stats, "Prune_TOP_Heads", args)

    torch.cuda.empty_cache()

    # Experiment 3: Prune BOTTOM heads (least related to massive activations)
    bottom_heads_config = {
        0: [1],      # Layer 0, Head 1 (score: 0.001)
        4: [11],     # Layer 4, Head 11 (score: 0.002)
        11: [8],     # Layer 11, Head 8 (score: 0.001)
    }

    bottom_pruned_stats = run_experiment(args, bottom_heads_config, "Prune_BOTTOM_Heads")
    results_bottom = compare_results(baseline_stats, bottom_pruned_stats, "Prune_BOTTOM_Heads", args)

    torch.cuda.empty_cache()

    # Create summary comparison
    create_summary_plot(results_top, results_bottom, args)

    # Save numerical results
    save_path = os.path.join(args.savedir, 'pruning_results_summary.txt')
    with open(save_path, 'w') as f:
        f.write("Head Pruning Effects on Massive Activations\n")
        f.write("="*80 + "\n\n")

        f.write("EXPERIMENT 1: Prune TOP heads (most related to massive activations)\n")
        f.write(f"Pruned heads: {top_heads_config}\n")
        f.write(f"Average Top1 change: {results_top['top1_change_pct'].mean():.2f}%\n")
        f.write(f"Max Top1 change: {results_top['top1_change_pct'].max():.2f}% (Layer {results_top['top1_change_pct'].argmax()})\n\n")

        f.write("EXPERIMENT 2: Prune BOTTOM heads (least related to massive activations)\n")
        f.write(f"Pruned heads: {bottom_heads_config}\n")
        f.write(f"Average Top1 change: {results_bottom['top1_change_pct'].mean():.2f}%\n")
        f.write(f"Max Top1 change: {results_bottom['top1_change_pct'].max():.2f}% (Layer {results_bottom['top1_change_pct'].argmax()})\n\n")

        f.write("\nConclusion:\n")
        if abs(results_top['top1_change_pct'].mean()) > abs(results_bottom['top1_change_pct'].mean()):
            f.write("✓ Pruning TOP heads has LARGER impact on massive activations\n")
            f.write("  → These heads are important for generating massive activations!\n")
        else:
            f.write("✗ Pruning TOP heads has SMALLER impact than expected\n")
            f.write("  → These heads may not be the primary cause of massive activations\n")

    print(f"\nResults summary saved to: {save_path}")
    print("\n" + "="*80)
    print("All experiments complete!")
    print("="*80)


def create_summary_plot(results_top, results_bottom, args):
    """Create a summary comparison plot."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    num_layers = len(results_top['top1_change_pct'])
    x = np.arange(num_layers)

    # Left: Change comparison
    ax1.plot(x, results_top['top1_change_pct'], marker='o', linewidth=2,
             markersize=8, label='Prune TOP Heads', color='red')
    ax1.plot(x, results_bottom['top1_change_pct'], marker='s', linewidth=2,
             markersize=8, label='Prune BOTTOM Heads', color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Change in Massive Activation (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Impact of Pruning Different Heads', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Average impact
    avg_impact = [
        abs(results_top['top1_change_pct']).mean(),
        abs(results_bottom['top1_change_pct']).mean()
    ]
    colors = ['red', 'blue']
    bars = ax2.bar(['Prune TOP\n(High Attention)', 'Prune BOTTOM\n(Low Attention)'],
                    avg_impact, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Average |Change| in Massive Activation (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Overall Impact Comparison', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(args.savedir, 'summary_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSummary plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--nsamples', type=int, default=20,
                       help='Number of samples for measuring massive activations')
    parser.add_argument('--prune_method', type=str, default='zero',
                       choices=['zero', 'mean'],
                       help='How to replace pruned heads')
    parser.add_argument('--savedir', type=str, default='results/head_pruning_massive/',
                       help='Directory to save results')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--revision', type=str, default='main')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    main(args)
