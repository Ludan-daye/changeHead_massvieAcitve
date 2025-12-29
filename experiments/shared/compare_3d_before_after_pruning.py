"""
Compare 3D feature visualization before and after pruning specific heads.
Generate side-by-side comparison for Layer 2.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import lib
import monkey_patch as mp


class HeadPruningHook:
    """Hook to prune specific attention heads."""

    def __init__(self, heads_to_prune, num_heads, strategy='zero'):
        self.heads_to_prune = set(heads_to_prune)
        self.num_heads = num_heads
        self.strategy = strategy

    def __call__(self, module, input, output):
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

        # Reshape back
        attn_output_pruned = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)

        if isinstance(output, tuple):
            return (attn_output_pruned,) + output[1:]
        else:
            return attn_output_pruned


def get_layer_features(model, layers, layer_id, tokenizer, device, text, pruned_heads=None):
    """Get features from a specific layer, optionally with pruned heads."""

    # Enable custom forward for the target layer
    if hasattr(layers[layer_id], 'attn'):  # GPT-2
        mp.enable_gpt2_custom_block(layers[layer_id], layer_id)

    # Apply pruning if specified
    hook_handle = None
    if pruned_heads is not None and len(pruned_heads) > 0:
        layer = layers[layer_id]
        if hasattr(layer, 'attn'):
            num_heads = layer.attn.num_heads
            pruning_hook = HeadPruningHook(
                heads_to_prune=pruned_heads,
                num_heads=num_heads,
                strategy='zero'
            )
            hook_handle = layer.attn.register_forward_hook(pruning_hook)

    # Run model
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    with torch.no_grad():
        model(inputs)

    # Get features
    feat = layers[layer_id].feat.abs()

    # Decode tokens
    seq_decoded = []
    for i in range(inputs.shape[1]):
        seq_decoded.append(tokenizer.decode(inputs[0, i].item()))

    # Cleanup hook
    if hook_handle is not None:
        hook_handle.remove()

    return feat, seq_decoded


def plot_3d_comparison(feat_before, feat_after, seq_decoded, layer_id, pruned_heads, savedir):
    """Create side-by-side 3D comparison plot."""

    fig = plt.figure(figsize=(20, 9))

    # Prepare data
    seq_len = feat_before.shape[1]
    hidden_dim = feat_before.shape[2]

    # Find top features to visualize (use baseline to identify them)
    feat_flat = feat_before.flatten()
    top_values, top_indices = torch.topk(feat_flat, k=10)

    # Get the feature dimensions that have massive activations
    top_dims = set()
    for idx in top_indices:
        dim = (idx.item() % hidden_dim)
        top_dims.add(dim)

    top_dims = sorted(list(top_dims))[:2]  # Keep top 2 dimensions

    print(f"Top feature dimensions: {top_dims}")

    # Plot 1: Before pruning
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_single(ax1, feat_before, seq_decoded, top_dims,
                   f'BEFORE Pruning\nLayer {layer_id}', 'steelblue')

    # Plot 2: After pruning
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_single(ax2, feat_after, seq_decoded, top_dims,
                   f'AFTER Pruning Head {pruned_heads}\nLayer {layer_id}', 'coral')

    # Overall title
    fig.suptitle(f'GPT-2 Layer {layer_id}: 3D Feature Comparison\nPruned Heads: {pruned_heads}',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(savedir, f'layer{layer_id}_3d_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n3D comparison saved to: {save_path}")
    plt.close()

    # Also create difference plot
    plot_difference(feat_before, feat_after, seq_decoded, top_dims, layer_id, pruned_heads, savedir)


def plot_3d_single(ax, feat, seq_decoded, feature_dims, title, color):
    """Plot a single 3D visualization."""

    seq_len = feat.shape[1]

    # For each feature dimension
    for dim_idx, dim in enumerate(feature_dims):
        values = feat[0, :, dim].cpu().numpy()

        # Create bars
        x_positions = np.arange(seq_len)
        y_positions = np.ones(seq_len) * dim
        z_base = np.zeros(seq_len)

        ax.bar3d(x_positions, y_positions, z_base,
                0.6, 0.6, values,
                color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Set labels
    ax.set_xlabel('Token', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Dim', fontsize=11, fontweight='bold')
    ax.set_zlabel('Magnitude', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Set token labels
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(seq_decoded, rotation=45, ha='right', fontsize=9)

    # Set feature dimension labels
    ax.set_yticks(feature_dims)
    ax.set_yticklabels([str(d) for d in feature_dims])

    # Set view angle
    ax.view_init(elev=20, azim=45)


def plot_difference(feat_before, feat_after, seq_decoded, feature_dims, layer_id, pruned_heads, savedir):
    """Plot the difference between before and after."""

    diff = feat_after - feat_before

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Heatmap of differences
    ax = axes[0]

    # Extract relevant dimensions
    diff_data = diff[0, :, feature_dims].cpu().numpy().T

    im = ax.imshow(diff_data, cmap='RdBu_r', aspect='auto', vmin=-1000, vmax=1000)
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax.set_title(f'Activation Difference (After - Before)\nLayer {layer_id}',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(seq_decoded)))
    ax.set_xticklabels(seq_decoded, rotation=45, ha='right')
    ax.set_yticks(range(len(feature_dims)))
    ax.set_yticklabels([str(d) for d in feature_dims])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Difference', fontsize=11)

    # Right: Bar plot of total change per feature
    ax = axes[1]

    total_change = diff[0, :, feature_dims].abs().sum(dim=0).cpu().numpy()

    bars = ax.bar(range(len(feature_dims)), total_change, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Absolute Change', fontsize=12, fontweight='bold')
    ax.set_title(f'Total Change in Massive Activation Dimensions\nAfter Pruning Head {pruned_heads}',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(feature_dims)))
    ax.set_xticklabels([str(d) for d in feature_dims])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, total_change)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(savedir, f'layer{layer_id}_difference_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Difference analysis saved to: {save_path}")
    plt.close()


def main(args):
    """Main function."""

    print("="*80)
    print("3D Feature Visualization: Before vs After Head Pruning")
    print("="*80)

    # Text to analyze
    text = "Summer is warm. Winter is cold."
    print(f"\nInput text: '{text}'")

    # Layer to analyze
    layer_id = args.layer_id
    print(f"Analyzing Layer: {layer_id}")

    # Heads to prune
    if args.layer_id == 2:
        pruned_heads = [7]  # Layer 2, Head 7 (highest attention to massive activations)
        print(f"Pruning heads: {pruned_heads} (Top head for this layer)")
    elif args.layer_id == 5:
        pruned_heads = [1]  # Layer 5, Head 1
        print(f"Pruning heads: {pruned_heads} (Highest scoring head overall)")
    else:
        pruned_heads = [0]  # Default
        print(f"Pruning heads: {pruned_heads}")

    # Load model (BEFORE)
    print("\n" + "="*80)
    print("STEP 1: Getting BASELINE features (no pruning)")
    print("="*80)

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    feat_before, seq_decoded = get_layer_features(
        model, layers, layer_id, tokenizer, device, text, pruned_heads=None
    )

    print(f"Baseline - Top activation: {feat_before.max().item():.2f}")
    print(f"Baseline - Median activation: {feat_before.median().item():.2f}")

    # Reload model and get AFTER features
    print("\n" + "="*80)
    print(f"STEP 2: Getting PRUNED features (pruning head {pruned_heads})")
    print("="*80)

    del model
    torch.cuda.empty_cache()

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    feat_after, _ = get_layer_features(
        model, layers, layer_id, tokenizer, device, text, pruned_heads=pruned_heads
    )

    print(f"After pruning - Top activation: {feat_after.max().item():.2f}")
    print(f"After pruning - Median activation: {feat_after.median().item():.2f}")

    # Calculate change
    change_pct = ((feat_after.max() - feat_before.max()) / feat_before.max()) * 100
    print(f"\nChange in max activation: {change_pct.item():.2f}%")

    # Plot comparison
    print("\n" + "="*80)
    print("STEP 3: Creating visualizations")
    print("="*80)

    plot_3d_comparison(feat_before, feat_after, seq_decoded, layer_id, pruned_heads, args.savedir)

    print("\n" + "="*80)
    print("Complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--layer_id', type=int, default=2,
                       help='Which layer to visualize (default: 2)')
    parser.add_argument('--savedir', type=str, default='results/3d_comparison/',
                       help='Directory to save results')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--revision', type=str, default='main')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    main(args)
