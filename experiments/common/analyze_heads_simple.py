"""
Simplified approach: Use hooks to capture attention weights directly.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import lib


def capture_attention_hook(name, storage_dict):
    """Create a hook to capture attention weights."""
    def hook(module, input, output):
        # GPT-2 attention output: (attn_output, present_key_value, attentions)
        # But attentions might not always be there
        # Let's capture from the internal computation
        pass
    return hook


def analyze_heads_simple(args):
    """Simplified head analysis using model's built-in attention output."""

    print("="*80)
    print("Analyzing Attention Heads (Simplified Approach)")
    print("="*80)

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"Loaded model: {args.model}")
    print(f"Number of layers: {len(layers)}")

    # Prepare data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)
    print(f"Loaded {len(testseq_list)} test sequences")

    # Storage for results
    all_layer_head_scores = {}

    # Process samples
    for sample_idx, test_seq in enumerate(testseq_list[:args.nsamples]):
        if sample_idx % 5 == 0:
            print(f"\nProcessing sample {sample_idx}/{args.nsamples}")

        with torch.no_grad():
            # Get model outputs with attentions
            outputs = model(test_seq, output_attentions=True, use_cache=False)

        # outputs.attentions: tuple of (batch_size, num_heads, seq_len, seq_len) for each layer
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            print("WARNING: Model did not return attentions")
            continue

        # For each layer
        for layer_id, attn_weights in enumerate(outputs.attentions):
            if attn_weights is None:
                continue

            # attn_weights: [batch, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len_q, seq_len_kv = attn_weights.shape

            # Get the layer's output features to find massive activations
            # We need to run again with our custom hooks...
            # For now, let's use a different approach: analyze attention to first token
            # (which often has massive activations)

            if layer_id not in all_layer_head_scores:
                all_layer_head_scores[layer_id] = []

            # Compute attention to first token (BOS/first position)
            # This is often where massive activations appear
            attn_to_first = attn_weights[:, :, :, 0].mean(dim=(0, 2))  # [num_heads]

            all_layer_head_scores[layer_id].append(attn_to_first.cpu().numpy())

    # Average scores across samples
    results = {}
    for layer_id, scores_list in all_layer_head_scores.items():
        avg_scores = np.mean(scores_list, axis=0)
        results[layer_id] = {
            'head_scores': avg_scores,
            'num_heads': len(avg_scores)
        }
        print(f"\nLayer {layer_id}:")
        print(f"  Avg attention to first token by head: {avg_scores}")
        print(f"  Top 3 heads: {np.argsort(avg_scores)[-3:][::-1]}")

    return results


def visualize_results(results, args):
    """Visualize which heads focus most on first token (massive activation location)."""

    num_layers = len(results)
    if num_layers == 0:
        print("No results to visualize")
        return

    # Create heatmap
    num_heads = results[0]['num_heads']
    heatmap_data = np.zeros((num_layers, num_heads))

    for layer_id, layer_data in results.items():
        heatmap_data[layer_id] = layer_data['head_scores']

    # Plot
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'L{i}' for i in range(num_layers)],
                cbar_kws={'label': 'Attention to First Token'})
    plt.xlabel('Attention Head', fontsize=13, fontweight='bold')
    plt.ylabel('Layer', fontsize=13, fontweight='bold')
    plt.title(f'{args.model.upper()}: Attention Head Focus on First Token\\n(Typical Massive Activation Location)',
              fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = os.path.join(args.savedir, f'{args.model}_head_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

    # Create ranking plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: All layers
    for layer_id, layer_data in results.items():
        scores = layer_data['head_scores']
        ax1.plot(range(num_heads), scores, marker='o', label=f'Layer {layer_id}', alpha=0.7)

    ax1.set_xlabel('Head Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention to First Token', fontsize=12, fontweight='bold')
    ax1.set_title('Head Attention Patterns by Layer', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Average across all layers
    all_scores = np.array([layer_data['head_scores'] for layer_data in results.values()])
    avg_scores_per_head = all_scores.mean(axis=0)
    std_scores_per_head = all_scores.std(axis=0)

    ax2.bar(range(num_heads), avg_scores_per_head, yerr=std_scores_per_head,
            capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Head Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Attention (±std)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Head Importance Across All Layers', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(num_heads))

    plt.tight_layout()
    rank_path = os.path.join(args.savedir, f'{args.model}_head_ranking.png')
    plt.savefig(rank_path, dpi=300, bbox_inches='tight')
    print(f"Ranking plot saved to: {rank_path}")
    plt.close()


def save_pruning_config(results, args):
    """Save pruning configuration."""

    config_path = os.path.join(args.savedir, f'{args.model}_pruning_config.txt')

    with open(config_path, 'w') as f:
        f.write(f"# Head Analysis Configuration for {args.model}\n")
        f.write(f"# Based on attention to first token (typical massive activation location)\n")
        f.write(f"# Generated from {args.nsamples} samples\n\n")

        for layer_id, layer_data in results.items():
            scores = layer_data['head_scores']
            sorted_indices = np.argsort(scores)[::-1]

            f.write(f"\nLayer {layer_id}:\n")
            f.write(f"  Total heads: {len(scores)}\n")
            f.write(f"  Top 3 heads with highest attention to first token:\n")
            for rank, head_idx in enumerate(sorted_indices[:3]):
                f.write(f"    Rank {rank+1}: Head {head_idx} (score: {scores[head_idx]:.4f})\n")

            f.write(f"  Bottom 3 heads (lowest attention):\n")
            for rank, head_idx in enumerate(sorted_indices[-3:][::-1]):
                f.write(f"    Rank {rank+1}: Head {head_idx} (score: {scores[head_idx]:.4f})\n")

    print(f"Pruning configuration saved to: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--nsamples', type=int, default=50, help='Number of samples')
    parser.add_argument('--savedir', type=str, default='results/head_analysis/',
                       help='Directory to save results')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--revision', type=str, default='main')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # Run analysis
    results = analyze_heads_simple(args)

    # Visualize
    visualize_results(results, args)

    # Save config
    save_pruning_config(results, args)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
