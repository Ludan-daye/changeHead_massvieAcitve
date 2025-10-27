"""
Prune or replace attention heads based on their relationship to massive activations.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

import lib
import monkey_patch as mp


class HeadPruningHook:
    """Hook to prune specific attention heads during forward pass."""

    def __init__(self, heads_to_prune, num_heads, strategy='zero'):
        """
        Args:
            heads_to_prune: list of head indices to prune
            num_heads: total number of heads
            strategy: 'zero' (set to zero) or 'random' (replace with random head)
        """
        self.heads_to_prune = set(heads_to_prune)
        self.num_heads = num_heads
        self.strategy = strategy
        print(f"Initialized pruning hook: {len(heads_to_prune)} heads to prune")
        print(f"Heads to prune: {sorted(heads_to_prune)}")

    def __call__(self, module, input, output):
        """Apply pruning during forward pass."""
        # Output format varies by model
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads

        # Reshape to separate heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)

        # Apply pruning strategy
        for head_idx in self.heads_to_prune:
            if self.strategy == 'zero':
                # Set head output to zero
                attn_output_reshaped[:, :, head_idx, :] = 0
            elif self.strategy == 'random':
                # Replace with random values (maintaining similar scale)
                scale = attn_output_reshaped[:, :, head_idx, :].std()
                attn_output_reshaped[:, :, head_idx, :] = torch.randn_like(
                    attn_output_reshaped[:, :, head_idx, :]
                ) * scale
            elif self.strategy == 'mean':
                # Replace with mean across other heads
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


def load_pruning_config(config_path):
    """Load pruning configuration from file."""
    pruning_config = {}

    with open(config_path, 'r') as f:
        current_layer = None
        for line in f:
            line = line.strip()
            if line.startswith('Layer '):
                current_layer = int(line.split()[1].rstrip(':'))
                pruning_config[current_layer] = {'top_heads': [], 'bottom_heads': []}
            elif 'Head' in line and current_layer is not None:
                # Parse head index
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'Head':
                        head_idx = int(parts[i+1])
                        if 'Top' in line or 'most related' in line:
                            pruning_config[current_layer]['top_heads'].append(head_idx)
                        elif 'Bottom' in line or 'least related' in line:
                            pruning_config[current_layer]['bottom_heads'].append(head_idx)

    return pruning_config


def apply_head_pruning(model, layers, pruning_config, args):
    """Apply head pruning hooks to the model."""

    hooks = []

    for layer_id, config in pruning_config.items():
        if layer_id >= len(layers):
            continue

        layer = layers[layer_id]

        # Determine which heads to prune based on strategy
        if args.prune_strategy == 'top':
            heads_to_prune = config['top_heads'][:args.num_heads_to_prune]
            print(f"Layer {layer_id}: Pruning TOP heads (most related to massive activations)")
        elif args.prune_strategy == 'bottom':
            heads_to_prune = config['bottom_heads'][:args.num_heads_to_prune]
            print(f"Layer {layer_id}: Pruning BOTTOM heads (least related to massive activations)")
        elif args.prune_strategy == 'random':
            all_heads = list(range(12))  # Assuming 12 heads for GPT-2
            heads_to_prune = np.random.choice(all_heads, args.num_heads_to_prune, replace=False).tolist()
            print(f"Layer {layer_id}: Pruning RANDOM heads")
        else:
            print(f"Unknown prune_strategy: {args.prune_strategy}")
            continue

        # Get number of heads
        if hasattr(layer, 'self_attn'):
            num_heads = layer.self_attn.num_heads
            attn_module = layer.self_attn
        elif hasattr(layer, 'attn'):
            num_heads = layer.attn.num_heads
            attn_module = layer.attn
        else:
            print(f"Could not find attention module in layer {layer_id}")
            continue

        # Create and register hook
        pruning_hook = HeadPruningHook(
            heads_to_prune=heads_to_prune,
            num_heads=num_heads,
            strategy=args.head_replacement
        )

        handle = attn_module.register_forward_hook(pruning_hook)
        hooks.append(handle)

    return hooks


def evaluate_with_pruning(args):
    """Main function to evaluate model with head pruning."""

    print("="*80)
    print("Attention Head Pruning Experiment")
    print("="*80)

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"Loaded model: {args.model}")

    # Load pruning configuration
    if args.pruning_config:
        print(f"\nLoading pruning configuration from: {args.pruning_config}")
        pruning_config = load_pruning_config(args.pruning_config)
        print(f"Loaded config for {len(pruning_config)} layers")
    else:
        print("No pruning config provided, using default")
        pruning_config = {}

    # Baseline evaluation (no pruning)
    print("\n" + "="*60)
    print("BASELINE EVALUATION (No Pruning)")
    print("="*60)
    baseline_results = {}
    for ds_name in args.datasets:
        ppl = lib.eval_ppl(ds_name, model, tokenizer, args.seed, device)
        baseline_results[ds_name] = ppl
        print(f"{ds_name} PPL: {ppl:.4f}")

    # Apply pruning
    print("\n" + "="*60)
    print(f"PRUNED EVALUATION (Strategy: {args.prune_strategy}, Heads: {args.num_heads_to_prune})")
    print("="*60)

    hooks = apply_head_pruning(model, layers, pruning_config, args)

    # Evaluate with pruning
    pruned_results = {}
    for ds_name in args.datasets:
        ppl = lib.eval_ppl(ds_name, model, tokenizer, args.seed, device)
        pruned_results[ds_name] = ppl
        print(f"{ds_name} PPL: {ppl:.4f}")

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Dataset':<15} {'Baseline':<12} {'Pruned':<12} {'Change':<12} {'% Change'}")
    print("-" * 65)

    for ds_name in args.datasets:
        baseline = baseline_results[ds_name]
        pruned = pruned_results[ds_name]
        change = pruned - baseline
        pct_change = (change / baseline) * 100
        print(f"{ds_name:<15} {baseline:<12.4f} {pruned:<12.4f} {change:<12.4f} {pct_change:+.2f}%")

    # Save results
    save_path = os.path.join(args.savedir, f'{args.model}_pruning_results.txt')
    with open(save_path, 'w') as f:
        f.write(f"Pruning Strategy: {args.prune_strategy}\n")
        f.write(f"Heads to Prune: {args.num_heads_to_prune}\n")
        f.write(f"Replacement Strategy: {args.head_replacement}\n\n")
        f.write(f"{'Dataset':<15} {'Baseline':<12} {'Pruned':<12} {'Change':<12} {'% Change'}\n")
        f.write("-" * 65 + "\n")
        for ds_name in args.datasets:
            baseline = baseline_results[ds_name]
            pruned = pruned_results[ds_name]
            change = pruned - baseline
            pct_change = (change / baseline) * 100
            f.write(f"{ds_name:<15} {baseline:<12.4f} {pruned:<12.4f} {change:<12.4f} {pct_change:+.2f}%\n")

    print(f"\nResults saved to: {save_path}")

    # Cleanup hooks
    for handle in hooks:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--pruning_config', type=str, help='Path to pruning config file')
    parser.add_argument('--prune_strategy', type=str, default='top',
                       choices=['top', 'bottom', 'random'],
                       help='Which heads to prune: top (most related), bottom (least), or random')
    parser.add_argument('--num_heads_to_prune', type=int, default=2,
                       help='Number of heads to prune per layer')
    parser.add_argument('--head_replacement', type=str, default='zero',
                       choices=['zero', 'mean', 'random'],
                       help='How to replace pruned heads')
    parser.add_argument('--datasets', type=str, nargs='+', default=['wikitext'],
                       help='Datasets to evaluate on')
    parser.add_argument('--savedir', type=str, default='results/pruning/',
                       help='Directory to save results')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--revision', type=str, default='main')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    evaluate_with_pruning(args)
