#!/usr/bin/env python3
"""
Experiment 1: Feasibility Test - OPTIMIZED for A100 GPU
Optimized version: Fully utilize A100 VRAM, use batch processing to accelerate computation
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
from lib.model_utils import is_llama_model


class HeadDisableHook:
    """Hook to disable all attention heads by zeroing their output"""
    def __init__(self, layer_id, num_heads, mode='disable_all', enable_heads=None):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.mode = mode
        self.enable_heads = enable_heads if enable_heads is not None else []

    def __call__(self, module, input, output):
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads

        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)

        if self.mode == 'disable_all':
            attn_output_reshaped[:, :, :, :] = 0
        elif self.mode == 'disable_except':
            for head_idx in range(self.num_heads):
                if head_idx not in self.enable_heads:
                    attn_output_reshaped[:, :, head_idx, :] = 0

        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


def get_data_batched(tokenizer, nsamples=50, seqlen=2048, batch_size=4, device=None):
    """
    Optimized data loading: supports batch processing
    """
    from datasets import load_dataset
    
    print(f"Loading dataset with batch_size={batch_size}...")
    valdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testenc = tokenizer("\n\n".join(valdata['text']), return_tensors='pt', add_special_tokens=False).input_ids
    
    testseq_list = []
    batch = []
    
    for i in range(nsamples):
        if (i + 1) * seqlen > testenc.shape[1]:
            break
        test_seq = testenc[:, (i * seqlen):((i+1) * seqlen)]
        batch.append(test_seq)
        
        # When reaching batch_size or the last sample, form a batch
        if len(batch) == batch_size or i == nsamples - 1:
            # Concatenate batch to [batch_size, seqlen]
            batched_seq = torch.cat(batch, dim=0).to(device)
            testseq_list.append(batched_seq)
            batch = []
    
    print(f"Created {len(testseq_list)} batches (batch_size={batch_size})")
    return testseq_list


def run_experiment_optimized(args, mode='baseline', enable_heads_dict=None):
    """
    Optimized experiment run function:
    1. Use batch processing
    2. Reduce CPU-GPU data transfers
    3. Optimize memory usage
    """
    print(f"\n{'='*80}")
    print(f"Running Experiment (OPTIMIZED): {mode.upper()}")
    print(f"{'='*80}\n")

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture for all layers
    print("Enabling feature capture for all layers...")
    for layer_id in range(len(layers)):
        if is_llama_model(args.model):
            mp.enable_llama_custom_decoderlayer(layers[layer_id], layer_id)
        elif "opt" in args.model:
            mp.enable_opt_custom_decoderlayer(layers[layer_id], layer_id)
        elif "gpt2" in args.model:
            mp.enable_gpt2_custom_block(layers[layer_id], layer_id)
        else:
            raise ValueError(f"Model {args.model} not supported")

    # Register hooks for head disabling if not baseline
    hooks = []
    if mode != 'baseline':
        print(f"\nRegistering hooks for mode: {mode}")
        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            
            if is_llama_model(args.model) or "opt" in args.model:
                num_heads = model.config.num_attention_heads
                target_module = layer.self_attn
            elif "gpt2" in args.model:
                num_heads = model.config.n_head
                target_module = layer.attn
            else:
                raise ValueError(f"Model {args.model} not supported")

            if mode == 'all_disabled':
                hook = HeadDisableHook(layer_id, num_heads, mode='disable_all')
                print(f"  Layer {layer_id}: Disabling all {num_heads} heads")
            elif mode == 'partial_restore' and enable_heads_dict:
                enable_heads = enable_heads_dict.get(layer_id, [])
                hook = HeadDisableHook(layer_id, num_heads, mode='disable_except', enable_heads=enable_heads)
                print(f"  Layer {layer_id}: Enabling heads {enable_heads}")

            handle = target_module.register_forward_hook(hook)
            hooks.append(handle)

    # Load data with batching - Optimization 1: Use batch processing
    print("\nLoading dataset with batching...")
    testseq_list = get_data_batched(
        tokenizer,
        nsamples=args.nsamples,
        seqlen=seq_len,
        batch_size=args.batch_size,  # New parameter
        device=device
    )

    # Storage for analysis
    n_layers = len(layers)
    layer_stats = {}

    for layer_id in range(n_layers):
        layer_stats[layer_id] = {
            'top1': [],
            'top2': [],
            'top3': [],
            'median': [],
            'dim138': [],
            'dim447': []
        }

    print(f"\nProcessing {len(testseq_list)} batches...")

    # Process samples - Optimization 2: Batch inference
    with torch.no_grad():
        for idx, testseq_batch in enumerate(tqdm(testseq_list, desc=f"Processing ({mode})")):
            # Forward pass - batch inference
            _ = model(testseq_batch)

            # Analyze each layer
            for layer_id in range(n_layers):
                layer = layers[layer_id]

                if not hasattr(layer, 'feat') or layer.feat is None:
                    continue

                # Get features: [batch, seq_len, hidden_dim]
                # Optimization 3: Compute directly on GPU, reduce CPU-GPU transfers
                feat = layer.feat
                if isinstance(feat, torch.Tensor) and feat.device.type == 'cpu':
                    # If already on CPU, move back to GPU for computation
                    feat = feat.to(device)
                
                feat_abs = feat.abs()

                # Flatten to [total_tokens, hidden_dim]
                if len(feat_abs.shape) == 3:
                    feat_abs = feat_abs.view(-1, feat_abs.shape[-1])

                # Get top-k values - compute on GPU
                sorted_vals, _ = torch.sort(feat_abs.flatten(), descending=True)

                # Only transfer to CPU and convert to Python values at the end
                layer_stats[layer_id]['top1'].append(sorted_vals[0].item())
                layer_stats[layer_id]['top2'].append(sorted_vals[1].item())
                layer_stats[layer_id]['top3'].append(sorted_vals[2].item())
                layer_stats[layer_id]['median'].append(torch.median(feat_abs).item())

                # Track specific dimensions
                if feat_abs.shape[1] > 447:
                    layer_stats[layer_id]['dim138'].append(torch.max(feat_abs[:, 138]).item())
                    layer_stats[layer_id]['dim447'].append(torch.max(feat_abs[:, 447]).item())

            # Optimization 4: Periodically clear GPU cache
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

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


# Import visualization and report generation functions from original version
from exp1_feasibility_test import generate_visualizations, generate_summary_report


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1: Optimized for A100 GPU'
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='llama2_13b', help='Model name')
    parser.add_argument('--access_token', type=str, default='type in your access token here',
                        help='Hugging Face access token')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'c4', 'RedPajama'], help='Dataset name')
    parser.add_argument('--nsamples', type=int, default=100,  # Increased to 100
                        help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=8,  # New: batch size
                        help='Batch size for processing (higher = faster but more memory)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Output arguments
    parser.add_argument('--savedir', type=str, default='results/exp1_llama2_13b_optimized/',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create directory structure
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'baseline'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'all_heads_disabled'), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'comparison'), exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENT 1: FEASIBILITY TEST (OPTIMIZED FOR A100)")
    print("="*80)
    print("\nOptimizations:")
    print(f"  ✓ Batch processing (batch_size={args.batch_size})")
    print(f"  ✓ Increased samples (nsamples={args.nsamples})")
    print("  ✓ Reduced CPU-GPU transfers")
    print("  ✓ Memory optimization")
    print("\nResearch Question:")
    print("  Do attention heads generate massive activations?")
    print("\nMethod:")
    print("  1. Run baseline (no pruning)")
    print("  2. Run with all attention heads disabled")
    print("  3. Compare activation magnitudes")
    print("\n" + "="*80)

    # Run baseline
    print("\n🔵 PHASE 1: Running Baseline Experiment (Optimized)")
    baseline_results, baseline_stats = run_experiment_optimized(args, mode='baseline')

    # Save baseline results
    with open(os.path.join(args.savedir, 'baseline', 'results.json'), 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                   for k, v in baseline_results.items()}, f, indent=2)

    # Run all heads disabled
    print("\n🔴 PHASE 2: Running All Heads Disabled Experiment (Optimized)")
    disabled_results, disabled_stats = run_experiment_optimized(args, mode='all_disabled')

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
    print("✅ EXPERIMENT 1 COMPLETE (OPTIMIZED)")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print(f"\nProcessed {args.nsamples} samples with batch_size={args.batch_size}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
