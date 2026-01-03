#!/usr/bin/env python3
"""
Experiment 1: Phase 2 Only - Run only the second phase (All Heads Disabled)
Reuse existing baseline results and run only the second phase to save time
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp
from lib.model_utils import is_llama_model


class HeadDisableHook:
    """Hook to disable all attention heads"""
    def __init__(self, layer_id, num_heads):
        self.layer_id = layer_id
        self.num_heads = num_heads

    def __call__(self, module, input, output):
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
        attn_output_reshaped[:, :, :, :] = 0  # Disable all heads
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


def run_phase2(args):
    """Run only the second phase: All Heads Disabled"""
    print(f"\n{'='*80}")
    print(f"Running Phase 2 Only: ALL HEADS DISABLED")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture
    print("Enabling feature capture for all layers...")
    for layer_id in range(len(layers)):
        if is_llama_model(args.model):
            mp.enable_llama_custom_decoderlayer(layers[layer_id], layer_id)
        elif "opt" in args.model:
            mp.enable_opt_custom_decoderlayer(layers[layer_id], layer_id)
        elif "gpt2" in args.model:
            mp.enable_gpt2_custom_block(layers[layer_id], layer_id)

    # Register hooks to disable all heads
    print("\nRegistering hooks to disable all attention heads...")
    hooks = []
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

        hook = HeadDisableHook(layer_id, num_heads)
        handle = target_module.register_forward_hook(hook)
        hooks.append(handle)
        if layer_id % 5 == 0:
            print(f"  Layer {layer_id}: Disabling all {num_heads} heads")

    # Load data
    print("\nLoading dataset...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Storage for analysis
    n_layers = len(layers)
    layer_stats = {}
    for layer_id in range(n_layers):
        layer_stats[layer_id] = {
            'top1': [], 'top2': [], 'top3': [],
            'median': [], 'dim138': [], 'dim447': []
        }

    print(f"\nProcessing {len(testseq_list)} samples...")

    # Process samples
    with torch.no_grad():
        for idx, testseq in enumerate(tqdm(testseq_list, desc="Phase 2")):
            _ = model(testseq)

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

                if feat_abs.shape[1] > 447:
                    layer_stats[layer_id]['dim138'].append(torch.max(feat_abs[:, 138]).item())
                    layer_stats[layer_id]['dim447'].append(torch.max(feat_abs[:, 447]).item())

    # Clean up
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

    return results, layer_stats


from exp1_feasibility_test import generate_visualizations, generate_summary_report


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Phase 2 Only')
    parser.add_argument('--model', type=str, default='llama2_13b')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='results/exp1_llama2_13b/')

    args = parser.parse_args()

    # Check baseline exists
    baseline_path = os.path.join(args.savedir, 'baseline', 'results.json')
    if not os.path.exists(baseline_path):
        print(f"❌ ERROR: Baseline not found at {baseline_path}")
        print("Please run full experiment first or check path.")
        return

    print("\n" + "="*80)
    print("EXPERIMENT 1: PHASE 2 ONLY (All Heads Disabled)")
    print("="*80)
    print(f"\n✓ Using existing baseline from: {baseline_path}")
    
    # Load baseline
    print("\n📂 Loading baseline results...")
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    baseline_results = {int(k): v for k, v in baseline_results.items()}
    print(f"✅ Loaded baseline for {len(baseline_results)} layers")

    # Run Phase 2
    print("\n🔴 PHASE 2: Running All Heads Disabled")
    disabled_results, disabled_stats = run_phase2(args)

    # Save Phase 2 results
    print("\n💾 Saving Phase 2 results...")
    os.makedirs(os.path.join(args.savedir, 'all_heads_disabled'), exist_ok=True)
    with open(os.path.join(args.savedir, 'all_heads_disabled', 'results.json'), 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                   for k, v in disabled_results.items()}, f, indent=2)
    print("✅ Phase 2 results saved")

    # Generate visualizations
    print("\n🎨 PHASE 3: Generating Visualizations")
    os.makedirs(os.path.join(args.savedir, 'comparison'), exist_ok=True)
    generate_visualizations(baseline_results, disabled_results, None, disabled_stats, args.savedir)

    # Generate summary
    print("\n📊 PHASE 4: Generating Summary Report")
    generate_summary_report(baseline_results, disabled_results, args.savedir)

    print("\n" + "="*80)
    print("✅ EXPERIMENT 1 COMPLETE (Phase 2-4)")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
