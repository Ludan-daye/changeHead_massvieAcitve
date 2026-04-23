#!/usr/bin/env python3
"""
RQ6: Single-Layer Activation - Which MLP layer(s) can independently generate MAs?

This experiment is the REVERSE of RQ2a: instead of disabling all MLPs to show MAs
disappear, we disable all MLPs EXCEPT one, and check if that single layer can
produce MAs on its own.

- If a single early layer (L_origin) alone recovers most of the MA → concentrated
  single-source generation
- If no single layer alone produces MA → dispersed/multi-source generation
- Combined with RQ2b (single-layer disable), this distinguishes:
    * Single-source with concentration   (disable 1 kills MA, enable 1 recovers)
    * Redundant multi-source (backup)    (disable 1 does nothing, enable 1 recovers)
    * Truly dispersed                    (neither disable nor enable of single is enough)

Output: per-layer top1 MA when only that layer's MLP is enabled.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib
import monkey_patch as mp


class MLPZeroHook:
    """Zero out the MLP output for layers that should be disabled."""
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


def run_single_layer_config(args, keep_layer_id, layers, model, tokenizer, device, seq_len):
    """Run forward pass with only `keep_layer_id`'s MLP enabled, all other MLPs zeroed.

    Returns: top1_mean (across sequences), peak top1_max.
    """
    # Register zero-hooks on all MLP modules except keep_layer_id
    hooks = []
    for lid in range(len(layers)):
        if lid == keep_layer_id:
            continue
        mlp = getattr(layers[lid], 'mlp', None)
        if mlp is None:
            continue
        h = mlp.register_forward_hook(MLPZeroHook())
        hooks.append(h)

    # Run forward passes and capture top1 MA
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=seq_len, device=device)
    top1_list = []
    with torch.no_grad():
        for seq in testseq_list:
            model(seq.to(device))
            # Get peak activation across all layers' block outputs
            peak = 0.0
            for lid in range(len(layers)):
                feat = getattr(layers[lid], 'feat', None)
                if feat is not None:
                    val = feat.detach().abs().max().item()
                    if val > peak:
                        peak = val
            top1_list.append(peak)

    # Cleanup
    for h in hooks:
        h.remove()

    return {
        'keep_layer': keep_layer_id,
        'top1_mean': float(np.mean(top1_list)),
        'top1_max': float(np.max(top1_list)),
        'top1_list': [float(x) for x in top1_list],
    }


def run_baseline(args, layers, model, tokenizer, device, seq_len):
    """Baseline: all MLPs enabled."""
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=seq_len, device=device)
    top1_list = []
    with torch.no_grad():
        for seq in testseq_list:
            model(seq.to(device))
            peak = 0.0
            for lid in range(len(layers)):
                feat = getattr(layers[lid], 'feat', None)
                if feat is not None:
                    val = feat.detach().abs().max().item()
                    if val > peak:
                        peak = val
            top1_list.append(peak)
    return {
        'top1_mean': float(np.mean(top1_list)),
        'top1_max': float(np.max(top1_list)),
    }


def run_all_disabled(args, layers, model, tokenizer, device, seq_len):
    """All MLPs disabled (reference floor)."""
    hooks = []
    for lid in range(len(layers)):
        mlp = getattr(layers[lid], 'mlp', None)
        if mlp is None:
            continue
        h = mlp.register_forward_hook(MLPZeroHook())
        hooks.append(h)

    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=seq_len, device=device)
    top1_list = []
    with torch.no_grad():
        for seq in testseq_list:
            model(seq.to(device))
            peak = 0.0
            for lid in range(len(layers)):
                feat = getattr(layers[lid], 'feat', None)
                if feat is not None:
                    val = feat.detach().abs().max().item()
                    if val > peak:
                        peak = val
            top1_list.append(peak)

    for h in hooks:
        h.remove()

    return {
        'top1_mean': float(np.mean(top1_list)),
        'top1_max': float(np.max(top1_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seqlen', type=int, default=1024)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--access_token', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--layers_to_scan', type=str, default='all',
                        help='Comma-separated layer ids or "all"')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"RQ6: SINGLE-LAYER ACTIVATION SCAN - {args.model}")
    print(f"{'='*80}\n")

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture on all layers
    for lid in range(len(layers)):
        lib.enable_custom_block(args.model, layers[lid], lid)

    n_layers = len(layers)
    print(f"Model has {n_layers} layers")

    # Determine which layers to scan
    if args.layers_to_scan == 'all':
        scan_layers = list(range(n_layers))
    else:
        scan_layers = [int(x.strip()) for x in args.layers_to_scan.split(',')]

    # 1. Baseline
    print("\n[1/3] Running baseline (all MLPs enabled)...")
    baseline = run_baseline(args, layers, model, tokenizer, device, seq_len)
    print(f"  Baseline Top1 MA: mean={baseline['top1_mean']:.1f}, max={baseline['top1_max']:.1f}")

    # 2. All disabled
    print("\n[2/3] Running all-MLPs-disabled (floor)...")
    floor = run_all_disabled(args, layers, model, tokenizer, device, seq_len)
    print(f"  All-disabled Top1 MA: mean={floor['top1_mean']:.1f}, max={floor['top1_max']:.1f}")

    # 3. Single-layer activation
    print(f"\n[3/3] Single-layer activation scan across {len(scan_layers)} layers...")
    per_layer = {}
    for lid in tqdm(scan_layers, desc='Layer scan'):
        result = run_single_layer_config(args, lid, layers, model, tokenizer, device, seq_len)
        per_layer[lid] = result

    # Summarize: recovery rate = (MA - floor) / (baseline - floor)
    summary = {
        'model': args.model,
        'n_layers': n_layers,
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline,
        'floor_all_disabled': floor,
        'per_layer': per_layer,
        'recovery_rate': {},
    }

    print(f"\n{'='*80}")
    print("SINGLE-LAYER ACTIVATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Layer':>6} | {'Top1 (mean)':>12} | {'Recovery%':>10}")
    print('-' * 40)

    denom = baseline['top1_mean'] - floor['top1_mean']
    for lid in scan_layers:
        t = per_layer[lid]['top1_mean']
        recovery = ((t - floor['top1_mean']) / denom * 100) if denom > 1e-6 else 0.0
        summary['recovery_rate'][lid] = float(recovery)
        print(f"{lid:>6} | {t:>12.1f} | {recovery:>9.1f}%")

    # Find best single layer
    best_lid = max(scan_layers, key=lambda l: per_layer[l]['top1_mean'])
    best_recovery = summary['recovery_rate'][best_lid]
    summary['best_single_layer'] = int(best_lid)
    summary['best_recovery_pct'] = float(best_recovery)

    print(f"\n>>> Best single layer: L{best_lid} recovers {best_recovery:.1f}% of MA")
    if best_recovery > 50:
        print(">>> VERDICT: Concentrated single-source generation")
    elif best_recovery > 20:
        print(">>> VERDICT: Partial single-source, likely multi-layer contribution")
    else:
        print(">>> VERDICT: Dispersed multi-source generation (no single layer suffices)")

    # Save
    out_path = os.path.join(args.savedir, f'{args.model}_rq6_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
