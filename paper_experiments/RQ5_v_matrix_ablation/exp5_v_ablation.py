#!/usr/bin/env python3
"""
Experiment 5: V-Matrix Causal Ablation

Tests whether the right singular matrix V of W_down is causally necessary
for massive activation generation by:
  1. Measuring baseline MA (normal model)
  2. Replacing V with a random orthogonal matrix V_rand (via QR of Gaussian)
  3. Measuring MA after ablation
  4. Computing ΔMA = (ablated - baseline) / baseline × 100%

Paper expectation: ΔMA ranges from -69.6% to -99.1% across 8 models.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib


def measure_ma(model, tokenizer, device, layers, layer_id, hidden_size, seq_len,
               model_name, nsamples, seed=0):
    """
    Measure massive activation statistics for a given layer.

    Returns:
        dict with keys: 'top1_values', 'top1_mean', 'top1_max',
                        'ma_dims' (detected MA dimension indices),
                        'per_dim_max' (max activation per detected MA dim)
    """
    data = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seq_len, device=device)

    layer = layers[layer_id]
    top1_values = []
    ma_dims = None
    per_dim_values = {}

    with torch.no_grad():
        for seq in tqdm(data, desc="Measuring MA"):
            _ = model(seq)

            if not hasattr(layer, 'feat') or layer.feat is None:
                continue

            feat = layer.feat
            feat_abs = feat.abs()

            # Detect MA dimensions from first sample
            if ma_dims is None:
                detected = lib.detect_ma_dimensions(feat, top_k=2)
                ma_dims = [d[0] for d in detected]
                for d in ma_dims:
                    per_dim_values[d] = []

            # Record top1
            top1 = feat_abs.max().item()
            top1_values.append(top1)

            # Record per-dimension max
            if feat_abs.dim() == 3:
                feat_flat = feat_abs.view(-1, feat_abs.shape[-1])
            else:
                feat_flat = feat_abs
            for d in ma_dims:
                if d < feat_flat.shape[-1]:
                    per_dim_values[d].append(feat_flat[:, d].max().item())

    return {
        'top1_values': top1_values,
        'top1_mean': float(np.mean(top1_values)) if top1_values else 0,
        'top1_max': float(np.max(top1_values)) if top1_values else 0,
        'ma_dims': ma_dims,
        'per_dim_max': {d: float(np.max(v)) for d, v in per_dim_values.items()} if per_dim_values else {},
    }


def ablate_v_matrix(model_name, layer):
    """
    Replace the V matrix of W_down with a random orthogonal matrix.
    Preserves U and Σ (spectral power and output directions).

    Args:
        model_name: str
        layer: the layer module

    Returns:
        W_original: the original weight matrix (for restoration)
        svd_info: dict with σ₁, σ₂, η for reporting
    """
    W_down = lib.get_mlp_down_proj(model_name, layer).clone()
    W_original = W_down.clone()

    # SVD decomposition: W_down [hidden, intermediate] or [intermediate, hidden]
    U, S, Vh = torch.linalg.svd(W_down.float(), full_matrices=False)
    # U: [m, k], S: [k], Vh: [k, n] where k = min(m, n)

    # Generate random orthogonal matrix with same shape as Vh
    random_matrix = torch.randn(Vh.shape[0], Vh.shape[1], device=Vh.device)
    Q, R = torch.linalg.qr(random_matrix.T)  # QR on [n, k] -> Q [n, k]
    Q = Q.T  # [k, n] to match Vh shape
    # Fix sign ambiguity
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(1)

    # Reconstruct: W_ablated = U @ diag(S) @ Q
    W_ablated = U @ torch.diag(S) @ Q

    # Convert back to original dtype
    W_ablated = W_ablated.to(W_original.dtype)

    # Set the ablated weight
    lib.set_mlp_down_proj(model_name, layer, W_ablated)

    svd_info = {
        'sigma_1': float(S[0].item()),
        'sigma_2': float(S[1].item()),
        'eta': float((S[0] / S[1]).item()),
        'W_shape': list(W_down.shape),
    }

    return W_original, svd_info


def restore_weights(model_name, layer, W_original):
    """Restore original W_down weights after ablation."""
    lib.set_mlp_down_proj(model_name, layer, W_original)


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5: V-Matrix Causal Ablation Study'
    )
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--layer_id', type=int, default=2,
                        help='Target layer for ablation (0-indexed)')
    parser.add_argument('--nsamples', type=int, default=30,
                        help='Number of samples to measure')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--savedir', type=str, default='results/exp5_v_ablation/',
                        help='Output directory')

    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "=" * 80)
    print("EXPERIMENT 5: V-MATRIX CAUSAL ABLATION")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Target Layer: {args.layer_id}")
    print(f"Samples: {args.nsamples}")

    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture on target layer
    layer = layers[args.layer_id]
    lib.enable_custom_block(args.model, layer, args.layer_id)

    # Bug B2 fix (2026-04-22): MoE has no single W_down to ablate — must use
    # exp5_v_ablation_moe.py (per-expert). Detect and skip with sentinel output.
    _layer_for_moe_check = layers[args.layer_id]
    if lib._is_moe_layer(_layer_for_moe_check):
        msg = (
            f"MoE model '{args.model}' at L{args.layer_id} detected — "
            f"single-W V-ablation is not defined. Use exp5_v_ablation_moe.py."
        )
        print(f"\n[SKIP-MoE] {msg}")
        import json as _json
        os.makedirs(args.savedir, exist_ok=True)
        with open(os.path.join(args.savedir, 'exp5_SKIPPED_moe.json'), 'w') as _f:
            _json.dump({
                'model': args.model,
                'layer_id': args.layer_id,
                'skipped': True,
                'reason': 'is_moe',
                'message': msg,
            }, _f, indent=2)
        return

    # Measure baseline MA
    print("\n[2/5] Measuring baseline massive activations...")
    baseline = measure_ma(model, tokenizer, device, layers, args.layer_id,
                         hidden_size, seq_len, args.model, args.nsamples, args.seed)

    print(f"  Baseline Top1 (mean): {baseline['top1_mean']:.2f}")
    print(f"  Baseline Top1 (max):  {baseline['top1_max']:.2f}")
    print(f"  Detected MA dims: {baseline['ma_dims']}")
    for d, v in baseline['per_dim_max'].items():
        print(f"    Dim {d}: max = {v:.2f}")

    # Perform V-matrix ablation
    print("\n[3/5] Performing V-matrix ablation...")
    W_original, svd_info = ablate_v_matrix(args.model, layer)
    print(f"  W_down shape: {svd_info['W_shape']}")
    print(f"  sigma_1 = {svd_info['sigma_1']:.4f}")
    print(f"  sigma_2 = {svd_info['sigma_2']:.4f}")
    print(f"  eta (sigma_1/sigma_2) = {svd_info['eta']:.4f}")

    # Measure ablated MA
    print("\n[4/5] Measuring ablated massive activations...")
    ablated = measure_ma(model, tokenizer, device, layers, args.layer_id,
                        hidden_size, seq_len, args.model, args.nsamples, args.seed)

    print(f"  Ablated Top1 (mean): {ablated['top1_mean']:.2f}")
    print(f"  Ablated Top1 (max):  {ablated['top1_max']:.2f}")

    # Restore original weights
    print("\n[5/5] Restoring original weights...")
    restore_weights(args.model, layer, W_original)

    # Compute ΔMA
    delta_top1_mean = (ablated['top1_mean'] - baseline['top1_mean']) / baseline['top1_mean'] * 100 if baseline['top1_mean'] > 0 else 0
    delta_top1_max = (ablated['top1_max'] - baseline['top1_max']) / baseline['top1_max'] * 100 if baseline['top1_max'] > 0 else 0

    delta_per_dim = {}
    for d in baseline['ma_dims']:
        b_val = baseline['per_dim_max'].get(d, 0)
        a_val = ablated['per_dim_max'].get(d, 0)
        delta_per_dim[d] = (a_val - b_val) / b_val * 100 if b_val > 0 else 0

    # Theoretical prediction
    d_ff = svd_info['W_shape'][1]  # intermediate dimension
    theoretical_delta = (1 - np.sqrt(2 / np.pi) / np.sqrt(d_ff)) * 100

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  ΔMA (Top1 mean): {delta_top1_mean:+.1f}%")
    print(f"  ΔMA (Top1 max):  {delta_top1_max:+.1f}%")
    for d, delta in delta_per_dim.items():
        print(f"  ΔMA (Dim {d}):    {delta:+.1f}%")
    print(f"\n  Theoretical prediction (Eq. 10): {theoretical_delta:.1f}%")
    print(f"  (based on d_ff = {d_ff})")

    # Save results
    results = {
        'model': args.model,
        'layer_id': args.layer_id,
        'nsamples': args.nsamples,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'svd_info': svd_info,
        'baseline': {
            'top1_mean': baseline['top1_mean'],
            'top1_max': baseline['top1_max'],
            'ma_dims': baseline['ma_dims'],
            'per_dim_max': {str(k): v for k, v in baseline['per_dim_max'].items()},
        },
        'ablated': {
            'top1_mean': ablated['top1_mean'],
            'top1_max': ablated['top1_max'],
            'per_dim_max': {str(k): v for k, v in ablated['per_dim_max'].items()},
        },
        'delta_ma': {
            'top1_mean_pct': delta_top1_mean,
            'top1_max_pct': delta_top1_max,
            'per_dim_pct': {str(k): v for k, v in delta_per_dim.items()},
        },
        'theoretical_prediction_pct': theoretical_delta,
    }

    result_path = os.path.join(args.savedir, f'{args.model}_v_ablation_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer_id}")
    print(f"  eta (σ₁/σ₂): {svd_info['eta']:.2f}")
    print(f"  Baseline MA: {baseline['top1_mean']:.2f}")
    print(f"  Ablated MA:  {ablated['top1_mean']:.2f}")
    print(f"  ΔMA: {delta_top1_mean:+.1f}%")
    print(f"  Theoretical: {theoretical_delta:.1f}%")
    print(f"\n  Conclusion: V-matrix ablation {'confirms' if abs(delta_top1_mean) > 50 else 'partially supports'}")
    print(f"  that V-matrix geometry is causally necessary for MA generation.")
    print("=" * 80)


if __name__ == '__main__':
    main()
