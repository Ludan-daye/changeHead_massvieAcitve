#!/usr/bin/env python3
"""
Experiment 5 (Multi-K Single-Layer): Project-out top-K v directions of W_down.

Unlike `exp5_v_ablation.py` (which replaces the full V matrix with a random
orthogonal Q, preserving Σ and only scrambling output directions — this fails
on flat-spectrum models like bloom_7b1 because the σ power just gets spread
across a random basis), this script does **projection removal** of the first
K right-singular directions:

    U, S, Vh  = SVD(W_down)                         # W_down shape = [hidden, intermediate]
    V_topk    = Vh[:K].T                            # [intermediate, K]
    P         = V_topk @ V_topk.T                   # [intermediate, intermediate]
    W_ablated = W_down @ (I - P)

Semantics:
  - W_down operates as (h2 @ W_down.T) in forward. After the ablation,
    the projection removes any component of h2's contribution that lies
    along v_1 … v_K (the top-K right singular directions). Equivalently,
    the SVD of W_ablated drops the top-K singular terms:
        W_ablated = Σ_{i > K} σ_i u_i v_i^T
  - K=1 is equivalent to `W @ (I - v1 v1^T)` (matches the macro script
    definition, but applied single-layer).
  - K=rank sends W_ablated → 0, so ΔMA → -100%.

Why this is consistent with RQ4's polynomial formula:

    MA_{j*} = Σ_i σ_i (h2 · v_i) u_i[j*]

Projecting out v_1 … v_K removes exactly the first K terms of that sum,
giving a **controlled truncation** of the MA contribution.

Output JSON schema:
{
  "model": ..., "layer_id": ..., "method": "projection_removal_top_k_v",
  "svd_info": {"sigma_1", "sigma_2", "eta", "W_shape"},
  "baseline": {...},
  "ablations": {"1": {...}, "2": {...}, ...}
}
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib


# ---------------------------------------------------------------------------
# MLP down-projection module access (for bias ablation)
# ---------------------------------------------------------------------------
def _get_down_proj_module(model_name: str, layer):
    """Return the nn.Module representing the MLP down-projection for this
    layer, if it is a single dense module with a `.bias` attribute. Used to
    optionally zero-out `W_down.bias` during ablation.

    Returns None for:
      - MoE layers (no single down-proj module; per-expert bias ablation not
        implemented here)
      - Models whose down-proj has no bias (e.g. most modern llama-style MLPs
        use bias=False)
      - glm4 (multi-path; conservatively skipped for bias ablation — most
        glm4 MLPs use bias=False anyway)
    """
    if lib._is_moe_layer(layer):
        return None
    # dense architectures — return the nn.Linear (or equivalent) module
    if ("llama" in model_name or "mistral" in model_name or "qwen" in model_name
            or "yi" in model_name):
        return getattr(layer.mlp, "down_proj", None)
    if "gpt2" in model_name:
        # GPT-2 uses Conv1D which stores the bias as `.bias`
        return getattr(layer.mlp, "c_proj", None)
    if "gptj" in model_name or "gpt-j" in model_name:
        return getattr(layer.mlp, "fc_out", None)
    if "bloom" in model_name:
        return getattr(layer.mlp, "dense_4h_to_h", None)
    if "falcon" in model_name:
        return getattr(layer.mlp, "dense_4h_to_h", None)
    if "opt" in model_name:
        return getattr(layer, "fc2", None)
    if "phi" in model_name:
        return getattr(layer.mlp, "fc2", None)
    if "pythia" in model_name:
        return getattr(layer.mlp, "dense_4h_to_h", None)
    if "glm4" in model_name:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None
        return getattr(mlp, "down_proj", None) or getattr(mlp, "dense_4h_to_h", None)
    return None


# ---------------------------------------------------------------------------
# MA measurement (aligned with exp5_v_ablation.measure_ma)
# ---------------------------------------------------------------------------
def measure_ma(model, tokenizer, device, layers, layer_id, hidden_size, seq_len,
               model_name, nsamples, seed=0):
    """
    Measure massive activation statistics for a given layer.

    Returns dict with keys:
        'top1_mean', 'top1_max', 'ma_dims', 'per_dim_max'
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

            if ma_dims is None:
                detected = lib.detect_ma_dimensions(feat, top_k=2)
                ma_dims = [d[0] for d in detected]
                for d in ma_dims:
                    per_dim_values[d] = []

            top1 = feat_abs.max().item()
            top1_values.append(top1)

            if feat_abs.dim() == 3:
                feat_flat = feat_abs.view(-1, feat_abs.shape[-1])
            else:
                feat_flat = feat_abs
            for d in ma_dims:
                if d < feat_flat.shape[-1]:
                    per_dim_values[d].append(feat_flat[:, d].max().item())

    return {
        'top1_mean': float(np.mean(top1_values)) if top1_values else 0.0,
        'top1_max': float(np.max(top1_values)) if top1_values else 0.0,
        'ma_dims': ma_dims,
        'per_dim_max': {int(d): float(np.max(v)) for d, v in per_dim_values.items()} if per_dim_values else {},
    }


# ---------------------------------------------------------------------------
# Projection-removal primitive
# ---------------------------------------------------------------------------
def _project_out_top_k(W: torch.Tensor, k: int) -> torch.Tensor:
    """
    Project out the top-K right singular directions of W.

    Args:
        W: [m, n] tensor (hidden, intermediate).
        k: number of v directions to remove (1-indexed count).

    Returns:
        W_ablated, same shape/dtype/device as W.

    Implementation:
        U, S, Vh = svd(W)                       # Vh: [min(m,n), n]
        V_topk = Vh[:k].T                       # [n, k]
        W_ablated = W @ (I - V_topk V_topk.T)

    Numerically equivalent (and more efficient) form used here:
        W_ablated = W - (W @ V_topk) @ V_topk.T
        # which expands to Σ_{i>k} σ_i u_i v_i^T
    """
    assert W.dim() == 2, f"Expected 2D W, got {W.dim()}D"
    W_orig_dtype = W.dtype
    Wf = W.float()
    # SVD in float32 for numerical stability
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    rank = min(Wf.shape[0], Wf.shape[1])
    k_eff = min(max(k, 0), rank)
    if k_eff == 0:
        return W.clone()
    V_topk = Vh[:k_eff].T.contiguous()          # [n, k]
    # (W @ V_topk) @ V_topk.T  — all in float32
    proj = Wf @ V_topk                          # [m, k]
    W_ablated = Wf - proj @ V_topk.T            # [m, n]
    return W_ablated.to(W_orig_dtype)


# ---------------------------------------------------------------------------
# Ablation helpers (dense + MoE)
# ---------------------------------------------------------------------------
def ablate_top_k(model_name: str, layer, k: int, ablate_bias: bool = False):
    """
    Apply projection-removal of top-K v directions to the layer's W_down.

    For dense layers: single W_down ← W_down @ (I - V_topk V_topkᵀ).
    For MoE layers: apply the same projection *per expert*, because each
    expert has its own W_e with its own SVD — we cannot use averaged SVD
    directions here.

    When ``ablate_bias=True``, also zero out the W_down module's `.bias`
    (if it exists). This diagnoses whether the residual MA after projection
    is carried by the constant bias term (relevant for bloom / older archs
    where `dense_4h_to_h` has bias=True; modern llama-style MLPs use
    bias=False so this is a no-op).

    Returns:
        saved_state : tuple (W_original, bias_original) — `bias_original`
                      is None if no bias was saved. Pass back to
                      `restore_weights`.
        svd_info    : dict with sigma_1/sigma_2/eta/W_shape from the
                      effective (for MoE) or direct (for dense) W_down.
    """
    bias_original = None  # will hold (module_ref, saved_bias_tensor) if zeroed

    if lib._is_moe_layer(layer):
        experts = layer.mlp.experts

        # Stacked-tensor case: experts.down_proj is a Parameter [E, H, I]
        if hasattr(experts, 'down_proj') and isinstance(experts.down_proj, torch.nn.Parameter):
            W_stack = experts.down_proj.data        # [E, H, I]
            W_original = W_stack.clone()
            for e in range(W_stack.shape[0]):
                W_e = W_stack[e]
                W_e_ablated = _project_out_top_k(W_e, k)
                W_stack[e].copy_(W_e_ablated.to(W_stack.dtype).to(W_stack.device))
            W_eff = lib._moe_effective_down_proj(layer).float()
            _U, S_eff, _Vh = torch.linalg.svd(W_eff, full_matrices=False)
            svd_info = {
                'sigma_1': float(S_eff[0].item()),
                'sigma_2': float(S_eff[1].item()) if S_eff.shape[0] > 1 else 0.0,
                'eta': float((S_eff[0] / S_eff[1]).item()) if S_eff.shape[0] > 1 else float('inf'),
                'W_shape': list(W_eff.shape),
                'moe_num_experts': int(W_stack.shape[0]),
                'moe_mode': 'stacked',
            }
            # MoE bias ablation: if the stacked experts also have a stacked
            # down-proj bias (most Qwen3MoE variants do NOT), zero it. Rare
            # but handled.
            if ablate_bias and hasattr(experts, 'down_proj_bias') \
                    and isinstance(experts.down_proj_bias, torch.nn.Parameter):
                b_saved = experts.down_proj_bias.data.clone()
                experts.down_proj_bias.data.zero_()
                bias_original = ('moe_stacked_bias', b_saved)
            return (W_original, bias_original), svd_info

        # Modular experts (nn.ModuleList) case
        W_originals = {}
        biases_originals = {}  # keyed by (idx, name)
        for idx, e in enumerate(experts):
            for name in ('down_proj', 'w2'):
                sub = getattr(e, name, None)
                if sub is not None and hasattr(sub, 'weight'):
                    W_e = sub.weight.data
                    W_originals[(idx, name)] = W_e.clone()
                    W_e_ablated = _project_out_top_k(W_e, k)
                    sub.weight.data.copy_(W_e_ablated.to(W_e.dtype).to(W_e.device))
                    if ablate_bias and getattr(sub, 'bias', None) is not None:
                        biases_originals[(idx, name)] = sub.bias.data.clone()
                        sub.bias.data.zero_()
                    break
        W_eff = lib._moe_effective_down_proj(layer).float()
        _U, S_eff, _Vh = torch.linalg.svd(W_eff, full_matrices=False)
        svd_info = {
            'sigma_1': float(S_eff[0].item()),
            'sigma_2': float(S_eff[1].item()) if S_eff.shape[0] > 1 else 0.0,
            'eta': float((S_eff[0] / S_eff[1]).item()) if S_eff.shape[0] > 1 else float('inf'),
            'W_shape': list(W_eff.shape),
            'moe_num_experts': len(W_originals),
            'moe_mode': 'modular',
        }
        if biases_originals:
            bias_original = ('moe_modular_biases', biases_originals)
        return (W_originals, bias_original), svd_info

    # Dense path
    W_down = lib.get_mlp_down_proj(model_name, layer).clone()
    W_original = W_down.clone()

    W_ablated = _project_out_top_k(W_down, k)
    lib.set_mlp_down_proj(model_name, layer, W_ablated)

    _U, S, _Vh = torch.linalg.svd(W_down.float(), full_matrices=False)
    svd_info = {
        'sigma_1': float(S[0].item()),
        'sigma_2': float(S[1].item()) if S.shape[0] > 1 else 0.0,
        'eta': float((S[0] / S[1]).item()) if S.shape[0] > 1 else float('inf'),
        'W_shape': list(W_down.shape),
    }

    # Dense bias ablation — zero `module.bias` if it exists.
    if ablate_bias:
        dp_mod = _get_down_proj_module(model_name, layer)
        if dp_mod is not None and getattr(dp_mod, 'bias', None) is not None:
            b_saved = dp_mod.bias.data.clone()
            dp_mod.bias.data.zero_()
            bias_original = ('dense_bias', dp_mod, b_saved)
            svd_info['bias_shape'] = list(b_saved.shape)
            svd_info['bias_norm_original'] = float(b_saved.float().norm().item())
            svd_info['bias_max_abs_original'] = float(b_saved.float().abs().max().item())
        else:
            svd_info['bias_shape'] = None
            svd_info['bias_norm_original'] = None

    return (W_original, bias_original), svd_info


def restore_weights(model_name, layer, saved_state):
    """Restore original W_down weights (and optionally bias) after ablation.

    Accepts either the legacy (W_only) form or the new tuple form
    (W_original, bias_original) produced by `ablate_top_k`.
    """
    # Unpack new tuple form (W_original, bias_original), else treat as legacy.
    bias_original = None
    if isinstance(saved_state, tuple) and len(saved_state) == 2 and (
            saved_state[1] is None or (isinstance(saved_state[1], tuple)
                                       and len(saved_state[1]) >= 2
                                       and isinstance(saved_state[1][0], str))):
        W_original, bias_original = saved_state
    else:
        W_original = saved_state

    if lib._is_moe_layer(layer):
        experts = layer.mlp.experts
        if hasattr(experts, 'down_proj') and isinstance(experts.down_proj, torch.nn.Parameter):
            if not isinstance(W_original, torch.Tensor) or W_original.dim() != 3:
                raise RuntimeError("MoE restore: expected a 3D Tensor for stacked experts")
            experts.down_proj.data.copy_(
                W_original.to(experts.down_proj.dtype).to(experts.down_proj.device)
            )
            # Restore stacked MoE bias if saved
            if bias_original is not None and bias_original[0] == 'moe_stacked_bias':
                b_saved = bias_original[1]
                experts.down_proj_bias.data.copy_(
                    b_saved.to(experts.down_proj_bias.dtype).to(experts.down_proj_bias.device))
            return
        if isinstance(W_original, dict):
            for (idx, name), W in W_original.items():
                sub = getattr(experts[idx], name, None)
                if sub is not None and hasattr(sub, 'weight'):
                    sub.weight.data.copy_(W.to(sub.weight.dtype).to(sub.weight.device))
            # Restore modular MoE biases if saved
            if bias_original is not None and bias_original[0] == 'moe_modular_biases':
                for (idx, name), b_saved in bias_original[1].items():
                    sub = getattr(experts[idx], name, None)
                    if sub is not None and getattr(sub, 'bias', None) is not None:
                        sub.bias.data.copy_(b_saved.to(sub.bias.dtype).to(sub.bias.device))
            return
        raise RuntimeError("MoE restore: unsupported W_original shape")
    lib.set_mlp_down_proj(model_name, layer, W_original)
    # Restore dense bias
    if bias_original is not None and bias_original[0] == 'dense_bias':
        _, dp_mod, b_saved = bias_original
        if getattr(dp_mod, 'bias', None) is not None:
            dp_mod.bias.data.copy_(b_saved.to(dp_mod.bias.dtype).to(dp_mod.bias.device))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5 (Multi-K, Single-Layer): Project out top-K v directions of W_down'
    )
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--layer_id', type=int, default=2,
                        help='Target layer for ablation (0-indexed; L_origin)')
    parser.add_argument('--peak_layer', type=int, default=None,
                        help='Layer to MEASURE MA on (default = layer_id). For models where '
                             'peak_layer != L_origin (e.g. bloom_7b1 peak=12 origin=3, '
                             'opt_6.7b peak=25 origin=1), you must set --peak_layer to see '
                             'the true effect of ablation after downstream propagation.')
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 2, 3],
                        help='K values to test (multiple allowed); e.g. --top_k 1 2 3 5 10')
    parser.add_argument('--nsamples', type=int, default=30,
                        help='Number of samples to measure')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--savedir', type=str, default='results/wikitext_run/RQ5_multi_v/',
                        help='Output directory')
    parser.add_argument('--ablate_bias', action='store_true',
                        help='In addition to v-direction projection removal, zero out '
                             'W_down.bias during each ablation run (and restore after). '
                             'Use for diagnosing bias-dominated MA on older archs '
                             '(bloom, gpt2, gptj, falcon, opt). Modern llama-style MLPs '
                             'use bias=False so this is a no-op.')

    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)

    # Dedup + sort K values
    k_list = sorted(set(int(k) for k in args.top_k if int(k) >= 1))
    if not k_list:
        raise ValueError("--top_k must include at least one K >= 1")

    measure_layer_id = args.peak_layer if args.peak_layer is not None else args.layer_id

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 (Multi-K, Single-Layer): Projection-Removal of Top-K v Directions")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Ablation Layer: {args.layer_id}")
    print(f"Measure Layer:  {measure_layer_id}{' (= layer_id)' if measure_layer_id == args.layer_id else ''}")
    print(f"K values: {k_list}")
    print(f"Samples: {args.nsamples}")
    print(f"Ablate bias: {args.ablate_bias}")

    # --- Load model ---
    print("\n[1/4] Loading model...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    layer = layers[args.layer_id]
    # enable feat capture on the MEASURE layer (may differ from ablation layer)
    lib.enable_custom_block(args.model, layers[measure_layer_id], measure_layer_id)
    if measure_layer_id != args.layer_id:
        # also enable on ablation layer in case ablate_top_k / SVD helpers expect it
        lib.enable_custom_block(args.model, layer, args.layer_id)

    # --- Baseline MA ---
    print("\n[2/4] Measuring baseline massive activations...")
    baseline = measure_ma(model, tokenizer, device, layers, measure_layer_id,
                         hidden_size, seq_len, args.model, args.nsamples, args.seed)
    print(f"  Baseline Top1 (mean): {baseline['top1_mean']:.2f}")
    print(f"  Baseline Top1 (max):  {baseline['top1_max']:.2f}")
    print(f"  Detected MA dims: {baseline['ma_dims']}")
    for d, v in baseline['per_dim_max'].items():
        print(f"    Dim {d}: max = {v:.2f}")

    # --- Loop over K ---
    print(f"\n[3/4] Running projection-removal ablations for K in {k_list} ...")
    ablations = {}
    svd_info_ref = None
    for k in k_list:
        print(f"\n  --- K = {k} ---")
        saved_state, svd_info = ablate_top_k(args.model, layer, k,
                                             ablate_bias=args.ablate_bias)
        if svd_info_ref is None:
            svd_info_ref = svd_info  # record once; SVD of original W is the same across K
            print(f"  W_down shape: {svd_info['W_shape']}")
            print(f"  sigma_1 = {svd_info['sigma_1']:.4f}")
            print(f"  sigma_2 = {svd_info['sigma_2']:.4f}")
            print(f"  eta (sigma_1/sigma_2) = {svd_info['eta']:.4f}")
            if args.ablate_bias:
                if svd_info.get('bias_shape') is None:
                    print(f"  [ablate_bias] No bias on W_down module — no-op.")
                else:
                    print(f"  [ablate_bias] bias shape: {svd_info['bias_shape']}, "
                          f"||b||₂ = {svd_info['bias_norm_original']:.4f}, "
                          f"max|b| = {svd_info['bias_max_abs_original']:.4f}")

        ablated = measure_ma(model, tokenizer, device, layers, measure_layer_id,
                            hidden_size, seq_len, args.model, args.nsamples, args.seed)
        print(f"  Ablated Top1 (mean): {ablated['top1_mean']:.2f}")
        print(f"  Ablated Top1 (max):  {ablated['top1_max']:.2f}")

        restore_weights(args.model, layer, saved_state)

        if baseline['top1_mean'] > 0:
            d_mean = (ablated['top1_mean'] - baseline['top1_mean']) / baseline['top1_mean'] * 100
        else:
            d_mean = 0.0
        if baseline['top1_max'] > 0:
            d_max = (ablated['top1_max'] - baseline['top1_max']) / baseline['top1_max'] * 100
        else:
            d_max = 0.0

        per_dim_pct = {}
        for d in (baseline['ma_dims'] or []):
            b_val = baseline['per_dim_max'].get(int(d), 0.0)
            a_val = ablated['per_dim_max'].get(int(d), 0.0)
            per_dim_pct[int(d)] = (a_val - b_val) / b_val * 100 if b_val > 0 else 0.0

        print(f"  ΔMA (top1 mean): {d_mean:+.1f}%")
        print(f"  ΔMA (top1 max):  {d_max:+.1f}%")
        for d, pct in per_dim_pct.items():
            print(f"  ΔMA (Dim {d}):    {pct:+.1f}%")

        ablations[str(k)] = {
            'top1_mean': ablated['top1_mean'],
            'top1_max': ablated['top1_max'],
            'per_dim_max': {str(kk): v for kk, v in ablated['per_dim_max'].items()},
            'delta_ma_mean_pct': d_mean,
            'delta_ma_max_pct': d_max,
            'per_dim_pct': {str(kk): v for kk, v in per_dim_pct.items()},
        }

    # --- Save results ---
    print("\n[4/4] Saving results...")
    results = {
        'model': args.model,
        'layer_id': args.layer_id,
        'measure_layer_id': measure_layer_id,
        'method': 'projection_removal_top_k_v',
        'ablate_bias': bool(args.ablate_bias),
        'k_values': k_list,
        'nsamples': args.nsamples,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'svd_info': svd_info_ref if svd_info_ref is not None else {},
        'baseline': {
            'top1_mean': baseline['top1_mean'],
            'top1_max': baseline['top1_max'],
            'ma_dims': [int(d) for d in (baseline['ma_dims'] or [])],
            'per_dim_max': {str(k): v for k, v in baseline['per_dim_max'].items()},
        },
        'ablations': ablations,
    }

    out_path = os.path.join(args.savedir, f'{args.model}_v_ablation_multi_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer_id}")
    if svd_info_ref is not None:
        print(f"  eta (σ₁/σ₂): {svd_info_ref['eta']:.2f}")
    print(f"  Baseline MA (top1 mean): {baseline['top1_mean']:.2f}")
    for k in k_list:
        print(f"  K={k:<3d}  ablated MA: {ablations[str(k)]['top1_mean']:>10.2f}"
              f"   ΔMA: {ablations[str(k)]['delta_ma_mean_pct']:+.1f}%")
    print(f"\n  Saved: {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
