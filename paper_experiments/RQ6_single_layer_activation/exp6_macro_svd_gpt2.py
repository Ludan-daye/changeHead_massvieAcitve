#!/usr/bin/env python3
"""
RQ6 Macro-SVD: treat origin layers (L0..L4 for GPT-2) as one block and SVD
the cumulative residual-stream change Δ = h_out - h_in.

Question: does the dispersed-multi-layer MA mechanism have a coherent
principal direction at the macro scale (even though no single layer does)?

Three sub-analyses:
  (A) Singular spectrum of Δ matrix over tokens → η = σ₁/σ₂
  (B) Function-word vs content-word projection onto v₁_macro
  (C) Linear regression: |proj on v₁_macro| → top1 MA, R²

Output: macro_svd_results.json
"""

import os, sys, argparse, json, time
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib

# Common English function words (same list as RQ3)
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'while',
    'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may',
    'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'no', 'not', 'only', 'also', 'so', 'such',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--origin_layers', type=str, default='0,1,2,3,4',
                        help='Comma-separated layer ids forming the macro-block')
    parser.add_argument('--nsamples', type=int, default=20)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--access_token', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    origin = [int(x) for x in args.origin_layers.split(',')]
    print(f"\nMacro-block layers: {origin}")

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # GPT-2 specific block access; extend if needed
    if 'gpt2' in args.model.lower():
        blocks = list(model.transformer.h)
    else:
        blocks = list(model.model.layers)

    n_layers = len(blocks)
    L_first, L_last = min(origin), max(origin)
    print(f"Will capture residual h_in (before L{L_first}) and h_out (after L{L_last})")

    # Storage
    h_in_store = []
    h_out_store = []
    block_peak_per_token = []  # peak |activation| across all blocks per token (proxy for MA)

    def make_pre_hook(buf):
        def hook(module, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            buf.append(x.detach().cpu().float().clone())
        return hook

    def make_post_hook(buf):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            buf.append(o.detach().cpu().float().clone())
        return hook

    pre_buf = []
    post_buf = []
    pre_handle = blocks[L_first].register_forward_pre_hook(make_pre_hook(pre_buf))
    post_handle = blocks[L_last].register_forward_hook(make_post_hook(post_buf))

    # Per-block peak capture (for MA proxy)
    per_block_peaks = [[] for _ in range(n_layers)]
    block_handles = []
    for i, blk in enumerate(blocks):
        def mk(idx):
            def h(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                # max abs per token: shape [B, T, D] → [B, T]
                per_block_peaks[idx].append(o.detach().abs().max(dim=-1)[0].cpu().clone())
            return h
        block_handles.append(blk.register_forward_hook(mk(i)))

    # Run forward passes
    print(f"\nRunning {args.nsamples} sequences (seqlen={args.seqlen})...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=min(args.seqlen, seq_len), device=device)

    all_token_ids = []
    with torch.inference_mode():
        for si, seq in enumerate(testseq_list):
            seq = seq.to(device)
            model(seq)
            all_token_ids.append(seq.view(-1).cpu().clone())

    # Cleanup hooks
    pre_handle.remove()
    post_handle.remove()
    for h in block_handles:
        h.remove()

    # Stack: each is [B=1, T, D] → flatten to [B*T, D]
    H_in = torch.cat([t.view(-1, t.shape[-1]) for t in pre_buf], dim=0)
    H_out = torch.cat([t.view(-1, t.shape[-1]) for t in post_buf], dim=0)
    Delta = H_out - H_in   # [N_tokens, D]
    print(f"Captured Δ matrix: {Delta.shape}")

    # Per-token MA (max across all layers)
    # Each entry in per_block_peaks[i] is [B, T]; concat across sequences
    per_block_max = []
    for i in range(n_layers):
        if per_block_peaks[i]:
            per_block_max.append(torch.cat([p.view(-1) for p in per_block_peaks[i]], dim=0))
    block_stack = torch.stack(per_block_max, dim=0)  # [n_layers, N_tokens]
    token_ma = block_stack.max(dim=0)[0].numpy()     # [N_tokens]
    token_ids = torch.cat(all_token_ids, dim=0).numpy()
    assert len(token_ma) == Delta.shape[0]

    # ---------- (A) SVD on Δ ----------
    print("\n[A] SVD on Δ matrix...")
    U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
    sigma = S.numpy()
    eta = sigma[0] / sigma[1] if len(sigma) > 1 else float('inf')
    print(f"  σ₁={sigma[0]:.3f}, σ₂={sigma[1]:.3f}, σ₃={sigma[2]:.3f}, "
          f"η=σ₁/σ₂={eta:.3f}")
    print(f"  Top-5 σ: {sigma[:5]}")
    print(f"  Variance explained by top-1: {sigma[0]**2 / (sigma**2).sum() * 100:.1f}%")
    print(f"  Variance explained by top-5: {(sigma[:5]**2).sum() / (sigma**2).sum() * 100:.1f}%")

    v1 = Vh[0]  # principal right singular vector  [D]

    # ---------- (B) Function vs content projections ----------
    print("\n[B] Function-word vs content-word projection on v1...")
    # Project Δ onto v1 (each token gets a scalar)
    proj = (Delta @ v1).numpy()  # [N_tokens]
    abs_proj = np.abs(proj)

    func_mask = np.array([
        tokenizer.decode([int(t)]).strip().lower() in FUNCTION_WORDS
        for t in token_ids
    ])
    n_func, n_content = int(func_mask.sum()), int((~func_mask).sum())
    print(f"  Function tokens: {n_func}, Content tokens: {n_content}")

    func_proj = abs_proj[func_mask]
    content_proj = abs_proj[~func_mask]
    func_mean = float(func_proj.mean()) if n_func > 0 else 0.0
    content_mean = float(content_proj.mean()) if n_content > 0 else 0.0
    ratio = func_mean / content_mean if content_mean > 0 else float('inf')

    print(f"  Function |proj| mean: {func_mean:.3f}")
    print(f"  Content  |proj| mean: {content_mean:.3f}")
    print(f"  Ratio (func/content): {ratio:.2f}×")

    # Cohen's d
    pooled_std = np.sqrt((func_proj.var() + content_proj.var()) / 2)
    cohen_d = (func_mean - content_mean) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d: {cohen_d:.3f}")

    # ---------- (C) Regression: |proj| → MA ----------
    print("\n[C] Linear regression: |proj on v₁_macro| → token MA...")
    from sklearn.linear_model import LinearRegression
    X = abs_proj.reshape(-1, 1)
    y = token_ma
    reg = LinearRegression().fit(X, y)
    r_squared = reg.score(X, y)
    print(f"  R² = {r_squared:.4f}")
    print(f"  slope = {reg.coef_[0]:.3f}, intercept = {reg.intercept_:.3f}")

    # ---------- Save ----------
    out = {
        'model': args.model,
        'origin_layers': origin,
        'timestamp': datetime.now().isoformat(),
        'n_tokens': int(Delta.shape[0]),
        'svd': {
            'sigma_top10': sigma[:10].tolist(),
            'eta_sigma1_over_sigma2': float(eta),
            'var_explained_top1_pct': float(sigma[0]**2 / (sigma**2).sum() * 100),
            'var_explained_top5_pct': float((sigma[:5]**2).sum() / (sigma**2).sum() * 100),
        },
        'projection': {
            'n_function_tokens': n_func,
            'n_content_tokens': n_content,
            'func_proj_mean': func_mean,
            'content_proj_mean': content_mean,
            'ratio': float(ratio),
            'cohen_d': float(cohen_d),
        },
        'regression': {
            'r_squared': float(r_squared),
            'slope': float(reg.coef_[0]),
            'intercept': float(reg.intercept_),
        },
    }
    out_path = os.path.join(args.savedir, f'{args.model}_macro_svd.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT for macro-block L{L_first}..L{L_last}:")
    print(f"  η = {eta:.2f}  ({'concentrated' if eta > 1.5 else 'dispersed (no dominant direction)'})")
    print(f"  Top-1 explains {sigma[0]**2/(sigma**2).sum()*100:.1f}% of variance")
    print(f"  Function-word ratio: {ratio:.2f}× ({'support' if ratio > 1.5 else 'weak/no support'})")
    print(f"  R² = {r_squared:.3f} ({'strong' if r_squared > 0.5 else 'weak'})")


if __name__ == '__main__':
    main()
