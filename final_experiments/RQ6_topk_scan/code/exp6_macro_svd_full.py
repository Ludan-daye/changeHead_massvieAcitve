#!/usr/bin/env python3
"""
RQ6 Macro-SVD Full: macro-block SVD + per-layer W2 SVD + cross-layer alignment.

For a given model and a list of 'origin_layers':
  1. Per-layer: SVD(W2) at each origin layer → v1_i, sigma_i
  2. Cross-layer alignment: cos(v1_i, v1_j) pairwise matrix
  3. Macro SVD: SVD of Δ = h_after_last - h_before_first (residual change)
  4. Function vs content word projection on v1_macro
  5. Alignment of v1_macro with each v1_i

Output tells us if:
  - Small models (GPT-2): multi-stage same-direction (high macro η, v1_i aligned)
  - Large models (GPT-J): single-stage (L2 v1 already = macro v1)
  - Dispersed-unaligned: each layer contributes orthogonally (low macro η)
"""

import os, sys, argparse, json
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib

FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'while',
    'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may',
    'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'no', 'not', 'only', 'also', 'so', 'such',
}


def get_blocks(model, name):
    nl = name.lower()
    if 'gpt2' in nl or 'gptj' in nl or 'gpt_j' in nl:
        return list(model.transformer.h)
    if 'opt' in nl:
        return list(model.model.decoder.layers)
    return list(model.model.layers)


def get_w2(model, name, layer_idx):
    """Return W2 (down projection) weight: shape [hidden, intermediate]."""
    nl = name.lower()
    blocks = get_blocks(model, name)
    blk = blocks[layer_idx]
    if 'gpt2' in nl:
        # GPT2Block.mlp.c_proj is Conv1D, .weight shape [intermediate, hidden] → transpose
        W = blk.mlp.c_proj.weight.detach().float().t()  # → [hidden, intermediate]
        return W
    if 'gptj' in nl or 'gpt_j' in nl:
        # GPTJ uses fc_out: Linear(intermediate, hidden), weight [hidden, intermediate]
        return blk.mlp.fc_out.weight.detach().float()
    if 'opt' in nl:
        return blk.fc2.weight.detach().float()  # [hidden, intermediate]
    return blk.mlp.down_proj.weight.detach().float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--origin_layers', type=str, required=True)
    p.add_argument('--nsamples', type=int, default=20)
    p.add_argument('--seqlen', type=int, default=512)
    p.add_argument('--savedir', type=str, required=True)
    p.add_argument('--access_token', type=str, default='')
    p.add_argument('--revision', type=str, default='main')
    args = p.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    origin = [int(x) for x in args.origin_layers.split(',')]
    print(f"\n{'='*80}\nMacro-SVD FULL: {args.model} layers {origin}\n{'='*80}", flush=True)

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    blocks = get_blocks(model, args.model)
    n_layers = len(blocks)
    L_first, L_last = min(origin), max(origin)

    # === Part 1: Per-layer W2 SVD ===
    print("\n[1] Per-layer W2 SVD...", flush=True)
    per_layer_info = {}
    v1_vectors = {}
    for lid in origin:
        W = get_w2(model, args.model, lid).cpu()
        print(f"  L{lid}: W2 shape {tuple(W.shape)}", flush=True)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        eta = (S[0] / S[1]).item() if len(S) > 1 else float('inf')
        v1 = Vh[0]  # [intermediate] since Vh is [min, intermediate]
        # Actually Vh shape = [min(hidden, intermediate), intermediate]
        # v1 lives in the intermediate space - but for cross-model we want
        # the direction in hidden space. Use U[:, 0] for that.
        u1 = U[:, 0]  # [hidden]
        per_layer_info[lid] = {
            'W_shape': list(W.shape),
            'sigma_top5': S[:5].tolist(),
            'eta': float(eta),
        }
        v1_vectors[lid] = u1  # store u1 (hidden-space direction) for cross-layer comparison
        print(f"    σ₁={S[0].item():.3f}, σ₂={S[1].item():.3f}, η={eta:.3f}", flush=True)

    # === Part 2: Cross-layer u1 alignment ===
    print("\n[2] Cross-layer u1 alignment (in hidden space)...", flush=True)
    alignment_matrix = {}
    for i in origin:
        alignment_matrix[i] = {}
        for j in origin:
            cos = torch.nn.functional.cosine_similarity(
                v1_vectors[i].unsqueeze(0), v1_vectors[j].unsqueeze(0)
            ).item()
            alignment_matrix[i][j] = float(abs(cos))
    print("  u1 cosine similarity matrix (absolute):")
    print("       " + " ".join(f"L{j:>2d}" for j in origin))
    for i in origin:
        row = " ".join(f"{alignment_matrix[i][j]:.3f}" for j in origin)
        print(f"    L{i:>2d}  {row}", flush=True)

    mean_align = np.mean([
        alignment_matrix[i][j]
        for i in origin for j in origin if i < j
    ])
    print(f"  Mean pairwise |cos|: {mean_align:.3f}", flush=True)

    # === Part 3: Macro SVD on Δ = h_out - h_in ===
    print(f"\n[3] Macro SVD on Δ (h after L{L_last} − h before L{L_first})...", flush=True)
    pre_buf, post_buf = [], []

    def pre_hook(m, args, kwargs):
        # Try positional first, then kwargs (for newer transformers)
        if args and len(args) > 0:
            x = args[0]
        elif 'hidden_states' in kwargs:
            x = kwargs['hidden_states']
        else:
            raise RuntimeError(f"Can't find hidden_states; args={args}, kwargs keys={list(kwargs.keys())}")
        pre_buf.append(x.detach().cpu().float().clone())

    def post_hook(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        post_buf.append(o.detach().cpu().float().clone())

    pre_h = blocks[L_first].register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_h = blocks[L_last].register_forward_hook(post_hook)

    # Per-block MA capture
    per_block_peaks = [[] for _ in range(n_layers)]

    def mk(idx):
        def h(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            per_block_peaks[idx].append(o.detach().abs().max(dim=-1)[0].cpu().clone())
        return h

    bhs = [blocks[i].register_forward_hook(mk(i)) for i in range(n_layers)]

    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=min(args.seqlen, seq_len), device=device)
    all_ids = []
    with torch.inference_mode():
        for seq in testseq_list:
            seq = seq.to(device)
            model(seq)
            all_ids.append(seq.view(-1).cpu().clone())

    pre_h.remove(); post_h.remove()
    for h in bhs: h.remove()

    H_in = torch.cat([t.view(-1, t.shape[-1]) for t in pre_buf], dim=0)
    H_out = torch.cat([t.view(-1, t.shape[-1]) for t in post_buf], dim=0)
    Delta = H_out - H_in
    print(f"  Δ shape: {Delta.shape}", flush=True)

    U_m, S_m, Vh_m = torch.linalg.svd(Delta, full_matrices=False)
    sigma_m = S_m.numpy()
    eta_m = float(sigma_m[0] / sigma_m[1]) if len(sigma_m) > 1 else float('inf')
    var1 = float(sigma_m[0]**2 / (sigma_m**2).sum() * 100)
    var5 = float((sigma_m[:5]**2).sum() / (sigma_m**2).sum() * 100)
    v1_macro = Vh_m[0]  # [hidden_dim]

    print(f"  σ top5: {sigma_m[:5]}", flush=True)
    print(f"  η_macro = {eta_m:.3f}", flush=True)
    print(f"  Var explained top-1: {var1:.1f}%, top-5: {var5:.1f}%", flush=True)

    # Align with per-layer u1
    print("\n  |cos(v1_macro, u1_i)|:")
    macro_per_layer_align = {}
    for lid in origin:
        cos = torch.nn.functional.cosine_similarity(
            v1_macro.unsqueeze(0), v1_vectors[lid].unsqueeze(0)
        ).item()
        macro_per_layer_align[lid] = float(abs(cos))
        print(f"    L{lid}: {abs(cos):.3f}", flush=True)

    # === Part 4: function vs content projection on v1_macro ===
    print("\n[4] Projection: func vs content on v1_macro...", flush=True)
    proj = (Delta @ v1_macro).numpy()
    abs_proj = np.abs(proj)
    token_ids = torch.cat(all_ids, dim=0).numpy()
    func_mask = np.array([
        tokenizer.decode([int(t)]).strip().lower() in FUNCTION_WORDS
        for t in token_ids
    ])
    func_mean = float(abs_proj[func_mask].mean()) if func_mask.any() else 0.0
    content_mean = float(abs_proj[~func_mask].mean()) if (~func_mask).any() else 0.0
    ratio = float(func_mean / content_mean) if content_mean > 0 else float('inf')
    pooled = np.sqrt((abs_proj[func_mask].var() + abs_proj[~func_mask].var()) / 2)
    cohen_d = float((func_mean - content_mean) / pooled) if pooled > 0 else 0.0
    print(f"  func_mean={func_mean:.3f}, content_mean={content_mean:.3f}", flush=True)
    print(f"  ratio={ratio:.2f}×, Cohen's d={cohen_d:.3f}", flush=True)

    # === Part 5: R² regression ===
    per_block_max = [torch.cat([p.view(-1) for p in per_block_peaks[i]], dim=0)
                     for i in range(n_layers) if per_block_peaks[i]]
    token_ma = torch.stack(per_block_max, dim=0).max(dim=0)[0].numpy()
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(abs_proj.reshape(-1, 1), token_ma)
    r2 = float(reg.score(abs_proj.reshape(-1, 1), token_ma))
    print(f"\n[5] R² = {r2:.3f}", flush=True)

    # === Save ===
    out = {
        'model': args.model,
        'origin_layers': origin,
        'timestamp': datetime.now().isoformat(),
        'per_layer': per_layer_info,
        'cross_layer_u1_align': alignment_matrix,
        'mean_pairwise_align': float(mean_align),
        'macro_svd': {
            'sigma_top10': sigma_m[:10].tolist(),
            'eta': eta_m,
            'var_top1_pct': var1,
            'var_top5_pct': var5,
        },
        'macro_per_layer_align': macro_per_layer_align,
        'projection': {
            'func_mean': func_mean,
            'content_mean': content_mean,
            'ratio': ratio,
            'cohen_d': cohen_d,
        },
        'regression_r2': r2,
    }
    out_path = os.path.join(args.savedir, f'{args.model}_macro_svd_full.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    print(f"\n{'='*80}\nVERDICT: {args.model}")
    print(f"  Per-layer η: {[per_layer_info[l]['eta'] for l in origin]}")
    print(f"  Mean u1 alignment across layers: {mean_align:.3f}")
    print(f"  Macro η: {eta_m:.3f}")
    print(f"  Macro top-1 var: {var1:.1f}%")
    print(f"  Func/content ratio: {ratio:.2f}×")
    print(f"  R²: {r2:.3f}")


if __name__ == '__main__':
    main()
