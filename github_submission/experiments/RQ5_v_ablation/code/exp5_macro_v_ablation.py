#!/usr/bin/env python3
"""
Experiment 5 (Macro version): Multi-layer macro-v₁ projection-out ablation.

用于模式 B 多层协作模型（GPT-2, OPT, Qwen3-32B, Qwen3.5-27B 等）：
单层 W_down 的 σ₁/σ₂ 弱 → 单层 v-ablation 效果差（<30%）。
但把多层 Δh 聚合起来做 SVD 会得到强 macro σ₁/σ₂，macro v₁ 是
多层协作写入 MA 的"合力方向"。

这个脚本做的事：
  1. 前向采样，hook 捕获 Δh_macro = h_after_last − h_before_first
  2. SVD(Δh_macro) → 得到 macro v₁ (hidden space, shape [hidden_dim])
  3. 对 origin_layers 范围内每一层的 W_down 做投影消除：
         W_down_ablated = (I − vv^T) @ W_down
     这会让 W_down 的输出不再有 macro v₁ 分量
  4. 再前向一次，测 MA 变化

论文预期：对模式 B 模型 ΔMA ≤ -80%。

区别于 exp5_v_ablation.py：
  - 单层版：只改 1 个 layer 的 V 矩阵（替换 V 为随机正交）
  - macro 版：改多个 layer 的 W_down，把 macro_v1 分量投影出去
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


def get_blocks(model, model_name: str):
    """和 exp6_macro_svd_full 保持一致的层枚举方式。"""
    nl = model_name.lower()
    if 'gpt2' in nl or 'gptj' in nl or 'gpt_j' in nl:
        return list(model.transformer.h)
    if 'opt' in nl:
        return list(model.model.decoder.layers)
    if 'bloom' in nl:
        return list(model.transformer.h)
    if 'falcon' in nl:
        return list(model.transformer.h)
    return list(model.model.layers)


def measure_ma(model, tokenizer, device, layers, layer_id_for_capture,
               hidden_size, seq_len, model_name, nsamples, seed=0):
    """和 exp5_v_ablation.measure_ma 对齐的 MA 测量。"""
    data = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seq_len, device=device)

    layer = layers[layer_id_for_capture]
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
        'top1_values': top1_values,
        'top1_mean': float(np.mean(top1_values)) if top1_values else 0.0,
        'top1_max': float(np.max(top1_values)) if top1_values else 0.0,
        'ma_dims': ma_dims,
        'per_dim_max': {int(d): float(np.max(v)) for d, v in per_dim_values.items()} if per_dim_values else {},
    }


def compute_macro_v1(model, tokenizer, device, origin_layers, model_name, nsamples, seqlen):
    """
    跑一次前向采样，返回 macro v₁ (torch tensor, shape [hidden_dim], on CPU).

    macro v₁ 定义: SVD(Δh) 中最大奇异向量在 hidden 空间的方向，
                 Δh = h_after_L_last − h_before_L_first，L_first/L_last 取自 origin_layers。
    """
    blocks = get_blocks(model, model_name)
    L_first, L_last = min(origin_layers), max(origin_layers)

    pre_buf, post_buf = [], []

    def pre_hook(m, args, kwargs):
        if args and len(args) > 0:
            x = args[0]
        elif 'hidden_states' in kwargs:
            x = kwargs['hidden_states']
        else:
            raise RuntimeError("Can't find hidden_states")
        pre_buf.append(x.detach().cpu().float().clone())

    def post_hook(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        post_buf.append(o.detach().cpu().float().clone())

    pre_h = blocks[L_first].register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_h = blocks[L_last].register_forward_hook(post_hook)

    testseq_list = lib.get_data(tokenizer, nsamples=nsamples,
                                seqlen=seqlen, device=device)
    with torch.inference_mode():
        for seq in testseq_list:
            seq = seq.to(device)
            _ = model(seq)

    pre_h.remove()
    post_h.remove()

    H_in = torch.cat([t.view(-1, t.shape[-1]) for t in pre_buf], dim=0)
    H_out = torch.cat([t.view(-1, t.shape[-1]) for t in post_buf], dim=0)
    Delta = H_out - H_in
    U_m, S_m, Vh_m = torch.linalg.svd(Delta, full_matrices=False)
    v1 = Vh_m[0].clone()  # [hidden_dim]
    eta_macro = float(S_m[0] / S_m[1]) if len(S_m) > 1 else float('inf')
    return v1, float(S_m[0].item()), eta_macro


def project_out_direction(W: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    把矩阵 W 的输出空间投影，使得沿 v 方向的分量为 0。

    约定 W 的 shape = [hidden, intermediate]（lib.get_mlp_down_proj 返回的形状）
    v 的 shape = [hidden]，已归一化。

    实现:  W_ablated = (I - v v^T) W
    这等价于对每一列 w_i 做 w_i → w_i - (v^T w_i) v
    """
    assert W.dim() == 2
    v = v / (v.norm() + 1e-12)
    v = v.to(device=W.device, dtype=W.dtype)
    # 为了数值稳定，在 float32 上计算
    W32 = W.float()
    v32 = v.float()
    proj = v32 @ W32                          # [intermediate]
    W_ablated = W32 - torch.outer(v32, proj)  # [hidden, intermediate]
    return W_ablated.to(W.dtype)


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5 (Macro): Multi-layer macro-v1 projection-out ablation'
    )
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--origin_layers', type=str, required=True,
                        help='逗号分隔的聚合层列表，例如 "0,1,2,3,4"')
    parser.add_argument('--capture_layer', type=int, default=None,
                        help='测 MA 的层；默认用 origin_layers 最大值')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seqlen', type=int, default=None,
                        help='序列长度；默认采用 model 的 seq_len')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--savedir', type=str, default='results/exp5_macro_v_ablation/')
    parser.add_argument('--force_fp32', action='store_true', help='glm4 系列建议开启')

    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)

    origin = [int(x) for x in args.origin_layers.split(',')]
    capture_layer = args.capture_layer if args.capture_layer is not None else max(origin)

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 (MACRO): Multi-layer macro v₁ projection-out ablation")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Origin layers (for macro aggregation): {origin}")
    print(f"MA capture layer: {capture_layer}")
    print(f"Samples: {args.nsamples}")

    # --- 1. 加载模型 ---
    print("\n[1/6] Loading model...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    seqlen = args.seqlen if args.seqlen is not None else seq_len

    # --- 2. 在 capture_layer 启用 feat 捕获 ---
    lib.enable_custom_block(args.model, layers[capture_layer], capture_layer)

    # --- 3. 测 baseline MA ---
    print("\n[2/6] Measuring baseline MA...")
    baseline = measure_ma(model, tokenizer, device, layers, capture_layer,
                         hidden_size, seqlen, args.model, args.nsamples, args.seed)
    print(f"  baseline top1_mean = {baseline['top1_mean']:.2f}")
    print(f"  ma_dims = {baseline['ma_dims']}")

    # --- 4. 计算 macro v1 ---
    print("\n[3/6] Computing macro v1 via Δh SVD...")
    v1_macro, sigma1_macro, eta_macro = compute_macro_v1(
        model, tokenizer, device, origin, args.model, args.nsamples, seqlen
    )
    print(f"  macro σ₁ = {sigma1_macro:.3f}")
    print(f"  macro η (σ₁/σ₂) = {eta_macro:.3f}")
    print(f"  macro v₁ shape = {tuple(v1_macro.shape)}")

    # --- 5. 对 origin_layers 所有层做 W_down 投影消除 ---
    # For dense models: W' = (I - v v^T) W_down in one shot.
    # For MoE models: apply (I - v v^T) to EVERY expert's W_down individually,
    # preserving per-expert diversity. Handled transparently by
    # `lib.project_out_mlp_down_proj`.
    print(f"\n[4/6] Projecting out macro v₁ from W_down of layers {origin}...")
    W_originals = {}
    for lid in origin:
        W_orig_saved = lib.project_out_mlp_down_proj(args.model, layers[lid], v1_macro)
        W_originals[lid] = W_orig_saved
        if lib._is_moe_layer(layers[lid]):
            if isinstance(W_orig_saved, torch.Tensor):
                print(f"  L{lid}: MoE stacked W_down {tuple(W_orig_saved.shape)} — (I - vv^T) applied per-expert")
            else:
                print(f"  L{lid}: MoE modular W_down across {len(W_orig_saved)} experts — (I - vv^T) applied per-expert")
        else:
            print(f"  L{lid}: dense W_down {tuple(W_orig_saved.shape)} ablated")

    # --- 6. 测 ablated MA ---
    print("\n[5/6] Measuring ablated MA...")
    ablated = measure_ma(model, tokenizer, device, layers, capture_layer,
                        hidden_size, seqlen, args.model, args.nsamples, args.seed)
    print(f"  ablated top1_mean = {ablated['top1_mean']:.2f}")

    # --- 7. 还原权重 ---
    print("\n[6/6] Restoring original weights...")
    for lid in origin:
        lib.restore_mlp_down_proj(args.model, layers[lid], W_originals[lid])

    # --- 计算 ΔMA ---
    if baseline['top1_mean'] > 0:
        delta_top1_mean = (ablated['top1_mean'] - baseline['top1_mean']) / baseline['top1_mean'] * 100
    else:
        delta_top1_mean = 0.0
    if baseline['top1_max'] > 0:
        delta_top1_max = (ablated['top1_max'] - baseline['top1_max']) / baseline['top1_max'] * 100
    else:
        delta_top1_max = 0.0

    delta_per_dim = {}
    for d in (baseline['ma_dims'] or []):
        b_val = baseline['per_dim_max'].get(int(d), 0.0)
        a_val = ablated['per_dim_max'].get(int(d), 0.0)
        delta_per_dim[int(d)] = (a_val - b_val) / b_val * 100 if b_val > 0 else 0.0

    # --- 保存 ---
    results = {
        'model': args.model,
        'origin_layers': origin,
        'capture_layer': capture_layer,
        'nsamples': args.nsamples,
        'seqlen': seqlen,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat(),
        'macro_svd': {
            'sigma1': sigma1_macro,
            'eta': eta_macro,
        },
        'baseline': {
            'top1_mean': baseline['top1_mean'],
            'top1_max': baseline['top1_max'],
            'ma_dims': [int(d) for d in (baseline['ma_dims'] or [])],
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
    }

    out_path = os.path.join(args.savedir, f'{args.model}_macro_v_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Baseline MA: {baseline['top1_mean']:.2f}")
    print(f"  Ablated  MA: {ablated['top1_mean']:.2f}")
    print(f"  ΔMA (top1 mean): {delta_top1_mean:+.1f}%")
    print(f"  ΔMA (top1 max):  {delta_top1_max:+.1f}%")
    print(f"  Macro η (σ₁/σ₂): {eta_macro:.3f}")
    print(f"\n  Saved: {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
