#!/usr/bin/env python3
"""
RQ1 — Attention 消融实验（简化独立版）

证伪 H₀（"Attention 是 MA 起源"）：
  关闭全部 attention 头后，若 MA 未归零，则 H₀ 失败。

主判据：
  residual% = top1_disabled / top1_baseline * 100  >  0   →   PASS

副判据（模式分类）：
  ΔMA% = (top1_disabled - top1_baseline) / top1_baseline * 100
  ΔMA < 0  →  Generative（attention 放大器）
  ΔMA > 0  →  Suppressive（attention 抑制器）

用法：
  python exp1_minimal.py --model gptj_6b --nsamples 30
  python exp1_minimal.py --model opt_6.7b --nsamples 30  # 预期 +744% Sup

输出：
  results/<model>/exp1_results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

import paper_experiments.lib as lib  # noqa: E402


# ============================================================================
# Attention 清零 hook
# ============================================================================
class ZeroAttentionHook:
    """让 attention 层的输出 = 0（保留其他层路径）"""

    def __init__(self):
        self.handles = []

    def __call__(self, module, input, output):
        # output 可能是 Tensor 或 (Tensor, ...) tuple（HF 不同模型不同）
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + tuple(output[1:])
        return torch.zeros_like(output)

    def attach(self, model, model_name):
        """对所有层的 attention 模块注册 forward hook"""
        # 不同架构 attention 子模块名字不同
        attn_attrs = ["self_attn", "self_attention", "attention", "attn"]

        layers = lib.get_layers(model, model_name)
        for layer in layers:
            for attr in attn_attrs:
                attn = getattr(layer, attr, None)
                if attn is not None:
                    self.handles.append(attn.register_forward_hook(self))
                    break

        if not self.handles:
            raise RuntimeError(
                f"Could not find attention submodule on layers of {model_name}"
            )

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================================
# MA 测量
# ============================================================================
@torch.no_grad()
def measure_top1(model, tokenizer, device, nsamples, seqlen):
    """跑 nsamples 个 wikitext 样本，返回所有层中 |h| 的全局最大值"""
    data = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seqlen, device=device)

    top1_values = []
    for seq in tqdm(data, desc="Measuring", leave=False):
        out = model(seq, output_hidden_states=True)
        per_seq_max = max(h.abs().max().item() for h in out.hidden_states)
        top1_values.append(per_seq_max)

    return {
        "top1_mean": sum(top1_values) / len(top1_values),
        "top1_max": max(top1_values),
        "top1_values": top1_values,
    }


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="RQ1: Attention ablation feasibility test")
    parser.add_argument("--model", required=True, help="Model name (see lib/model_dict.py)")
    parser.add_argument("--nsamples", type=int, default=30, help="WikiText samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--savedir",
        default="results",
        help="Output directory (will create <savedir>/<model>/)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"RQ1 — Attention Ablation Feasibility Test")
    print(f"{'=' * 60}")
    print(f"Model:     {args.model}")
    print(f"Samples:   {args.nsamples} × seqlen={args.seqlen}")

    # ---------- 1. 加载模型 ----------
    print(f"\n[1/3] Loading model...")
    model, tokenizer, device, _, _, seqlen = lib.load_llm(args)
    model.eval()
    seqlen = args.seqlen or seqlen

    # ---------- 2. Baseline ----------
    print(f"\n[2/3] Measuring baseline MA...")
    baseline = measure_top1(model, tokenizer, device, args.nsamples, seqlen)
    print(f"  Baseline top1_max  = {baseline['top1_max']:.3f}")
    print(f"  Baseline top1_mean = {baseline['top1_mean']:.3f}")

    # ---------- 3. 关闭所有 attention，再测 ----------
    print(f"\n[3/3] Disabling all attention layers...")
    hook = ZeroAttentionHook()
    hook.attach(model, args.model)
    try:
        disabled = measure_top1(model, tokenizer, device, args.nsamples, seqlen)
    finally:
        hook.detach()
    print(f"  Disabled top1_max  = {disabled['top1_max']:.3f}")
    print(f"  Disabled top1_mean = {disabled['top1_mean']:.3f}")

    # ---------- 4. 计算判据 ----------
    base_max = baseline["top1_max"]
    dis_max = disabled["top1_max"]

    if base_max > 0:
        residual_pct = dis_max / base_max * 100
        delta_ma_pct = (dis_max - base_max) / base_max * 100
    else:
        residual_pct = float("inf")
        delta_ma_pct = float("inf")

    pass_h0_falsified = residual_pct > 0
    mode = "generative" if delta_ma_pct < 0 else "suppressive"

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  residual%        = {residual_pct:.2f}%   ← 主判据 (>0 → PASS)")
    print(f"  ΔMA%             = {delta_ma_pct:+.2f}%  ← 模式分类")
    print(f"  Mode             = {mode}")
    print(f"  H₀ (attention is origin) FALSIFIED? {pass_h0_falsified}")

    # ---------- 5. 保存 ----------
    out_dir = os.path.join(args.savedir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp1_results.json")

    result = {
        "model": args.model,
        "nsamples": args.nsamples,
        "seqlen": seqlen,
        "timestamp": datetime.now().isoformat(),
        "baseline_top1_max": base_max,
        "baseline_top1_mean": baseline["top1_mean"],
        "disabled_top1_max": dis_max,
        "disabled_top1_mean": disabled["top1_mean"],
        "residual_pct": residual_pct,
        "delta_ma_pct": delta_ma_pct,
        "mode": mode,
        "h0_falsified": pass_h0_falsified,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
