"""
Experiment 5c: Predict-entropy vs |h₂·v₁| correlation.

Tests hypothesis C from EXPERIMENT_PLAN.md RQ3:
    "MA is written at the lowest-entropy token positions."

For each token position t in N wikitext sequences at the origin layer:
    - capture h₂(t) via forward-pre hook on down_proj input
    - compute predict entropy H(t) = -Σ_v p(v | x_1..t) log p(v | x_1..t)
      from model's output logits at position t
    - record (token_id, alignment = |h₂(t) · v₁|, entropy H(t))

Then:
    - Spearman ρ(H, |h₂·v₁|) — expected < -0.5 if H(C) holds
    - Cohen's d between bottom-10% entropy vs overall |h₂·v₁|
      — expected > 0.8 if H(C) holds
    - Also: mean |h₂·v₁| per entropy decile (for plotting)

Output: {savedir}/exp5c_entropy_results.json
Usage: python3 exp5c_entropy.py --model qwen3_0.6b --layer_id 2 --nsamples 30 --savedir ...
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib
from scipy.stats import spearmanr


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer_id", type=int, required=True)
    p.add_argument("--nsamples", type=int, default=30)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--savedir", required=True)
    p.add_argument("--access_token", default="")
    p.add_argument("--revision", default="main")
    args = p.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print(f"=== H(C) entropy test: {args.model} L{args.layer_id} ===")
    print("Loading model/data...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    data = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    layer = layers[args.layer_id]

    # Capture h₂ (input to down_proj = post-activation) via forward_pre_hook
    h2_list = []
    mlp_parts = lib.get_mlp_submodules(args.model, layer)
    down_proj = mlp_parts["down_proj"]

    def pre_hook(module, inp):
        h2_list.append(
            inp[0].detach().float().cpu().clone()
            if isinstance(inp, tuple)
            else inp.detach().float().cpu().clone()
        )

    handle = down_proj.register_forward_pre_hook(pre_hook)

    # SVD of W_down to get v₁
    print("Computing v₁ from W_down SVD...")
    W2 = lib.get_mlp_down_proj(args.model, layer).detach().cpu().float()
    # W2 shape [hidden, intermediate]; v₁ is the top right singular vector
    # = top left singular vector of W2.T = [intermediate, hidden]
    U, S, Vt = torch.linalg.svd(W2, full_matrices=False)
    # h₂ ∈ R^intermediate; its projection on v₁ = right-singular vec of W_down in intermediate space
    # W_down = U @ diag(S) @ V.T; V.T rows are right singular vectors in intermediate space
    # To project h₂ (intermediate) onto top input direction, use Vt[0] (intermediate-space vec)
    # BUT exp5_function_words uses W2.t() = [intermediate, hidden], so v₁ is actually U's top col.
    # Let's match exp5: W2_exp5 = get_mlp_down_proj().t() = [intermediate, hidden]
    #   SVD: W2_exp5 = U' S' V'.T, top right singular of W2_exp5 is V'[0] in hidden-space.
    # For our h₂ @ v₁ projection (h₂ in intermediate), we want the TOP LEFT singular of W2_exp5
    # which is U'[:,0] in intermediate-space.
    # So: compute SVD of W2.T (= [intermediate, hidden]) and take U[:,0]
    W2_T = W2.t()  # [intermediate, hidden]
    U2, S2, _ = torch.linalg.svd(W2_T, full_matrices=False)
    v1 = U2[:, 0].numpy()  # [intermediate]
    print(f"  v₁ shape: {v1.shape}, top singular {S2[0].item():.4f}, ratio σ₁/σ₂={S2[0].item()/S2[1].item():.3f}")

    # Per-position records
    positions = []  # list of dicts: {seq, pos, token_id, align_abs, entropy}

    print(f"Running {len(data)} sequences...")
    with torch.no_grad():
        for seq_idx, batch in enumerate(tqdm(data)):
            tokens = batch.to(device)
            token_ids = batch.squeeze(0).cpu().numpy()
            h2_list.clear()

            # Forward pass, keeping logits
            out = model(tokens)
            logits = out.logits if hasattr(out, "logits") else out[0]
            # logits: [1, seq_len, vocab] → float32 on CPU, compute per-position entropy
            logits = logits.detach().float().cpu()
            # softmax + entropy
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            # H[b,t] = -sum_v p log p
            H = -(p * logp).sum(dim=-1)  # [1, seq_len]
            H = H[0].numpy()  # [seq_len]

            if not h2_list:
                continue
            h2 = h2_list[-1]
            if h2.dim() == 3:
                h2 = h2[0]
            h2 = h2.numpy()  # [seq_len, intermediate]

            # Alignment per position: |h₂(t) · v₁|
            align = np.abs(h2 @ v1)  # [seq_len]

            n = min(len(H), len(align), len(token_ids))
            # Skip first token (no context) and last (no next-token prediction possible)
            for t in range(1, n - 1):
                positions.append({
                    "seq": seq_idx,
                    "pos": int(t),
                    "token_id": int(token_ids[t]),
                    "align_abs": float(align[t]),
                    # Entropy of predicting x_{t+1} given x_{1..t} is H[t] (logits at pos t)
                    "entropy": float(H[t]),
                })

    handle.remove()
    print(f"Collected {len(positions)} position records")

    # Convert to arrays
    aligns = np.array([r["align_abs"] for r in positions])
    Hs = np.array([r["entropy"] for r in positions])
    n_total = len(aligns)

    # Spearman correlation
    rho, pval = spearmanr(Hs, aligns)
    print(f"Spearman ρ(entropy, |h₂·v₁|) = {rho:.4f}  p={pval:.3e}")

    # Cohen's d: bottom-10% entropy vs rest
    thr = np.quantile(Hs, 0.10)
    low_mask = Hs <= thr
    high_mask = Hs > thr
    d_low_vs_rest = cohens_d(aligns[low_mask], aligns[high_mask])
    d_low_vs_top = cohens_d(aligns[low_mask], aligns[Hs >= np.quantile(Hs, 0.90)])
    print(f"Cohen's d (bottom-10% entropy vs rest) = {d_low_vs_rest:.4f}")
    print(f"Cohen's d (bottom-10% vs top-10% entropy) = {d_low_vs_top:.4f}")

    # Entropy decile stats
    deciles = np.quantile(Hs, np.linspace(0, 1, 11))
    decile_stats = []
    for i in range(10):
        mask = (Hs >= deciles[i]) & (Hs < deciles[i + 1]) if i < 9 else (Hs >= deciles[i])
        if mask.sum() > 0:
            decile_stats.append({
                "decile": i + 1,
                "entropy_range": [float(deciles[i]), float(deciles[i + 1])],
                "n": int(mask.sum()),
                "mean_align_abs": float(aligns[mask].mean()),
                "median_align_abs": float(np.median(aligns[mask])),
                "max_align_abs": float(aligns[mask].max()),
            })

    # Top-K alignment positions and their entropy percentile
    align_rank = np.argsort(-aligns)  # descending
    entropy_percentile = np.array([(Hs <= Hs[i]).mean() * 100 for i in range(len(Hs))])

    topK_records = {}
    for K in [50, 100, 200, 500]:
        top_idx = align_rank[:K]
        topK_records[str(K)] = {
            "median_entropy_percentile": float(np.median(entropy_percentile[top_idx])),
            "mean_entropy_percentile": float(entropy_percentile[top_idx].mean()),
            "frac_in_bottom_20pct_entropy": float((entropy_percentile[top_idx] <= 20).mean()),
            "frac_in_bottom_30pct_entropy": float((entropy_percentile[top_idx] <= 30).mean()),
            "frac_in_bottom_40pct_entropy": float((entropy_percentile[top_idx] <= 40).mean()),
            "frac_in_bottom_50pct_entropy": float((entropy_percentile[top_idx] <= 50).mean()),
        }

    out = {
        "model": args.model,
        "layer_id": args.layer_id,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "n_positions": n_total,
        "v1_top_singular": float(S2[0].item()),
        "v1_sigma_ratio": float(S2[0].item() / S2[1].item()),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "cohens_d_bottom10_vs_rest": float(d_low_vs_rest),
        "cohens_d_bottom10_vs_top10": float(d_low_vs_top),
        "decile_stats": decile_stats,
        "topK_entropy_distribution": topK_records,
    }

    fp = os.path.join(args.savedir, "exp5c_entropy_results.json")
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\nSaved summary: {fp}")

    # Save raw per-position records for later analysis (NPZ is compact for arrays)
    raw_fp = os.path.join(args.savedir, "exp5c_raw_positions.npz")
    np.savez_compressed(
        raw_fp,
        align_abs=aligns.astype(np.float32),
        entropy=Hs.astype(np.float32),
        token_id=np.array([r["token_id"] for r in positions], dtype=np.int32),
        seq=np.array([r["seq"] for r in positions], dtype=np.int32),
        pos=np.array([r["pos"] for r in positions], dtype=np.int32),
    )
    print(f"Saved raw: {raw_fp}  ({os.path.getsize(raw_fp)/1e6:.2f} MB)")

    # Also print top-K snippet for quick inspection
    print(f"\nTop-100 |h₂·v₁| positions entropy distribution:")
    print(f"  median percentile: {topK_records['100']['median_entropy_percentile']:.1f}%")
    print(f"  % in bottom-20 H: {100*topK_records['100']['frac_in_bottom_20pct_entropy']:.1f}%")
    print(f"  % in bottom-40 H: {100*topK_records['100']['frac_in_bottom_40pct_entropy']:.1f}%")


if __name__ == "__main__":
    main()
