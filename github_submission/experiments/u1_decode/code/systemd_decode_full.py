"""
Full dump of Top-K token SET (not just top-30 most common) for each model,
plus the hidden-dim sparsity of the u₁ direction (output side of W_down).

Output: /tmp/hc_v2/systemd_full_tokens.json

Keys per model:
    n_positions
    topK
    unique_tokens_count
    concentration_ratio  = topK / unique_count (lower = more dispersed)
    u1_top_dim_weight    = |u₁|_max (peak dimension coefficient)
    u1_top_dim_idx       = argmax_j |u₁[j]|
    u1_top5_dims         = top-5 hidden-dim indices with largest |u₁[j]|
    u1_top5_weights      = corresponding weights
    u1_entropy           = -Σ p_j log p_j where p_j = u₁[j]² / Σ u₁[j]²
    u1_effective_dim     = exp(u1_entropy)
    hidden_size          = dim of hidden state
    token_list           = [(text, count), ...]  full sorted by count desc
    top30_positions      = [(text, align_abs, entropy), ...] first 30 positions
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tokenizer_and_Wdown(model_name: str):
    """Load tokenizer + the origin-layer W_down matrix. Keeps RAM low."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from lib.model_dict import resolve_model_id
    model_id = resolve_model_id(model_name)
    trust = any(t in model_name.lower() for t in ("glm4", "falcon", "mpt", "phi"))
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust, use_fast=False)
    return tok, model_id, trust


L_ORIGIN = {
    "qwen3_0.6b": 2, "qwen3_1.7b": 0, "qwen3_4b": 0, "qwen3_8b": 2,
    "qwen3_14b": 6, "qwen3_32b": 6, "qwen2_7b": 3, "qwen2.5_7b": 3,
    "qwen1.5_14b": 3, "qwen3.5_9b": 22, "qwen3.5_27b": 9,
    "glm4_9b": 1, "llama3.1_8b": 1, "yi_9b": 8,
    # 2026-04-22: add glm4_32b + MoE models (layer IDs match master_runner.sh)
    "glm4_32b": 0, "qwen3_30b_a3b": 1, "qwen3.5_35b_a3b": 9,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/tmp/hc_v2")
    p.add_argument("--out", default="/tmp/hc_v2/systemd_full_tokens.json")
    p.add_argument("--topK", type=int, default=500)
    p.add_argument("--skip_Wdown", action="store_true",
                   help="Skip W_down SVD (tokenizer-only mode, fast)")
    args = p.parse_args()

    base = args.base
    models = []
    for d in sorted(os.listdir(base)):
        if not d.startswith("systemd_"):
            continue
        npz = os.path.join(base, d, "exp5c_raw_positions.npz")
        summary = os.path.join(base, d, "exp5c_entropy_results.json")
        if os.path.exists(npz):
            models.append((d[len("systemd_"):], npz, summary))

    print(f"Found {len(models)} models")
    all_res = {}

    for model, npz_path, summary_path in models:
        print(f"\n=== {model} ===")
        try:
            tok, model_id, trust = load_tokenizer_and_Wdown(model)
        except Exception as e:
            print(f"  tokenizer FAIL: {e}")
            continue

        z = np.load(npz_path)
        align = z["align_abs"].astype(np.float64)
        H = z["entropy"].astype(np.float64)
        token_ids = z["token_id"].astype(np.int64)
        valid = np.isfinite(align) & np.isfinite(H)
        align, H, token_ids = align[valid], H[valid], token_ids[valid]
        n = len(align)

        K = min(args.topK, n)
        order = np.argsort(-align)[:K]
        top_ids = token_ids[order]
        top_align = align[order]
        top_H = H[order]

        texts = [tok.decode([int(t)]) for t in top_ids]
        ctr = Counter(texts)
        token_list = ctr.most_common()
        unique_n = len(ctr)

        # u₁ sparsity: read from exp5c summary if possible, else compute
        u1_info = {}
        if not args.skip_Wdown:
            try:
                # Recompute u₁ from W_down of origin layer by quickly loading that layer only
                import gc
                from transformers import AutoModelForCausalLM, AutoConfig
                L = L_ORIGIN.get(model)
                if L is not None:
                    dtype = torch.bfloat16 if "glm4" in model else torch.float16
                    print(f"  loading model for u₁ extraction (L={L}) ...")
                    cache_dir = "./model_weights"
                    mdl = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=dtype, cache_dir=cache_dir,
                        low_cpu_mem_usage=True, device_map="cpu",
                        trust_remote_code=trust,
                    )
                    # find decoder layers
                    if hasattr(mdl, "model") and hasattr(mdl.model, "layers"):
                        layers = mdl.model.layers
                    elif hasattr(mdl, "transformer") and hasattr(mdl.transformer, "h"):
                        layers = mdl.transformer.h
                    elif hasattr(mdl, "transformer") and hasattr(mdl.transformer, "encoder"):
                        layers = mdl.transformer.encoder.layers
                    else:
                        layers = None
                    if layers is None:
                        raise RuntimeError("can't find layers")
                    block = layers[L]
                    # try common down-proj names
                    mlp = getattr(block, "mlp", getattr(block, "feed_forward", None))
                    dp = None
                    for name in ("down_proj", "dense_4h_to_h", "c_proj", "fc2", "w2"):
                        if hasattr(mlp, name):
                            dp = getattr(mlp, name)
                            break
                    if dp is None:
                        raise RuntimeError("can't find down_proj")
                    W2 = dp.weight.detach().cpu().float()  # [hidden, intermediate]
                    # SVD of W2^T gives U in intermediate, V in hidden. We want u₁ of W_down
                    # in HIDDEN space = V[:,0] (right singular vector of W2)
                    # W_down = W2: [hidden, intermediate]
                    # u₁ output side in hidden space = U_W2[:,0] = left singular of W2
                    U, S, Vt = torch.linalg.svd(W2, full_matrices=False)
                    u1 = U[:, 0].numpy()  # [hidden]
                    h_sz = W2.shape[0]
                    abs_u = np.abs(u1)
                    # sparsity metrics
                    p = (u1 ** 2)
                    p = p / p.sum()
                    p_nz = p[p > 0]
                    H_u = float(-(p_nz * np.log(p_nz)).sum())
                    eff_dim = float(np.exp(H_u))
                    top5_idx = np.argsort(-abs_u)[:5]
                    u1_info = {
                        "hidden_size": int(h_sz),
                        "u1_top_dim_weight": float(abs_u.max()),
                        "u1_top_dim_idx": int(top5_idx[0]),
                        "u1_top5_dims": top5_idx.tolist(),
                        "u1_top5_weights": [float(abs_u[i]) for i in top5_idx],
                        "u1_entropy_bits": H_u / np.log(2),
                        "u1_effective_dim": eff_dim,
                        "u1_sparsity_pct_top1": float(p[top5_idx[0]] * 100),
                        "u1_sparsity_pct_top5": float(p[top5_idx].sum() * 100),
                        "singular_top": float(S[0].item()),
                        "sigma_ratio": float(S[0].item() / S[1].item()),
                    }
                    del mdl, W2, U, Vt, u1
                    gc.collect()
                    print(f"  u₁ top dim {u1_info['u1_top_dim_idx']} weight={u1_info['u1_top_dim_weight']:.4f} "
                          f"eff_dim={eff_dim:.1f}/{h_sz} ({eff_dim/h_sz*100:.1f}%)")
            except Exception as e:
                print(f"  u₁ FAIL: {e}")

        rec = {
            "n_positions": int(n),
            "topK": K,
            "unique_tokens_count": unique_n,
            "concentration_ratio": unique_n / K,
            "token_list_full": [[t, c] for t, c in token_list],
            "top30_positions": [
                {"text": texts[i], "align_abs": float(top_align[i]), "entropy": float(top_H[i])}
                for i in range(min(30, K))
            ],
            **u1_info,
        }
        all_res[model] = rec
        print(f"  unique Top-{K}: {unique_n} ({unique_n / K * 100:.0f}% concentration)")
        print(f"  top10 tokens: {token_list[:10]}")

    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
