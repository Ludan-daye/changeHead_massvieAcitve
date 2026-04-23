"""
Decode the Top-K |h2.v1| token positions per model back to text tokens,
and look at what characterizes them empirically (no pre-baked hypothesis).

Input: /tmp/hc_v2/systemd_<model>/exp5c_raw_positions.npz
Output: /tmp/hc_v2/topK_tokens.json  — per-model Top-K token text + counts

Run on primary server (117.50.223.194) where tokenizers are cached.
Uses the same load_model args-shim as exp5c_entropy.py but only loads tokenizer.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tokenizer_only(model_name: str):
    """Load only the tokenizer (no model weights) for a model name."""
    from transformers import AutoTokenizer
    from lib.model_dict import resolve_model_id
    model_id = resolve_model_id(model_name)
    trust = any(t in model_name.lower() for t in ("glm4", "falcon", "mpt", "phi"))
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust, use_fast=False)


def analyze_model(model: str, npz_path: str, tokenizer, topK: int = 200) -> dict:
    z = np.load(npz_path)
    align = z["align_abs"].astype(np.float64)
    H = z["entropy"].astype(np.float64)
    token_ids = z["token_id"].astype(np.int64)
    valid = np.isfinite(align) & np.isfinite(H)
    align = align[valid]
    H = H[valid]
    token_ids = token_ids[valid]
    n = len(align)

    order = np.argsort(-align)[:topK]
    top_ids = token_ids[order]
    top_align = align[order]
    top_H = H[order]

    # decode each
    top_text = [tokenizer.decode([int(t)]) for t in top_ids]

    # token frequency
    ctr = Counter(top_text)
    top_freq = ctr.most_common(30)

    # Empirically characterize WITHOUT pre-baked categories
    # - Length distribution
    lengths = [len(t) for t in top_text]
    # - Whitespace-starts
    ws_start = sum(1 for t in top_text if t.startswith(" "))
    # - Pure punctuation/space
    struct = sum(1 for t in top_text if t.strip() in {
        ".", ",", "!", "?", ";", ":", '"', "'", "-", "—",
        "(", ")", "[", "]", "{", "}", "@", "#", "$", "%", "&", "*",
        "=", "/", "\\", "|", "<", ">", "~", "`", "+"
    } or "\n" in t or "\t" in t or t == "" or t == " ")
    # - Single-char alphabetic (likely BPE subword)
    subword = sum(1 for t in top_text if len(t.strip()) <= 2 and t.strip().isalpha())
    # - Starts with uppercase (often proper noun)
    upper = sum(1 for t in top_text if t.strip() and t.strip()[0].isupper() and not t.strip()[0].isdigit())
    # - Digit-containing
    digit = sum(1 for t in top_text if any(c.isdigit() for c in t))
    # - Has non-ASCII
    nonascii = sum(1 for t in top_text if any(ord(c) > 127 for c in t))

    return {
        "model": model,
        "n_total": int(n),
        "topK": topK,
        "top_tokens_unique": len(set(top_text)),
        "top_tokens_freq": top_freq,
        "categories": {
            "struct_punct_whitespace": struct,
            "subword_1_2_char_alpha": subword,
            "starts_with_uppercase": upper,
            "contains_digit": digit,
            "non_ascii": nonascii,
            "whitespace_start": ws_start,
        },
        "length_stats": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "min": min(lengths),
            "max": max(lengths),
        },
        "top30_samples": top_text[:30],
        "top30_align": [float(a) for a in top_align[:30]],
        "top30_entropy": [float(h) for h in top_H[:30]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/tmp/hc_v2")
    parser.add_argument("--out", default="/tmp/hc_v2/topK_tokens.json")
    parser.add_argument("--topK", type=int, default=200)
    args = parser.parse_args()

    base = args.base
    models = []
    for d in sorted(os.listdir(base)):
        if not d.startswith("systemd_"):
            continue
        npz = os.path.join(base, d, "exp5c_raw_positions.npz")
        if os.path.exists(npz):
            models.append((d[len("systemd_"):], npz))

    print(f"Found {len(models)} models")
    all_results = {}
    for model, npz in models:
        print(f"\n=== {model} ===")
        try:
            tok = load_tokenizer_only(model)
        except Exception as e:
            print(f"  tokenizer FAIL: {e}")
            continue
        try:
            r = analyze_model(model, npz, tok, topK=args.topK)
        except Exception as e:
            print(f"  analyze FAIL: {e}")
            continue
        all_results[model] = r
        print(f"  unique Top-{args.topK}: {r['top_tokens_unique']}")
        print(f"  top-5 most frequent: {r['top_tokens_freq'][:5]}")
        print(f"  categories: {r['categories']}")
        print(f"  samples: {r['top30_samples'][:10]}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
