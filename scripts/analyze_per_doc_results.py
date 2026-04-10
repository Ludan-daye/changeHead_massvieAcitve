"""ACL 2026 review issue 2: per-document independent sampling analysis.

Reads result JSONs produced by the remote batch run (exp1/exp3/exp6 across 8
paper models with 256 independent C4 documents) and produces:

  results_per_doc/per_doc_summary.json   -- machine-readable summary
  paper/new_table1.md                    -- markdown rows for the camera-ready Table 1
  results_per_doc/old_vs_new_comparison.md
  figures/exp_perdoc/{model}_top1.png    -- distribution histograms

Run after `scp -r vicuna@<host>:.../results/per_doc /Users/a1-6/.../results_per_doc/`.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_per_doc" / "per_doc"
SUMMARY_PATH = ROOT / "results_per_doc" / "per_doc_summary.json"
TABLE1_PATH = ROOT / "paper" / "new_table1.md"
COMPARE_PATH = ROOT / "results_per_doc" / "old_vs_new_comparison.md"
FIG_DIR = ROOT / "figures" / "exp_perdoc"

# Paper-reported numbers (from current Table 1 in Massive activation.tex), used for sanity gate.
# (model_key, eta, baseline_top1_critical_layer, delta_v_pct)
OLD_TABLE = {
    "gpt2":            {"eta": 2.52, "delta_v": -70.7, "name": "GPT-2"},
    "opt_7b":          {"eta": 2.87, "delta_v": -77.3, "name": "OPT-6.7B"},
    "qwen2.5_7b":      {"eta": 2.64, "delta_v": -99.1, "name": "Qwen2.5-7B"},
    "gptj_6b":         {"eta": 1.91, "delta_v": -69.6, "name": "GPT-J-6B"},
    "llama2_13b":      {"eta": 2.23, "delta_v": -78.8, "name": "LLaMA-2-13B"},
    "mistral_7b_v03":  {"eta": 1.85, "delta_v": -73.5, "name": "Mistral-7B"},
    "bloom_7b1":       {"eta": 1.18, "delta_v": -69.6, "name": "BLOOM-7B1"},
    "falcon_7b_local": {"eta": None, "delta_v": -75.9, "name": "Falcon-7B"},
}

MODELS = list(OLD_TABLE.keys())


def bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=42):
    """Percentile bootstrap CI for the mean of a 1-D array."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(means, (1 + confidence) / 2 * 100))
    return float(arr.mean()), lo, hi


def load_exp1(model):
    """Load per-layer top1 baseline + disabled and the per-document arrays."""
    base = RESULTS_DIR / "exp1" / model
    bp = base / "baseline" / "results.json"
    dp = base / "all_heads_disabled" / "results.json"
    if not (bp.exists() and dp.exists()):
        return None
    with open(bp) as f:
        baseline = json.load(f)
    with open(dp) as f:
        disabled = json.load(f)
    meta_path = base / "sampling_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    # Find the layer with the largest baseline top1_mean (= "critical layer" by exp1)
    best_layer = max(
        baseline.keys(),
        key=lambda k: float(baseline[k].get("top1_mean", float("-inf"))),
    )
    base_layer = baseline[best_layer]
    dis_layer = disabled.get(best_layer, {})

    base_top1 = base_layer.get("top1_per_sample") or []
    dis_top1 = dis_layer.get("top1_per_sample") or []
    n = min(len(base_top1), len(dis_top1))
    delta_pct = []
    for i in range(n):
        b = base_top1[i]
        d = dis_top1[i]
        if b > 0:
            delta_pct.append((d - b) / b * 100.0)
    base_mean, base_lo, base_hi = bootstrap_ci(base_top1) if base_top1 else (float("nan"),) * 3
    delta_mean, delta_lo, delta_hi = bootstrap_ci(delta_pct) if delta_pct else (float("nan"),) * 3
    return {
        "critical_layer_exp1": int(best_layer),
        "n_docs": len(base_top1),
        "baseline_top1_mean": base_mean,
        "baseline_top1_ci95": [base_lo, base_hi],
        "delta_head_mean_pct": delta_mean,
        "delta_head_ci95": [delta_lo, delta_hi],
        "sampling_meta": meta,
    }


def load_exp6(model):
    base = RESULTS_DIR / "exp6" / model
    f = base / "v_ablation_results.json"
    if not f.exists():
        return None
    with open(f) as fh:
        r = json.load(fh)
    baseline_vals = (r.get("baseline") or {}).get("top1_values") or []
    base_mean, base_lo, base_hi = bootstrap_ci(baseline_vals) if baseline_vals else (float("nan"),) * 3

    # Pick the most aggressive remove_top_k and the keep_top_1 entries (closest to paper's setup).
    remove_blocks = (r.get("ablation_results") or {}).get("remove_top_k") or {}
    keep_blocks = (r.get("ablation_results") or {}).get("keep_top_k") or {}

    # use largest k for remove (~ "destroy V structure")
    if remove_blocks:
        kmax = max(remove_blocks.keys(), key=lambda x: int(x))
        rm = remove_blocks[kmax]
        rm_vals = rm.get("top1_values") or []
        delta_remove = []
        for i, b in enumerate(baseline_vals):
            if b > 0 and i < len(rm_vals):
                delta_remove.append((rm_vals[i] - b) / b * 100.0)
        rm_mean, rm_lo, rm_hi = bootstrap_ci(delta_remove) if delta_remove else (float("nan"),) * 3
    else:
        kmax, rm_mean, rm_lo, rm_hi = None, float("nan"), float("nan"), float("nan")

    # keep_top_1 = "use only first singular direction"
    keep1 = keep_blocks.get("1") or {}
    keep1_vals = keep1.get("top1_values") or []
    delta_keep1 = []
    for i, b in enumerate(baseline_vals):
        if b > 0 and i < len(keep1_vals):
            delta_keep1.append((keep1_vals[i] - b) / b * 100.0)
    k1_mean, k1_lo, k1_hi = bootstrap_ci(delta_keep1) if delta_keep1 else (float("nan"),) * 3

    return {
        "critical_layer_exp6": r.get("critical_layer"),
        "n_docs": len(baseline_vals),
        "baseline_top1_mean": base_mean,
        "baseline_top1_ci95": [base_lo, base_hi],
        "sigma_ratio_eta": r.get("sigma_ratio"),
        "remove_top_k_max": kmax,
        "delta_v_remove_max_mean_pct": rm_mean,
        "delta_v_remove_max_ci95": [rm_lo, rm_hi],
        "delta_v_keep_top1_mean_pct": k1_mean,
        "delta_v_keep_top1_ci95": [k1_lo, k1_hi],
        "sampling_meta": r.get("sampling_meta"),
    }


def load_exp3(model):
    base = RESULTS_DIR / "exp3" / model
    if not base.exists():
        return None
    candidates = sorted(base.glob("layer_*/exp3_detailed_results.json"))
    if not candidates:
        return None
    f = candidates[0]
    with open(f) as fh:
        r = json.load(fh)
    stats = r.get("statistics") or {}
    svd = r.get("svd_results") or {}
    reg = r.get("regression") or {}
    return {
        "layer_id": r.get("layer_id"),
        "sigma_ratio_eta": svd.get("sigma_ratio"),
        "function_alignment_mean": stats.get("function_alignment_mean"),
        "function_alignment_std": stats.get("function_alignment_std"),
        "content_alignment_mean": stats.get("content_alignment_mean"),
        "content_alignment_std": stats.get("content_alignment_std"),
        "function_trigger_rate": stats.get("function_trigger_rate"),
        "content_trigger_rate": stats.get("content_trigger_rate"),
        "regression_r_squared": reg.get("r_squared"),
        "regression_p_value": reg.get("p_value"),
        "n_function_tokens": stats.get("function_count"),
        "n_content_tokens": stats.get("content_count"),
        "sampling_meta": r.get("sampling_meta"),
    }


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def fmt_ci(lo, hi, digits=1):
    if any(v is None or math.isnan(v) for v in (lo, hi)):
        return "—"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    summary = {}
    for model in MODELS:
        summary[model] = {
            "name": OLD_TABLE[model]["name"],
            "exp1": load_exp1(model),
            "exp3": load_exp3(model),
            "exp6": load_exp6(model),
        }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {SUMMARY_PATH}")

    # ---- Markdown table for paper ----
    rows = ["| Model | η (σ₁/σ₂) | Top1 baseline | Δhead-ablation % | ΔV-keep_top1 % | ΔV-remove_max % |",
            "|---|---|---|---|---|---|"]
    for m in MODELS:
        e1 = summary[m]["exp1"]
        e6 = summary[m]["exp6"]
        eta = (e6 or {}).get("sigma_ratio_eta")
        base_top1 = (e1 or {}).get("baseline_top1_mean") or (e6 or {}).get("baseline_top1_mean")
        d_head = (e1 or {}).get("delta_head_mean_pct")
        d_keep = (e6 or {}).get("delta_v_keep_top1_mean_pct")
        d_rmv = (e6 or {}).get("delta_v_remove_max_mean_pct")
        rows.append(
            f"| {summary[m]['name']} | {fmt(eta)} | {fmt(base_top1)} | {fmt(d_head)} | {fmt(d_keep)} | {fmt(d_rmv)} |"
        )
    TABLE1_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE1_PATH, "w") as f:
        f.write("# Per-document independent sampling — new Table 1 numbers\n\n")
        f.write("Protocol: 256 independent C4 validation documents × 256 tokens, seed=42.\n\n")
        f.write("\n".join(rows) + "\n")
    print(f"wrote {TABLE1_PATH}")

    # ---- Old vs new comparison ----
    cmp_rows = ["| Model | η old → new | ΔV old (paper) | ΔV new (remove_max) | abs delta |",
                "|---|---|---|---|---|"]
    sanity_flags = []
    for m in MODELS:
        e6 = summary[m]["exp6"] or {}
        eta_new = e6.get("sigma_ratio_eta")
        d_new = e6.get("delta_v_remove_max_mean_pct")
        d_old = OLD_TABLE[m]["delta_v"]
        eta_old = OLD_TABLE[m]["eta"]
        if d_new is not None and d_old is not None and not math.isnan(d_new):
            diff = abs(d_new - d_old)
            cmp_rows.append(
                f"| {OLD_TABLE[m]['name']} | {fmt(eta_old)} → {fmt(eta_new)} | {fmt(d_old, 1)}% | {fmt(d_new, 1)}% | {fmt(diff, 1)} pp |"
            )
            if diff > 15:
                sanity_flags.append((m, diff))
        else:
            cmp_rows.append(f"| {OLD_TABLE[m]['name']} | — | — | — | — |")
    with open(COMPARE_PATH, "w") as f:
        f.write("# Old vs new ΔV comparison (sanity gate)\n\n")
        f.write("\n".join(cmp_rows) + "\n\n")
        if sanity_flags:
            f.write("## ⚠️  Sanity gate WARNING\n\n")
            for m, diff in sanity_flags:
                f.write(f"- {m}: |Δ| = {diff:.1f} percentage points (>15)\n")
            f.write("\nDo NOT update the paper before reviewing these.\n")
        else:
            f.write("## ✅ Sanity gate OK\n\nAll deltas within ±15 pp of paper values.\n")
    print(f"wrote {COMPARE_PATH}")

    if sanity_flags:
        print("\n⚠️  SANITY GATE WARNING (>15 pp deviation):")
        for m, diff in sanity_flags:
            print(f"   {m}: {diff:.1f} pp")
        print("Inspect old_vs_new_comparison.md before touching the paper.")
    else:
        print("\n✅ Sanity gate passed; all ΔV deviations < 15 pp.")


if __name__ == "__main__":
    main()
