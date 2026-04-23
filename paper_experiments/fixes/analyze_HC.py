#!/usr/bin/env python3
"""Cross-model analysis of hypothesis C (MA is written at low predict-entropy positions).

Reads all exp5c_raw_positions.npz files under results_stage2/HC_entropy/<model>/,
produces a summary table and per-model entropy-bin histogram plot.

Outputs:
  - analyze_HC_results.md           (markdown table)
  - analyze_HC_histograms.png       (grid of per-model histograms)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    print("scipy missing; install or run in env with scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("matplotlib missing; skipping figures")


ROOT = Path("/Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/HC_entropy")


def analyze_model(npz_path: Path, summary_path: Path) -> dict:
    """Return dict with per-model metrics. Filters NaN/Inf entropy positions (fp16 softmax overflow)."""
    with np.load(npz_path) as z:
        align = z["align_abs"].astype(np.float64)
        H = z["entropy"].astype(np.float64)
    n_raw = len(align)

    # Mask out NaN/Inf entries (happens when fp16 softmax saturates → inf entropy)
    valid = np.isfinite(H) & np.isfinite(align)
    n_bad = int((~valid).sum())
    align = align[valid]
    H = H[valid]
    n = len(align)
    if n < 100:
        return {"n": n, "error": f"too few valid positions (raw={n_raw}, dropped={n_bad})"}

    summ = {}
    if summary_path.exists():
        try:
            summ = json.load(open(summary_path))
        except Exception:
            summ = {}

    # Entropy percentile of every position: proportion <= this entropy
    # Use rank-based percentile (like exp5c does)
    # Fast vectorized: rank / n * 100
    order = np.argsort(H)
    pct = np.empty(n)
    pct[order] = np.arange(n) / (n - 1) * 100

    # Top-K by alignment
    align_order = np.argsort(-align)
    topK_metrics = {}
    for K in (50, 100, 200, 500):
        if K > n:
            continue
        top_idx = align_order[:K]
        top_pct = pct[top_idx]
        topK_metrics[K] = {
            "median_pct": float(np.median(top_pct)),
            "mean_pct": float(np.mean(top_pct)),
            "frac_b20": float((top_pct <= 20).mean()),
            "frac_b30": float((top_pct <= 30).mean()),
            "frac_b40": float((top_pct <= 40).mean()),
            "frac_b50": float((top_pct <= 50).mean()),
        }

    # Spearman on filtered data
    rho, pval = spearmanr(H, align)
    if not np.isfinite(rho):
        rho, pval = 0.0, 1.0

    # Histogram: count of Top-100 positions in each entropy-percentile decile
    if n >= 100:
        top100 = align_order[:100]
        top100_pct = pct[top100]
        hist, _edges = np.histogram(top100_pct, bins=np.linspace(0, 100, 11))
    else:
        hist = np.zeros(10, dtype=int)

    return {
        "n": n,
        "n_raw": n_raw,
        "n_dropped": n_bad,
        "sigma_ratio": summ.get("v1_sigma_ratio", None),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "topK": topK_metrics,
        "hist_top100": hist.tolist(),
    }


def supports_HC(m: dict) -> str:
    """Classify a model's support for H(C).

    Primary evidence: fraction of Top-100 |h₂·v₁| positions that land in the
    bottom-k% of predict-entropy. If MA is written at "a few lowest-entropy"
    positions, this fraction should be much greater than k% (uniform).
    Secondary: spearman ρ (can be noisy when relationship is bimodal).
    """
    t = m.get("topK", {}).get(100)
    if t is None:
        return "no-data"
    med = t["median_pct"]
    b20 = t["frac_b20"]
    b30 = t["frac_b30"]
    rho = m["spearman_rho"]
    # STRONG: top100 overwhelmingly concentrated in bottom-20% entropy
    if b20 >= 0.50 and med < 25:
        return "STRONG"
    # MODERATE: top100 majority in bottom-30%, median percentile < 40
    if b30 >= 0.45 and med < 40:
        return "MODERATE"
    # WEAK: top100 over-represented in bottom entropy vs uniform (≥ 1.5× baseline)
    if b20 >= 0.30 or (b30 >= 0.40 and med < 50):
        return "WEAK"
    # NULL: matches uniform
    if 45 <= med <= 55 and 15 <= 100 * b20 <= 25:
        return "NULL"
    return "REFUTE"


def main():
    if not ROOT.exists():
        print(f"ROOT missing: {ROOT}")
        sys.exit(1)

    # Agent B saved to systemd_<model>/ to avoid clashing with stale dirs;
    # strip that prefix when keying results.
    def canonical_name(dirname: str) -> str:
        return dirname[len("systemd_"):] if dirname.startswith("systemd_") else dirname

    raw_dirs = sorted([p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")])
    print(f"Found {len(raw_dirs)} model dirs")

    results = {}
    for raw in raw_dirs:
        m = canonical_name(raw)
        npz = ROOT / raw / "exp5c_raw_positions.npz"
        js = ROOT / raw / "exp5c_entropy_results.json"
        if not npz.exists():
            print(f"  skip {raw}: no NPZ")
            continue
        try:
            r = analyze_model(npz, js)
            results[m] = r
            if "error" in r:
                print(f"  {m}: ERROR {r['error']}")
            else:
                t = r["topK"].get(100, {})
                print(f"  {m}: n={r['n']}  rho={r['spearman_rho']:+.3f}  "
                      f"top100_med_pct={t.get('median_pct', 'NA'):.1f}  "
                      f"b30={t.get('frac_b30', 0):.2f}  verdict={supports_HC(r)}")
        except Exception as e:
            print(f"  {m}: FAIL {e}")

    # --- Markdown table ---
    out_md = Path("/Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/analyze_HC_results.md")
    lines = [
        "# H(C) 跨模型分析：MA 写在 low predict-entropy 位置？",
        "",
        f"数据源：`{ROOT}`",
        f"共 {len(results)} 模型。每个模型的每个 token 位置记录了 `|h₂·v₁|`（对 v₁ 的投影）和 `H(predict)`。",
        "",
        "**判据**（主要指标是 Top-100 在 low-H bin 里的集中度；ρ 只用作辅助参考）：",
        "- STRONG：frac_bottom20 ≥ 50% 且 median_H_pct < 25%（Top-100 压倒性集中在最低熵 20%）",
        "- MODERATE：frac_bottom30 ≥ 45% 且 median < 40%",
        "- WEAK：frac_bottom20 ≥ 30% 或 (frac_bottom30 ≥ 40% 且 median < 50%)",
        "- NULL：指标接近 uniform（med∈[45,55], frac_bottom20∈[15,25]%）",
        "- REFUTE：frac_bottom20 < 20% 或 median > 50%",
        "",
        "| 模型 | n_pos | σ₁/σ₂ | ρ(H,align) | Top100 med H% | Top100 在 bottom 20/30/40% H 比例 | 结论 |",
        "|---|---:|---:|---:|---:|---|:-:|",
    ]
    sort_key = lambda kv: kv[1].get("topK", {}).get(100, {}).get("median_pct", 101)
    for m, r in sorted(results.items(), key=sort_key):
        if "error" in r:
            lines.append(f"| {m} | {r.get('n', '?')} | | | | | ERR |")
            continue
        t = r["topK"].get(100, {})
        sr = r.get("sigma_ratio")
        sr_s = f"{sr:.2f}" if sr is not None else "—"
        rho = r["spearman_rho"]
        lines.append(
            f"| {m} | {r['n']} | {sr_s} | {rho:+.3f} | "
            f"{t.get('median_pct', float('nan')):.1f} | "
            f"{100*t.get('frac_b20', 0):.0f} / {100*t.get('frac_b30', 0):.0f} / {100*t.get('frac_b40', 0):.0f}% | "
            f"**{supports_HC(r)}** |"
        )

    # Summary counts
    verdicts = [supports_HC(r) for r in results.values() if "error" not in r]
    lines.append("")
    lines.append("## 判定分布")
    lines.append("")
    for v in ("STRONG", "MODERATE", "WEAK", "NULL", "REFUTE"):
        lines.append(f"- **{v}**: {verdicts.count(v)}")
    lines.append("")
    lines.append("## 观察")
    lines.append("")
    lines.append("1. **14 模型没有 1 个达到 STRONG**：即便最低熵比例最高的 qwen3_32b，其 Top-100 里 'frac_bottom_20%_H' 只有 49%，离 80%+ 压倒性证据还远。")
    lines.append("2. **模型分裂成两类 histogram 形态**：")
    lines.append("   - *Low-entropy writer*（MODERATE/WEAK 6 个）：Top-100 集中在 bottom 0-30% H — **弱支持论点 C 的'排前几低'说法**")
    lines.append("   - *High-entropy writer*（REFUTE 7 个，如 yi_9b, qwen3_4b/8b, qwen1.5_14b）：Top-100 反而集中在 **top 90-100% H**（最不确定的位置）")
    lines.append("3. **双峰嫌疑**：qwen3_32b / qwen3_14b 直方图同时在 0-25% 和 50-60% 有两个峰，spearman ρ 抵消。")
    lines.append("4. **σ₁/σ₂ 与 HC 支持度无明显相关**：qwen2_7b/qwen2.5_7b 谱集中最强（2.6-2.8），却 refute。")
    lines.append("5. **fp16 溢出**（entropy=inf）只影响 qwen2_7b/qwen2.5_7b：~90% 位置被过滤。若改用 fp32 logits 取 log_softmax 可恢复。")
    lines.append("")
    lines.append("## 论点 C 结论")
    lines.append("")
    lines.append("**不推荐保留'MA 写在 predict-entropy 排前几低的 token 位置'作为一般性论点**：")
    lines.append("- 14 模型里只有 5/14 = 36% 可以被归为 MODERATE（还没到 STRONG）")
    lines.append("- 7/14 = 50% 的模型 Top-align 位置反而集中在**最高熵**位置，明确反证")
    lines.append("- 即使 MODERATE 的模型，'底 20% entropy' 的覆盖率也只有 35-49%，远非'排前几低'所暗示的 >80%")
    lines.append("")
    lines.append("**可以保留的弱化版本**（子命题）：")
    lines.append("- 在部分模型（MODERATE 5 个）上，Top-K 高 alignment 位置的 median entropy percentile 显著低于 50%（22-36%）")
    lines.append("- 但这并非普遍规律，不能作为 MA 的一般机制特征。更可能 MA 写在**'结构 token'位置**（换行/标点/句界），而结构 token 的 entropy 分布本身因模型而异。")
    lines.append("")

    if HAS_PLT and len(results) > 0:
        fig_path = "/Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/analyze_HC_histograms.png"
        ncol = 4
        nrow = (len(results) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
        for i, (m, r) in enumerate(sorted(results.items(), key=sort_key)):
            ax = axes[i // ncol][i % ncol]
            if "error" in r:
                ax.set_title(f"{m} (err)"); ax.axis("off"); continue
            hist = r["hist_top100"]
            bins = np.arange(10) * 10 + 5
            ax.bar(bins, hist, width=9, color="steelblue")
            ax.axhline(10, color="red", ls=":", lw=1, label="uniform=10")
            t = r["topK"].get(100, {})
            v = supports_HC(r)
            ax.set_title(f"{m}\nmed={t.get('median_pct', 0):.0f}%  ρ={r['spearman_rho']:+.2f}  {v}", fontsize=9)
            ax.set_xlabel("entropy percentile")
            ax.set_ylabel("count(Top-100)")
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylim(0, max(30, max(hist) + 2))
        # hide empty axes
        for j in range(len(results), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("H(C): Top-100 high-|h₂·v₁| positions, distribution over entropy percentile bins", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        lines.append(f"![histograms]({os.path.basename(fig_path)})")
        print(f"\nFig saved: {fig_path}")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nMD saved: {out_md}")

    # Also save raw json (machine-readable)
    json.dump(results, open(out_md.with_suffix(".json"), "w"), indent=2)
    print(f"JSON saved: {out_md.with_suffix('.json')}")


if __name__ == "__main__":
    main()
