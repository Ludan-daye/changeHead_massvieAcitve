# Aggregated results

## Files

### `ALL_EXPERIMENTS_SUMMARY_v2.json`

Consolidated metrics for all 26 models × 6 research questions.

Top-level keys: model ids (29 total; 3 placeholders `deepseek_v2_lite`,
`qwen2.5_0.5b_optimized`, `qwen2.5_7b_old_nan` are kept for legacy reasons but
have no real data).

Per-model sub-keys:

| Key        | Content                                          |
|------------|--------------------------------------------------|
| `model`    | canonical model id (string) |
| `exp1`     | RQ1 - attention disabled: `baseline_top1`, `disabled_top1`, `residual_pct`, `delta_ma_pct`, `generative_or_suppressive` |
| `exp2`     | RQ2b - per-layer MLP disable (critical_layer) |
| `exp2a`    | RQ2a - all MLP disabled: `baseline_top1`, `disabled_top1`, `retain_pct` |
| `exp2c`    | RQ2c - greedy progressive MLP ablation: `category`, `l_origin_from_step1`, `final_disabled_set` |
| `exp3`     | RQ3 - function-word v1 projection (peak layer, legacy) |
| `exp3_origin` | RQ3 - function-word v1 projection (origin layer, re-run) |
| `exp4`     | RQ4 - SVD alignment (peak layer, legacy) |
| `exp4_origin` | RQ4 - SVD alignment (origin layer, re-run) |
| `exp5`     | RQ5 - V replacement single-layer |
| `exp5_origin` | RQ5 - single-layer at origin layer |
| `exp5b`    | RQ5 - macro v1 projection-out |
| `exp6`     | RQ6 - top-K delete/retain; macro-SVD |
| `exp7`     | RQ7 - training dynamics (Pythia; out of scope) |

### `ALL_26_u1_combined.json`

Per model: top-100 tokens aligned with `+u1` and `-u1` of the origin layer's
`W_down`. Keys are the 26 canonical model ids; values contain `top_pos`,
`top_neg`, `u1_vector`, `layer_id`.

## Using the aggregated JSON in analysis

```python
import json

with open("aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json") as f:
    R = json.load(f)

# RQ1 residual percentage across all models
for m, d in R.items():
    if d.get("exp1") and "residual_pct" in d["exp1"]:
        print(f"{m}: residual = {d['exp1']['residual_pct']:.2f}%")
```

For models where per-model raw result directories are not present
(RQ1/RQ2a: 15 of 26), this aggregated JSON is the only source. See each
experiment's `results/<model>/MISSING.txt` for pointers.
