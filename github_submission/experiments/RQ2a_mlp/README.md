# RQ2a - MLP contribution (all-MLP disable)

## Purpose

Verify hypothesis H1: "MLP is the MA creator". If disabling every MLP collapses MA,
then MLP is the origin.

## Method

Forward-hook every MLP block to output zeros; otherwise identical to RQ1.

## Key metric

- `retain_% = disabled_top1 / baseline_top1 * 100` (lower = MLP is the origin).

## How to reproduce

```bash
python code/exp2a_mlp_feasibility_test.py \
    --model_name gptj_6b \
    --nsamples 30 --seqlen 2048 \
    --output_dir results/gptj_6b/
```

For MoE models, use `code/exp2a_mlp_feasibility_test_fixed.py` (handles the
`SparseMoeBlock` tuple return).

## Key findings

- 20 / 24 dense models: `retain <= 10%`. `bloom_7b1` retains 0 % (strongest support).
- 4 anomalies (retain > 15 %): `qwen3.5_35b_a3b` (81 %, MoE artifact), `gpt2` (39
  %, old arch), `qwen3.5_9b` (32 %), `qwen3.5_27b` (20 %). The entire qwen3.5 dense
  family retains > 15 %, hinting at a non-MLP auxiliary source.
- The companion RQ2c script (`code/exp2c_mlp_internal_analysis.py`) performs
  progressive layer ablation to categorize models into CONCENTRATED / FEW-SOURCE /
  DISPERSED and identify the **origin layer** used for all downstream RQ3/4/5.

## Result layout per model

```
results/<model>/
├── baseline/results.json
├── all_mlp_disabled/results.json
└── comparison/
    ├── EXPERIMENT_2A_SUMMARY.txt
    └── 4 diagnostic PNGs
```

For models without raw per-model dirs, see
`aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json -> <model> -> exp2a`.
