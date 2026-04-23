# RQ1 - Attention contribution

## Purpose

Falsify the hypothesis H0: "attention is the origin of Massive Activations".
Measure whether MA persists when every attention layer is hooked to output zeros.

## Method

For each model:

1. Baseline: run WikiText-2 through the unmodified network, collect the top-1
   activation magnitude per token per layer (`baseline/results.json`).
2. Intervention: register a forward-hook on every attention module that replaces
   the attention output by zeros. Run the same data
   (`all_heads_disabled/results.json`).
3. Compare: `comparison/EXPERIMENT_1_SUMMARY.txt` + 4 diagnostic PNGs + top-1 /
   percentage-change / heatmap plots.

## Key metrics

- `residual_% = disabled_top1 / baseline_top1 * 100` -- if > 0, MA has a
  non-attention origin.
- `Delta MA % = (disabled - baseline) / baseline * 100` -- sign classifies models
  as *generative* (<0) vs *suppressive* (>0).

## How to reproduce

```bash
python code/exp1_feasibility_test.py \
    --model_name gptj_6b \
    --nsamples 30 \
    --seqlen 2048 \
    --output_dir results/gptj_6b/
```

## Key findings (26 models)

- **H0 falsified**: 25 / 25 models with complete data retain non-zero MA after
  attention disable. Minimum residual = **1.69 %** (`gptj_6b`).
- Generative (Delta MA < 0): 17 models. Attention = amplifier / broadcaster.
- Suppressive (Delta MA > 0): 8 models. Attention = steady-state regulator.
- Same family can flip direction with size: `qwen2.5_0.5b` generative,
  `qwen2.5_7b` suppressive; `glm4_9b` generative, `glm4_32b` suppressive.

See `../../CONCLUSIONS.md` for the full synthesis.

## Result layout per model

```
results/<model>/
├── baseline/results.json
├── all_heads_disabled/results.json
├── comparison/
│   ├── EXPERIMENT_1_SUMMARY.txt
│   ├── exp1_top1_comparison.png
│   ├── exp1_critical_dimensions.png
│   ├── exp1_percentage_change_heatmap.png
│   └── exp1_layerwise_breakdown.png
└── table1_rq1.json
```

For 15 models (mostly Qwen3 and recent releases) only the aggregated metrics in
`aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json` are available. A `MISSING.txt` file is
present in those per-model dirs with a pointer.
