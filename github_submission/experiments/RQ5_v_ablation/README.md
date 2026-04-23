# RQ5 - Causal V ablation

## Purpose

Causally test whether `v1` (top right singular vector of `W_down`) is load-bearing
for MA by intervening on it.

## Variants

- `exp5_v_ablation.py` (single-layer): replace `v1`-direction of `W_down[L_origin]`
  with a random orthogonal vector. Suitable for Pattern A models.
- `exp5_macro_v_ablation.py` (multi-layer): collect `Delta h_macro` across origin
  layers, SVD to get macro-v1, then project it out of each origin layer. Needed
  for Pattern B / DISPERSED models.
- `exp6_v_ablation_fixed.py` (fixes branch): includes correct submodule whitelist
  for GLM-4 / Yi-9B / MoE, and reads L_origin from `L_ORIGIN.json`.

## Key metrics

- `Delta MA % = (ablated - baseline) / baseline * 100`. Expected <= -80 % for
  Pattern A; macro variant should achieve similar for Pattern B.
- `Delta PPL`: to check that the model is not catastrophically broken elsewhere.

## How to reproduce

```bash
# Pattern A (single layer)
python code/exp5_v_ablation.py \
    --model_name gptj_6b \
    --layer_id $L_ORIGIN \
    --nsamples 30 --seqlen 2048 \
    --output_dir results/gptj_6b/

# Pattern B (multi-layer macro)
python code/exp5_macro_v_ablation.py \
    --model_name gpt2 \
    --origin_layers 0,1,2,3,4 \
    --nsamples 30 --seqlen 2048 \
    --output_dir results/gpt2/
```

## Key findings

- 11 / 11 models with macro-v1 data satisfy Delta MA <= -70 %.
- Pattern A models (gptj_6b, bloom_7b1, falcon_7b, llama3.1_8b, yi_9b) all hit
  Delta MA <= -85 % on single-layer V ablation.
- The causal direction of `v1` is confirmed: replacing it zeros out MA while
  preserving PPL within 10 % of baseline.

## Result layout per model

```
results/<model>/
└── <model>_v_ablation_results.json
```

All 26 models have raw data.
