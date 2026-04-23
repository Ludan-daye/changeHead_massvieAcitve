# RQ6 - Top-K activation scan + macro-SVD

## Purpose

Directly manipulate activations at the origin layer: delete or retain the top-K
most-MA activations and measure the effect on MA magnitude and on PPL. The
macro-SVD variant aggregates across multiple layers to capture Pattern B models.

## Variants

- `exp6_single_layer_activation.py`: single-layer top-K delete / retain scan.
- `exp6_macro_svd_full.py`: SVD of accumulated `Delta h2` across origin layers.
- `exp6_progressive_ablation.py`: equivalent to RQ2c; greedy layer-by-layer
  ablation to find the minimal set whose ablation reduces MA most.

## Key metrics

- `top1_after_topK_delete`: residual MA after ablating top-K activations.
- `PPL_after`: language-modeling perplexity after intervention.
- `eta = max_rank_1_explained_variance / rank_2_explained_variance`: macro-SVD
  spectral concentration.
- `u1 alignment`: cosine between macro-u1 and each layer's u1.

## How to reproduce

```bash
# Single-layer
python code/exp6_single_layer_activation.py \
    --model_name gptj_6b \
    --layer_id $L_ORIGIN \
    --topk 10 --mode delete \
    --output_dir results/gptj_6b/

# Macro-SVD
python code/exp6_macro_svd_full.py \
    --model_name gpt2 \
    --origin_layers 0,1,2,3,4 \
    --output_dir results/gpt2/
```

## Key findings

- Deleting top-10 MA activations at L_origin restores near-baseline PPL on Pattern A
  models: the top-K intersection of position and direction carries the causal MA.
- macro-SVD across origin layers yields a single dominant direction for Pattern B:
  `gpt2`: eta = 3.48, u1 alignment = 0.856, R^2 = 0.87. This validates that
  Pattern B is "multi-layer collaboration along one shared macro direction".

## Result layout per model

```
results/<model>/
├── <model>_rq6_results.json
└── (for RQ6 macro variants, <model>_macro_svd_full.json)
```

All 26 models have raw data.
