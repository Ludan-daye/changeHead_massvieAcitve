# RQ4 - SVD alignment geometry

## Purpose

Decompose the origin-layer `W_down = U Sigma V^T` and examine whether MA tokens
project cleanly onto the top singular direction.

## Method

1. At L_origin, compute `U, Sigma, V^T = SVD(W_down)`.
2. Report `sigma1 / sigma2` (spectral concentration).
3. Compute `max_j |u1_j|` (output-subspace concentration).
4. List the top tokens whose `|h2 . v1|` is largest.
5. Regress `top1_activation` on `|h2 . v1|` (R^2 should be close to 1 for Pattern A).

## Key metrics

- `sigma1 / sigma2`: > 3 => Pattern A (concentrated spectrum).
- `max_j |u1_j|`: > 0.6 => output stacks into a few hidden dims.
- Top-K tokens: the "structural token vocabulary" (see RQ3 interpretation).
- `trigger_rate`: fraction of tokens with MA >= threshold.

## How to reproduce

```bash
python code/exp3_svd_alignment_analysis.py \
    --model_name gptj_6b \
    --layer_id $L_ORIGIN \
    --nsamples 30 --seqlen 2048 \
    --output_dir results/gptj_6b/
```

## Key findings

- Pattern A models (gptj_6b, bloom_7b1, falcon_7b, llama3.1_8b, yi_9b):
  `sigma1 / sigma2 >= 3`, `max |u1| >= 0.6`, R^2 > 0.85. The v1 subspace fully
  explains MA.
- Pattern B models (gpt2, opt_6.7b, qwen3_32b, qwen3.5_27b): `sigma1 / sigma2 < 3`
  at any single layer; their macro version (multi-layer `Delta h2`) is needed.
- Top-K tokens across all models are dominated by structural tokens
  (newline, punctuation, language-specific stopwords).

## Result layout per model

```
results/<model>/
├── exp3_detailed_results.json
├── EXPERIMENT_3_SUMMARY.txt
├── exp3_singular_values.png
├── exp3_alignment_comparison.png
├── exp3_projection_regression.png
├── exp3_top_tokens.png
├── exp3_trigger_rate.png
└── table1_rq4.json
```

All 26 models have raw data.
