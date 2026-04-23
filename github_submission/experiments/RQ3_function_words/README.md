# RQ3 - Function-word / structural-token v1 projection

## Purpose

At the MA origin layer, test whether the magnitude of the v1-projection of `h2`
differs between function-word positions and content-word positions. This probes the
*location* of MA marks in the input sequence.

## Method

1. Identify the origin layer L_origin (from RQ2c greedy ablation, in
   `aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json -> <model> -> exp2c.l_origin_from_step1`).
2. Compute SVD of `W_down[L_origin]` to get `v1`.
3. For each token in WikiText-2, take `h2` (the hidden state input to `down_proj`)
   and compute `|h2 . v1|`.
4. Split tokens into function / content sets (via a whitelist of top-k English
   function words + punctuation heuristics). Compare distributions via Cohen's d.

## Key metrics

- Cohen's d (effect size) between the two distributions.
- Top-K tokens by `|h2 . v1|` magnitude (see `exp5_detailed_results.json`).
- Concentration ratio `|h2 . v1|_max / median`.

## How to reproduce

```bash
python code/exp5_function_words_svd_mapping.py \
    --model_name gptj_6b \
    --layer_id $L_ORIGIN \
    --nsamples 30 --seqlen 2048 \
    --output_dir results/gptj_6b/
```

Use the fixed version `code/exp5_function_words_svd_mapping_fixed.py` for MoE and
GLM-4 / Yi-9B (adds missing submodule whitelist; fixes a bug where only function
words were stored).

## Key findings (refined interpretation)

- In the raw Cohen's d tables, 18 of 24 models have d > 0.2 (function-word class
  has larger v1 projection). Several models (d < 0) originally looked like
  counter-evidence.
- **Re-interpretation after RQ4 Top-K inspection**: in `gpt2` at L3, the top-10
  tokens are dominated by newline / punctuation / CJK / special symbols. Only
  1 / 10 top token is a grammatical function word.
- The real separator is **"structural" (newline, punctuation, whitespace, special)
  vs semantic**, not purely function vs content. The function-word set is a
  subset of structural tokens.
- Top-K listings in `exp5_detailed_results.json` confirm the structural-token
  interpretation across all families.

## Result layout per model

```
results/<model>/
├── exp5_detailed_results.json
├── EXP5_SUMMARY.txt
├── exp5_alignment_v1.png
├── exp5_asymmetry_analysis.png
├── exp5_concentration_top5.png
├── exp5_stability_analysis.png
└── table1_rq3.json
```

All 26 models have raw data (mix of `wikitext_run/RQ3_origin/`, legacy
`RQ3_primary`, and re-runs).
