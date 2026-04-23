# HC entropy - positional entropy of MA positions

## Purpose

Show that positions carrying MA have systematically lower next-token entropy than
average positions, supporting the "low-entropy anchor" interpretation: structural
tokens are predictable and therefore good candidates for the model to "write on".

## Method

1. Forward-pass WikiText-2 through the model.
2. Identify MA positions (top-K `|h2 . v1|` at the origin layer).
3. At each position, compute the entropy of the softmax output distribution of the
   next-token logits.
4. Compare the distributions: MA-position entropy vs uniform-random-position
   entropy.

## Key metrics

- `entropy_ma_mean`, `entropy_ma_median`, `entropy_normal_mean`, ...
- `cohens_d` between MA and normal distributions.
- Raw per-position entropies in `exp5c_raw_positions.npz` (can be re-plotted /
  histogrammed).

## How to reproduce

```bash
python code/exp5c_entropy.py \
    --model_name gptj_6b \
    --layer_id $L_ORIGIN \
    --nsamples 30 --seqlen 2048 \
    --output_dir results/gptj_6b/
```

## Key findings

- MA positions have notably lower entropy than average (Cohen's d typically > 0.5,
  sometimes > 1 on structural-heavy models).
- This reinforces the RQ3 finding: MA sits at predictable positions (newlines,
  punctuation, BOS/EOS) where the model has already narrowed down the
  distribution to a few candidates.

## Result layout per model

```
results/<model>/
├── exp5c_entropy_results.json
└── exp5c_raw_positions.npz     <- large (can be removed if space is tight)
```

All 26 models have raw data.

Note: The `.npz` files are the largest single contributor to repo size
(~1 MB each, 26 MB total). Keep if you want to replot; safe to delete if you
trust the pre-aggregated `exp5c_entropy_results.json` values.
