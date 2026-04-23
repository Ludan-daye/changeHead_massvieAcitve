# u1 decode - output-direction token vocabulary

## Purpose

Interpret the output subspace of the MA generator. Given `W_down` at L_origin,
its left singular vector `u1` points into the residual-stream direction where
MA lands. Projecting the unembedding matrix `W_U` onto `u1` gives a score per
vocabulary token; the top-k of that score is the "MA-aligned vocabulary".

## Method

1. Compute `u1 = U[:, 0]` from `SVD(W_down[L_origin])`.
2. Compute `scores[v] = W_U[v, :] . u1` for every vocab token v.
3. Report top-100 tokens sorted by score, separately for `+u1` and `-u1` sides.

## Key output

- Per model: top-100 tokens in the `+u1` direction and top-100 in `-u1`.
- Aggregated across 26 models: `aggregated/ALL_26_u1_combined.json`.

## How to reproduce

```bash
python code/systemd_decode_full.py \
    --model_name gptj_6b \
    --layer_id $L_ORIGIN \
    --topk 100 \
    --output results/gptj_6b/gptj_6b_u1.json
```

## Key findings

Across 26 models, top-ranked tokens in the `+u1` direction are:

- **English-dominant**: `\n`, `\n\n`, ` ` (space), `.`, `,`, BOS / EOS.
- **Multi-lingual (Qwen / GLM / Yi)**: CJK punctuation (full-width comma,
  period), full-width space, CJK function chars (的, 是, 了, ...).
- **Code-heavy models**: indentation tokens, `{`, `}`, `;`.

This provides a model-independent signature of "structural token vocabulary" and
reinforces the RQ3/RQ4 finding that MA lives at structural positions.

## Result layout per model

```
results/<model>/
└── <model>_u1.json
```

The `<model>_u1.json` files are extracted from
`aggregated/ALL_26_u1_combined.json`; the aggregated JSON is the canonical source.
