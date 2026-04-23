# Experiments overview (8)

Six research questions (RQ1-RQ6) plus two supporting analyses (HC entropy, u1 decode).

| Exp   | Purpose                                                     | Primary output                              | Runtime (1 model, single GPU) |
|:-----:|-------------------------------------------------------------|---------------------------------------------|:-----------------------------:|
| RQ1   | Disable all attention; does MA persist?                     | `EXPERIMENT_1_SUMMARY.txt`, `results.json`, 4 png | 3-15 min               |
| RQ2a  | Disable all MLP; does MA collapse?                          | `EXPERIMENT_2A_SUMMARY.txt`, `results.json`, 4 png | 3-15 min               |
| RQ3   | Function-word vs content-word v1 projection                 | `EXP5_SUMMARY.txt`, `exp5_detailed_results.json`, 4 png | 5-20 min     |
| RQ4   | sigma1/sigma2 + top-aligned tokens (SVD analysis)           | `EXPERIMENT_3_SUMMARY.txt`, `exp3_detailed_results.json`, 5 png | 5-15 min |
| RQ5   | Replace V matrix with random orthogonal; measure MA change  | `<model>_v_ablation_results.json`           | 10-30 min               |
| RQ6   | Top-K activation deletion / retention scan (macro-SVD)      | `<model>_rq6_results.json`, `<model>_macro_svd_full.json` | 15-40 min    |
| HC    | Entropy histograms: MA-vs-normal token position entropy     | `exp5c_entropy_results.json`, `exp5c_raw_positions.npz` | 5-15 min        |
| u1    | Decode top-K tokens aligned with u1 (W_down top left svec)  | `<model>_u1.json`                           | 2-10 min                |

## RQ1 -- Attention contribution

**Purpose**: falsify H0 ("attention is the origin of MA"). Attention is hooked to
output zeros everywhere; the forward pass is re-run on WikiText to measure residual
MA (top-1 activation magnitude across layers).

- Key metric: `residual_% = disabled_top1 / baseline_top1 * 100`
- Sub-metric: `Delta MA %` → sign decides *generative* (< 0) vs *suppressive* (> 0)
- Result across 25 models with data: **no model reaches zero** (min residual = 1.69%
  on gptj_6b). H0 is falsified; attention is a propagator, not a creator.

Code: `experiments/RQ1_attention/code/exp1_feasibility_test.py`

## RQ2a -- MLP contribution

**Purpose**: verify H1 ("MLP is the MA creator"). Same protocol as RQ1 but disables
MLP outputs layer-wide.

- Key metric: `retain_% = disabled_top1 / baseline_top1 * 100` (lower = MLP is origin)
- 20 / 24 dense models have `retain <= 10%` (strong origin). `bloom_7b1` drops to
  zero. 4 anomalies: qwen3.5_35b_a3b (MoE script quirk), gpt2 (old arch),
  qwen3.5_9b, qwen3.5_27b.

Code: `experiments/RQ2a_mlp/code/exp2a_mlp_feasibility_test.py`

## RQ3 -- Function-word / structural-token v1 projection

**Purpose**: show that MA is concentrated at specific token *types*, not uniformly
across the sequence. At the origin layer, project each token's `h2` onto `v1` (top
right singular vector of the origin layer's `W_down`), then compute the Cohen's d
between the distribution at function-word positions and the distribution at
content-word positions.

- Modern re-analysis (see CONCLUSIONS.md): the sharpest separation is actually
  between **structural tokens** (newlines, punctuation, whitespace-tokens) and
  everything else, not purely function vs content. Top-K of `|h2 . v1|` is
  dominated by newline and punctuation tokens.

Code: `experiments/RQ3_function_words/code/exp5_function_words_svd_mapping.py`

## RQ4 -- SVD alignment geometry

**Purpose**: at the origin layer, compute `SVD(W_down) = U Sigma V^T`. The MA
direction must be captured by `v1` and by `u1` (for the output subspace).

- Key metrics: `sigma1 / sigma2` (spectral concentration; >= 3 = Pattern A),
  `max_j |u1_j|` (output coordinate concentration), top-tokens aligned with `v1`.

Code: `experiments/RQ4_svd_alignment/code/exp3_svd_alignment_analysis.py`

## RQ5 -- V matrix causal test

**Purpose**: causally confirm that `v1` is load-bearing. At the origin layer, replace
`W_down`'s right subspace around `v1` with a random orthogonal perturbation, run
inference, and measure the change in MA.

- `exp5_v_ablation.py`: single-layer V replacement.
- `exp5_macro_v_ablation.py`: multi-layer macro-v1 projection-out (needed for
  Pattern B models).

Expected: Delta MA <= -80% for Pattern A models; macro version handles Pattern B.

Code: `experiments/RQ5_v_ablation/code/{exp5_v_ablation.py, exp5_macro_v_ablation.py}`

## RQ6 -- Top-K activation scan + macro-SVD

**Purpose**: directly manipulate activations. Delete / retain the top-K most-MA
activations, re-run, and measure MA/PPL. The macro-SVD variant aggregates
`Delta h2` across multiple layers.

Code: `experiments/RQ6_topk_scan/code/{exp6_single_layer_activation.py, exp6_macro_svd_full.py, exp6_progressive_ablation.py}`

## HC entropy

**Purpose**: characterize the entropy (positional uncertainty) of the next-token
distribution at MA positions vs normal positions. Provides evidence that MA
positions carry low-entropy anchor information.

Output: `exp5c_entropy_results.json` + `exp5c_raw_positions.npz` with raw positional
samples.

Code: `experiments/HC_entropy/code/exp5c_entropy.py`

## u1 decode

**Purpose**: interpret the output direction `u1` by projecting the unembedding matrix
onto it and listing the top-k tokens most aligned with `+u1`.

- Key output: per model, a list of top-100 tokens whose unembedding row aligns with
  `u1` of the origin layer's `W_down`. Often dominated by "\n", " ", punctuation,
  and in some models, by language-specific stopwords.

Code: `experiments/u1_decode/code/systemd_decode_full.py`

## Runtime reference

For a 7-8B dense model on a single A100, the full RQ1-RQ6 + HC + u1 pipeline takes
approximately 1-2 hours. Larger models (14B+) take 2-3x longer. The aggregate
`ALL_EXPERIMENTS_SUMMARY_v2.json` can be reproduced end-to-end in ~18 h on a
single A100 across all 26 models, or ~10 h on two A100s.
