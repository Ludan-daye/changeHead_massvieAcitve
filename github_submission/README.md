# Massive Activations: Mechanism and Empirical Evidence

Supplementary code and data for the research project on **Massive Activations (MA)** in
Large Language Models. This repository organizes the empirical results of **26 LLMs ×
8 experiments** that together characterize the origin, propagation, and stabilization
of MAs.

The main claim, supported across the 26 models in this submission:

> **Massive Activations are MLP-written "marks" on structural tokens** (newlines,
> punctuation, function words, special symbols). These marks are then broadcast by
> attention via the softmax/V mechanism and stabilized into a steady state by layer
> norms and down-stream modules.

## Five-step mechanism

1. **Mark formation.** MLP layers write extreme activations in a low-rank direction
   `v_1` (top right singular vector of `W_down`) at structural-token positions.
2. **Attention sink.** In softmax, a token carrying MA in `v_1` hijacks attention
   weights because `Q·K` becomes dominated by the `v_1` component.
3. **Broadcast.** `V_{MA_token}` is then attention-broadcast to all positions, aligning
   the whole sequence along `v_1`.
4. **Bifurcation.** Downstream attention heads project either along `+v_1`
   (*generative*, 17 / 25 models) or `-v_1` (*suppressive*, 8 / 25 models).
5. **Steady state.** Residual streams, layer norms, and GELU act as soft gates that
   cap the absolute magnitude.

Two orthogonal axes emerge:

- **Generation mode**: Pattern A (single dominant layer) vs Pattern B (multi-layer
  distributed) — probed by RQ2b/RQ2c/RQ6.
- **Regulation mode**: Generative vs Suppressive — probed by RQ1.

## Repository layout

```
github_submission/
├── README.md                          <- this file
├── MODELS.md                          <- 26 models with size/type/source
├── EXPERIMENTS.md                     <- 8 experiments: purpose + code + runtime
├── CONCLUSIONS.md                     <- findings synthesis across 26 models
├── .gitignore
├── experiments/
│   ├── RQ1_attention/      code/ + results/<model>/
│   ├── RQ2a_mlp/           code/ + results/<model>/
│   ├── RQ3_function_words/ code/ + results/<model>/
│   ├── RQ4_svd_alignment/  code/ + results/<model>/
│   ├── RQ5_v_ablation/     code/ + results/<model>/
│   ├── RQ6_topk_scan/      code/ + results/<model>/
│   ├── HC_entropy/         code/ + results/<model>/
│   └── u1_decode/          code/ + results/<model>/
├── lib/                               <- shared Python utilities
├── monkey_patch/                      <- attention hooks for HF Transformers
├── aggregated/
│   ├── ALL_EXPERIMENTS_SUMMARY_v2.json   <- all 26 models × 6 RQs metrics
│   └── ALL_26_u1_combined.json           <- u1 direction top-k tokens
└── docs/                              <- extended notes, mechanism references
```

## Reproducibility

1. Install deps (Python 3.10+, `torch>=2.0`, `transformers==4.36.0`,
   `scikit-learn`, `scipy`):
   ```bash
   pip install -r ../paper_experiments/requirements.txt
   ```
2. Point `lib/model_dict.py` at your local model paths or use HF fallback.
3. Run any of the experiment scripts, e.g.:
   ```bash
   python experiments/RQ1_attention/code/exp1_feasibility_test.py \
       --model_name gptj_6b --nsamples 30 --seqlen 2048
   ```
4. Aggregate results into JSON (matching `aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json`
   schema) for downstream plotting.

Each per-experiment `results/<model>/` directory contains the raw outputs from that
experiment on that model: `.json` metrics, `.txt` summaries, and diagnostic plots
(`.png`). When a model's raw per-model directory was unavailable for RQ1/RQ2a, a
`MISSING.txt` is provided pointing to the corresponding entry in
`aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json` (where all 26 models × 6 RQs metrics
are consolidated).

## Models covered

26 LLMs spanning GPT-2 / GPT-J / BLOOM / Falcon / LLaMA / Mistral / OPT / Qwen 1.5 /
Qwen 2 / Qwen 2.5 / Qwen 3 / Qwen 3.5 / GLM-4 / Yi families. See `MODELS.md`.

## Experiments at a glance

| ID  | Purpose                                     | Code                                 |
|:---:|---------------------------------------------|--------------------------------------|
| RQ1 | Attention disabled -> MA residual?          | `exp1_feasibility_test.py`           |
| RQ2a| MLP disabled -> MA collapses?               | `exp2a_mlp_feasibility_test.py`      |
| RQ3 | Function-word vs content-word v1 projection | `exp5_function_words_svd_mapping.py` |
| RQ4 | SVD alignment (sigma1/sigma2, top-tokens)   | `exp3_svd_alignment_analysis.py`     |
| RQ5 | Replace V with random orthogonal -> MA?     | `exp5_v_ablation.py` (+ macro)       |
| RQ6 | Top-K activation deletion / retention scan  | `exp6_single_layer_activation.py`    |
| HC  | Entropy histograms of MA-vs-normal positions| `exp5c_entropy.py`                   |
| u1  | Decode top-k tokens aligned with u1         | `systemd_decode_full.py`             |

See `EXPERIMENTS.md` for detailed descriptions and runtime estimates.

## Citation

Ma L. et al. *Function Words as Geometric Anchors: A Mechanistic Study of Massive
Activations in Large Language Models* (ACL submission, 2026). See
`docs/MA_FRAMEWORK.md` and `docs/EXPERIMENT_PLAN.md` for the full theoretical and
experimental framework.
