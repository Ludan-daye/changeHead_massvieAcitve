# Experiment Results

Latest experimental results for "Function Words as Geometric Anchors".

## RQ1: Attention Mechanism Contribution Analysis (WikiText-2)

**Date**: 2026-04-15
**Hardware**: NVIDIA A100 80GB PCIe
**Dataset**: WikiText-2 (30 samples, 4096 tokens each)

| Model | Key Layer | Base MA | ΔTop1 | Pattern |
|-------|-----------|---------|-------|---------|
| Qwen2.5-7B | 16 | 11509.9 | +109.0% | Suppressive |
| LLaMA-2-7B-Chat | 26 | 2194.6 | -76.2% | Generative |
| BLOOM-7B1 | 12 | 3631.3 | -99.9% | Generative |
| GPT-J-6B | 16 | 4185.3 | -98.2% | Generative |
| Falcon-7B | 23 | 1871.8 | -25.8% | Hybrid |
| Mistral-7B | 19 | 318.4 | -18.7% | Hybrid |

**Pattern Classification**:
- **Suppressive** (ΔTop1 > 0): Attention suppresses MA → Qwen2.5
- **Generative** (ΔTop1 < -50%): Attention promotes MA → LLaMA-2, BLOOM, GPT-J
- **Hybrid** (|ΔTop1| < 50%): Weak/mixed effect → Falcon, Mistral

Each model folder contains:
- `table1_rq1.json` — Summary data for Table 1
- `baseline/results.json` — Per-layer activation statistics (baseline)
- `all_heads_disabled/results.json` — Per-layer statistics (all heads disabled)
- `comparison/` — Visualization PNGs + text summary report
