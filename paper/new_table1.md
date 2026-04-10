# Per-document independent sampling — new Table 1 numbers

Protocol: 256 independent C4 validation documents × 256 tokens, seed=42.

| Model | η (σ₁/σ₂) | Top1 baseline | Δhead-ablation % | ΔV-keep_top1 % | ΔV-remove_max % |
|---|---|---|---|---|---|
| GPT-2 | 1.19 | 3000.53 | 0.24 | -62.05 | -81.38 |
| OPT-6.7B | 1.92 | 366.66 | 210.45 | -6.16 | -29.43 |
| Qwen2.5-7B | 2.64 | 11323.06 | 89.25 | -1.65 | -99.11 |
| GPT-J-6B | 1.91 | 4204.25 | -98.21 | 3.78 | -68.41 |
| LLaMA-2-13B | — | — | — | — | — |
| Mistral-7B | 1.08 | 323.70 | -28.16 | -86.24 | -26.54 |
| BLOOM-7B1 | 1.62 | 3641.62 | -99.94 | -14.77 | -14.61 |
| Falcon-7B | 2.86 | 1880.03 | -24.36 | -69.40 | -44.54 |
