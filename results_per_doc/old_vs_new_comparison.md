# Old vs new ΔV comparison (sanity gate)

| Model | η old → new | ΔV old (paper) | ΔV new (remove_max) | abs delta |
|---|---|---|---|---|
| GPT-2 | 2.52 → 1.19 | -70.7% | -81.4% | 10.7 pp |
| OPT-6.7B | 2.87 → 1.92 | -77.3% | -29.4% | 47.9 pp |
| Qwen2.5-7B | 2.64 → 2.64 | -99.1% | -99.1% | 0.0 pp |
| GPT-J-6B | 1.91 → 1.91 | -69.6% | -68.4% | 1.2 pp |
| LLaMA-2-13B | — | — | — | — |
| Mistral-7B | 1.85 → 1.08 | -73.5% | -26.5% | 47.0 pp |
| BLOOM-7B1 | 1.18 → 1.62 | -69.6% | -14.6% | 55.0 pp |
| Falcon-7B | — → 2.86 | -75.9% | -44.5% | 31.4 pp |

## ⚠️  Sanity gate WARNING

- opt_7b: |Δ| = 47.9 percentage points (>15)
- mistral_7b_v03: |Δ| = 47.0 percentage points (>15)
- bloom_7b1: |Δ| = 55.0 percentage points (>15)
- falcon_7b_local: |Δ| = 31.4 percentage points (>15)

Do NOT update the paper before reviewing these.
