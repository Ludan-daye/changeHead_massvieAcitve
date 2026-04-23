# Models covered (26)

All 26 models are autoregressive decoder-only Transformers. 24 are dense MLP;
**2 are Mixture-of-Experts (Qwen3-30B-A3B, Qwen3.5-35B-A3B)** and are analyzed
separately — the main MA theory (single-V direction) does not directly apply to the
per-expert structure.

| # | Model id (canonical) | Family | Params | Type | HuggingFace id |
|:-:|----------------------|--------|-------:|:----:|----------------|
| 1 | `gpt2`                    | GPT-2     | 124M  | dense | `gpt2` |
| 2 | `opt_6.7b`                | OPT       | 6.7B  | dense | `facebook/opt-6.7b` |
| 3 | `bloom_7b1`               | BLOOM     | 7.1B  | dense | `bigscience/bloom-7b1` |
| 4 | `falcon_7b`               | Falcon    | 7B    | dense | `tiiuae/falcon-7b` |
| 5 | `gptj_6b`                 | GPT-J     | 6B    | dense | `EleutherAI/gpt-j-6b` |
| 6 | `mistral_7b_v03`          | Mistral   | 7B    | dense | `mistralai/Mistral-7B-v0.3` |
| 7 | `llama2_7b_chat`          | LLaMA-2   | 7B    | dense | `meta-llama/Llama-2-7b-chat-hf` |
| 8 | `llama2_13b`              | LLaMA-2   | 13B   | dense | `meta-llama/Llama-2-13b-hf` |
| 9 | `llama3.1_8b`             | LLaMA-3.1 | 8B    | dense | `meta-llama/Meta-Llama-3.1-8B` |
| 10| `qwen1.5_14b`             | Qwen 1.5  | 14B   | dense | `Qwen/Qwen1.5-14B` |
| 11| `qwen2_7b`                | Qwen 2    | 7B    | dense | `Qwen/Qwen2-7B` |
| 12| `qwen2.5_0.5b`            | Qwen 2.5  | 0.5B  | dense | `Qwen/Qwen2.5-0.5B` |
| 13| `qwen2.5_7b`              | Qwen 2.5  | 7B    | dense | `Qwen/Qwen2.5-7B` |
| 14| `qwen3_0.6b`              | Qwen 3    | 0.6B  | dense | `Qwen/Qwen3-0.6B` (private) |
| 15| `qwen3_1.7b`              | Qwen 3    | 1.7B  | dense | `Qwen/Qwen3-1.7B` (private) |
| 16| `qwen3_4b`                | Qwen 3    | 4B    | dense | `Qwen/Qwen3-4B` (private) |
| 17| `qwen3_8b`                | Qwen 3    | 8B    | dense | `Qwen/Qwen3-8B` (private) |
| 18| `qwen3_14b`               | Qwen 3    | 14B   | dense | `Qwen/Qwen3-14B` (private) |
| 19| `qwen3_32b`               | Qwen 3    | 32B   | dense | `Qwen/Qwen3-32B` (private) |
| 20| `qwen3_30b_a3b`           | Qwen 3    | 30B (3B active) | **MoE** | `Qwen/Qwen3-30B-A3B` (private) |
| 21| `qwen3.5_9b`              | Qwen 3.5  | 9B    | dense | internal (no public HF) |
| 22| `qwen3.5_27b`             | Qwen 3.5  | 27B   | dense | internal (no public HF) |
| 23| `qwen3.5_35b_a3b`         | Qwen 3.5  | 35B (3B active) | **MoE** | internal (no public HF) |
| 24| `glm4_9b`                 | GLM-4     | 9B    | dense | `THUDM/glm-4-9b` |
| 25| `glm4_32b`                | GLM-4     | 32B   | dense | `THUDM/glm-4-32b` |
| 26| `yi_9b`                   | Yi        | 9B    | dense | `01-ai/Yi-9B` |

## Naming notes

Some legacy per-model directories use alternate names that are auto-mapped:

| canonical | also seen as |
|-----------|--------------|
| `bloom_7b1`      | `bloom_7b` |
| `mistral_7b_v03` | `mistral_7b` |

In `aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json`, the **canonical** name is always
the key. Per-experiment result dirs in `experiments/<RQ>/results/<model>/` also use
the canonical name.

## MoE caveat

Qwen3-30B-A3B and Qwen3.5-35B-A3B use `SparseMoeBlock` with per-expert W_up / W_down.
The per-expert MA generation path is separate from the dense-MLP path; our
single-V-direction analysis does not cover these models in the main results. See
`docs/EXPERIMENT_PLAN.md` §MoE for the preliminary per-expert investigation.
