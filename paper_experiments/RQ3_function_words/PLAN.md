# RQ3 — 功能词 SVD 映射（执行计划）

## 做什么

在**起源层**（`exp2.critical_layer`）测：
- 功能词的 h₂ 在 W_down 的 v₁ 方向上投影（`function_alignment_mean`）
- 内容词的同方向投影（`content_alignment_mean`）
- 两组的效应量 Cohen's d

Cohen's d > 0 且大 → 功能词确实是 MA 的 mark 载体。

## 为什么必须 25 个全做

已做的 16 个都在 **peak_layer** 上做，**9 个 Cohen's d 为负**（内容词反而更对齐），是"mark 经 attention 广播到 peak 层已被稀释"的实证。要测真实的"功能词 mark"，必须回起源层。

## 使用的脚本

| 脚本 | 关键参数 | 输出 |
|---|---|---|
| `paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py` | `--model --layer_id L_ORIGIN --nsamples 30 --savedir` | `{model}_result.json` 含 `cohens_d, p_value, function_alignment_mean, content_alignment_mean` |

## 模型清单（25 个全做）

```bash
# 23 个已知起源层的，直接跑
bash run_rq345_origin_layer.sh "" rq3

# llama2_13b + glm4_32b 等依赖前置后再跑（填进 L_ORIGIN 表）
```

### 每模型层号表

| 模型 | L_origin | 模式 | 优先级 | 预期 Cohen's d |
|---|:-:|:-:|:-:|:-:|
| bloom_7b1 | 3 | A | ★★★ | > 1.0 |
| falcon_7b | 3 | A | ★★★ | > 1.0 |
| gptj_6b | 2 | A | **★ Pilot** | > 1.0（验收关键）|
| llama3.1_8b | 1 | A | ★★★ | > 0.5 |
| qwen2.5_0.5b | 0 | A | ★★★ | > 0.5 |
| qwen2.5_7b | 3 | A | ★★★ | > 0.5 |
| qwen2_7b | 3 | A | ★★★ | > 0.5 |
| qwen3_0.6b | 2 | A | ★★★ | > 0.5 |
| qwen3_1.7b | 2 | A | ★★★ | > 0.5 |
| yi_9b | 1 | A | ★★★ | > 0.5 |
| mistral_7b_v03 | 1 | 中间 | ★★ | 0.3 - 0.8 |
| qwen1.5_14b | 2 | 中间 | ★★ | 0.3 - 0.8 |
| qwen3_4b | 5 | 中间 | ★★ | 0.3 - 0.8 |
| qwen3_8b | 6 | 中间 | ★★ | 0.3 - 0.8 |
| qwen3_14b | 6 | 中间 | ★★ | 0.3 - 0.8 |
| qwen3.5_9b | 26 | 中间 | ★★ | 0.3 - 0.8 |
| glm4_9b | 17 | 中间→真 B | ★★ | < 0.3（弱，需 macro）|
| qwen3_30b_a3b (MoE) | 2 | 中间 | ★ | 弱（MoE 异常）|
| gpt2 | 3 | B | ★ | < 0.3（弱，需 macro）|
| opt_6.7b | 1 | B | ★ | < 0.3（弱）|
| qwen3_32b | 43 | B | ★ | < 0.3（弱）|
| qwen3.5_27b | 54 | B→真 B | ★ | < 0.3（弱）|
| qwen3.5_35b_a3b (MoE) | 39 | B | ★ | < 0.3（弱）|
| llama2_13b | 待 RQ2b | — | ⏸ | — |
| glm4_32b | 待 RQ1 fp32 | — | ⏸ | — |

## 验收标准

| 模式 | `cohens_d` 阈值 | `p_value` 阈值 |
|---|:-:|:-:|
| A (10 个) | **> 0.5**（GPT-J 预期 > 1.0）| < 1e-10 |
| 中间 (9 个) | 0.3 - 0.8 | < 1e-5 |
| B + MoE (6 个) | 0 - 0.3（弱是对的） | — |

**模式 B 的"弱结果"是正确结果**——证明单层 v₁ 捕不到多层协作的 mark，需要 macro v₁（RQ3 不做，归 RQ6 macro-SVD 里的 `projection.cohen_d` 字段）。

## 执行成本

- 每模型 ~5-10 min
- 23 个 ≈ **2.5h**（双卡 ~1.5h）
- 后续 llama2_13b + glm4_32b ≈ +30 min

## 输出

`results/wikitext_run/RQ3_origin/{model}/*.json`（**不覆盖**旧的 `RQ3/` 错层结果，保留对照）。
