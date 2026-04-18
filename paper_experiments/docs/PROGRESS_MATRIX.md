# 实验进度矩阵（function word + MA 框架）

> 数据来源：`ALL_EXPERIMENTS_SUMMARY.json`（25 模型 × exp1-exp7）
> 作用域：**排除 RQ7 训练动力学**，只覆盖 RQ1-RQ6
> 生成日期：2026-04-17

---

## 图例

| 符号 | 含义 |
|:-:|---|
| `✓` | 完成且层位正确、数据有效 |
| `◐` | 完成但**层位错误**（用了 peak_layer 而非 critical_layer） |
| `·` | 未跑 |
| `⚠` | 数据异常（NaN / Inf / 0 / 缺字段） |

---

## 一、26 模型 × 6 实验矩阵

| # | 模型 | RQ1<br>attn | RQ2<br>mlp | RQ3<br>func.words | RQ4<br>svd | RQ5<br>v-abl | RQ6<br>macro | 起源层 | 峰值层 | 生成模式 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | bloom_7b1 | ✓ | ✓ (-95%) | · | · | · | · | L3 | L12 | A 单层 |
| 2 | falcon_7b | ✓ | ✓ (-87%) | · | · | · | · | L3 | L23 | A 单层 |
| 3 | gpt2 | ✓ | ✓ (-6.7%) | · | · | · | · | L3 | L16 | B 多层 |
| 4 | gptj_6b | ✓ | ✓ (-90%) | · | · | · | · | L2 | L16 | A 单层 |
| 5 | llama2_13b | ✓ | ⚠ 缺 | · | · | · | · | 未知 | L22 | 未知 |
| 6 | llama3.1_8b | ✓ | ✓ (-87%) | ◐ | ◐ | ◐ L30 | ✓ | **L1** | L17 | A 单层 |
| 7 | mistral_7b_v03 | ✓ | ✓ (-41%) | · | · | · | · | L1 | L25 | 中间 |
| 8 | opt_6.7b | ✓ +250% 抑制 | ✓ (-12%) | · | · | · | · | L1 | L25 | B 多层 |
| 9 | qwen1.5_14b | ✓ | ✓ (-69%) | ◐ | ◐ | ◐ L38 | ✓ | **L2** | L37 | 中间 |
| 10 | qwen2.5_0.5b | ✓ | ✓ (-90%) | ◐ | ◐ | ◐ L22 | ✓ | **L0** | L15 | A 单层 |
| 11 | qwen2.5_7b | ✓ +266% 抑制 | ✓ (-85%) | · | ◐ | · | · | L3 | L16 | A 单层 |
| 12 | qwen2_7b | ✓ +∞ 抑制 | ✓ (-89%) | ◐ | ◐ | ◐ L26 | ✓ | **L3** | L16 | A 单层 |
| 13 | qwen3_0.6b | ✓ | ✓ (-94%) | ◐ | ◐ | ◐ L26 | ✓ | **L2** | L25 | A 单层 |
| 14 | qwen3_1.7b | ✓ | ✓ (-84%) | ◐ | ◐ | ◐ L26 | ✓ | **L2** | L25 | A 单层 |
| 15 | qwen3_4b | ✓ | ✓ (-34%) | ◐ | ◐ | ◐ L34 | ✓ | L5 | L17 | 中间 |
| 16 | qwen3_8b | ✓ | ✓ (-39%) | ◐ | ◐ | ◐ L34 | ✓ | L6 | L33 | 中间 |
| 17 | qwen3_14b | ✓ +85% 抑制 | ✓ (-67%) | ◐ | ◐ | ◐ L38 | ✓ | L6 | L33 | 中间 |
| 18 | qwen3_30b_a3b (MoE) | ✓ | ✓ (-58%) | ◐ | ◐ | · | ✓ | L2 | L36 | 中间 |
| 19 | qwen3_32b | ✓ +59% 抑制 | ✓ (-15%) | ◐ | ◐ | ◐ L62 | ✓ | L43 | L53 | B 多层 |
| 20 | qwen3.5_9b | ✓ | ✓ (-50%) | ◐ | ◐(n=1) | ◐ L30 | ✓ | L26 | L31 | 中间 |
| 21 | qwen3.5_27b | ✓ | ✓ (-30%) | ◐ | ◐ | ◐ L62 | ✓ | L54 | L58 | B 多层 |
| 22 | qwen3.5_35b_a3b (MoE) | ✓ | ✓ (-6.7%) | ◐ | ◐ | · | ✓ | L39 | L39 | B 多层 |
| 23 | yi_9b | ✓ +27% 抑制 | ✓ (-88%) | ◐ | ◐ | ◐ L46 | ✓ | **L1** | L47 | A 单层 |
| 24 | glm4_9b | ✓ | ✓ (-33%) | ◐ | ◐ | ◐ L38 | ✓ | L17 | L1 | 中间 |
| 25 | glm4_32b | ⚠ Inf | ⚠ 全 NaN | · | ⚠ NaN | ⚠ NaN | ⚠ NaN | 不可定 | L0 | 未知 |

**汇总**：

| RQ | 完成 | 层位正确 | 需修复 |
|---|:-:|:-:|:-:|
| RQ1 (attn 消融) | 23 / 25 | 23 | 0 |
| RQ2 (MLP 消融) | 24 / 25 | 24 | 0 |
| RQ3 (function words) | 16 / 25 | 0 | 16（全错层） |
| RQ4 (SVD 对齐) | 18 / 25 | 0 | 18（全错层） |
| RQ5 (V 消融) | 14 / 25 | 0 | 14（全错层 / 模式 B 需换 RQ6） |
| RQ6 (macro-SVD) | 17 / 25 | 17 | 0 |

---

## 二、核心 Bug：RQ3/4/5 层位错误

**根因**：`paper_experiments/run_all_rq.sh:21-25` 和 `run_rq345_peak_layer.sh:13-20` 把 `KEY_LAYER` 读成 `table1_rq1.json["key_layer"]`（peak_layer），传给了 RQ3/4/5 的 `--layer_id`。正确做法是读 RQ2 结果 `exp2.critical_layer`（起源层）。

**两种错位类型**：

| 类型 | 触发条件 | 错在哪 | 修复 |
|---|---|---|---|
| **A. 层号错位** | 单层 ΔMA ≥ 60%（模式 A / 部分中间） | 起源层 ≠ 峰值层，脚本用了峰值层 | 改 `--layer_id = exp2.critical_layer` 重跑 RQ3/4/5 |
| **B. 模式错位** | 单层 ΔMA < 60%（模式 B / MoE） | 不该用单层 RQ5，应用 macro-SVD | 用 `RQ6_single_layer_activation/exp6_macro_svd_*` |

**直接证据**（`MA_CONCLUSIONS_AND_ARGUMENTS.md:183`）：
> GPT-J L2 消融 MA → **-99.1%**，L16 消融 → **-0.01%**。

---

## 三、数据异常清单

| 模型 | 问题 | 处理 |
|---|---|---|
| glm4_32b | exp1 Infinity、exp2 L10+ 全 NaN、exp4/5/6 全 NaN | 疑似 dtype / overflow；需查 loader + fp16→fp32 cast |
| llama2_13b | 缺 exp2 | 必须补 RQ2 才能确定起源层 |
| qwen2_7b exp1 | ΔTop1 = +∞（baseline 接近 0） | RQ1 结果不可信，重测 baseline |
| qwen3.5_9b exp4 | num_layers = 1（其他都 5） | 数据截断，重跑 |
| qwen3_30b_a3b / qwen3.5_35b_a3b (MoE) | 缺 exp5 | 需 expert-level v-ablation 框架（单独设计） |

---

## 四、重跑方案（按优先级）

### Tier A — 起源层重跑 RQ3 / RQ4 / RQ5（核心 bug 修复）

修改脚本：把 `KEY_LAYER` 改成读取 `exp2.critical_layer`（新建 `run_rq345_origin_layer.sh`）。结果存 `results/wikitext_run/RQ3-5_origin/` 避免覆盖原对照数据。

| 子批 | 对象 | 起源层范围 | 预期 RQ5 ΔMA | 成本 |
|---|---|---|---|---|
| **A1** 模式 A（10 个）| bloom_7b1, falcon_7b, gptj_6b, llama3.1_8b, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, qwen3_1.7b, yi_9b | L0-L3 | **-85% ~ -99%** | ≈ 3h |
| **A2** 中间形态（7 个） | mistral_7b_v03, qwen1.5_14b, glm4_9b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3.5_9b | L1-L26 | **-30% ~ -70%** | ≈ 3h |
| **A3** 模式 B（5 个）| gpt2, opt_6.7b, qwen3_32b, qwen3.5_27b, qwen3.5_35b_a3b | 单层做意义弱 | < -30%，佐证"非单层主导" | ≈ 2h |

### Tier B — 数据补洞

| 任务 | 成本 |
|---|---|
| llama2_13b 补 RQ2 | ~15 min |
| qwen2_7b exp1 baseline 重测 | ~10 min |
| qwen3.5_9b RQ4 重跑（完整 5 层） | ~20 min |
| glm4_32b 全流程 fp32 重测 | ~1h |

### Tier C — 暂缓（需新框架）

- MoE 模型 qwen3_30b_a3b, qwen3.5_35b_a3b 的 expert-level v-ablation
- RQ7 训练动力学（本轮排除）

---

## 五、建议执行顺序

1. 先修脚本（`run_rq345_origin_layer.sh`），对 **GPT-J** 单模型做 pilot 验证：起源层 RQ5 应出 -90%+
2. pilot 通过后批量跑 Tier A1（10 个模式 A 模型）
3. Tier B 数据补洞（可与 A 并行）
4. Tier A2 → Tier A3
5. 结果合并到新 ALL_EXPERIMENTS_SUMMARY_v2.json，更新本表格
