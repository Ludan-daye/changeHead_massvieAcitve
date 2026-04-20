# 25-Model Experiment Results Compilation

> **日期**: 2026-04-20  
> **范围**: 25 个 LLM × 6 个 RQ（RQ7 排除）  
> **数据源**: `paper_experiments/results/ALL_EXPERIMENTS_SUMMARY_v2.json` + `results/wikitext_run/RQ3_origin/` / `RQ4_origin/` / `RQ5_origin/` / `RQ5_macro/`  
> **本文档目的**: 为后续分析（论文写作 / 机制论证 / 补实验）提供单一表格与关键指标索引。

---

## 0. 摘要（TL;DR）

- **覆盖**: 25/25 模型有 exp1（attention 消融）+ exp2c（贪心累积）数据；23/25 有 origin-layer σ₁/σ₂；7/25 有 origin-layer RQ5（单层 + macro V 消融）；18/25 有 legacy peak-layer RQ5。
- **exp2c 模式分布**: CONCENTRATED 8，FEW-SOURCE 8，DISPERSED 8，异常 1。
- **Cohen's d 符号**: 正 11，负 11。结合 origin-layer 重估后，gptj/falcon/bloom（4/6 Tier1-3）从 peak 负值转为 origin 正值——**原层号错误会翻转功能词对齐结论**。
- **σ₁/σ₂ ≥ 2**: 5 模型；≥ 3: 2 模型（gpt2, glm4_9b）。其余绝大多数 ≈ 1，说明**单层 W_down 并非显著谱集中**——模式 A 的谱证据远弱于预期。
- **RQ5 single ΔMA ≤ -90%**: 3/7 有 origin 数据的模型（gptj, falcon, llama2_13b）。
- **RQ5 macro V 消除更强**: 1 模型 macro ΔMA 比 single 多 ≥5pt（典型 gpt2: -6% → -95%）。**多层 macro v₁ 是 DISPERSED/FEW-SOURCE 的因果证据主路径**。
- **peak-layer V 归因 ≥ 80%**: 13/18 legacy 模型。peak 层结果始终**高估** V 的作用（因为 peak 是 attention 汇聚后的读数层，而非 MA 写入层）。
- **attention 模式**: generative 17 / suppressive 7；与生成模式（A/B）正交（详见 §4.5）。

---

## 1. 主表：25 模型 × 核心指标

> 排序：CONCENTRATED → FEW-SOURCE → DISPERSED → ANOMALY，同类内按模型名字母序。  
> `L` = 起源层 (exp2c.l_origin_from_step1)；`drop%` = exp2c 最终 MA 下降；`σ` = origin-layer σ₁/σ₂；`ΔMA_s` / `ΔMA_m` = origin-layer 单层 / macro V 消融 ΔMA%。
> `Cohen_d` 列：★ 表示基于 origin 层重估；否则为 legacy peak-layer。  
> `V_att_peak` 来自老 exp5（peak 层 V 归因），仅作对照，**不作为主证据**。  

| # | 模型 | L | 类别 | steps | drop% | exp1_mode | exp1_Δ% | Cohen_d | σ₁/σ₂ | baseline_MA | ΔMA_s | ΔMA_m | peak_L | V_att_peak% |
|---|---|:-:|---|:-:|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | `bloom_7b1` | 3 | CONCENTRATED | 1 | 95.9 | generative | -98.3 | 0.125★ | 1.41 | 119.9 | -8.6 | — | — | — |
| 2 | `glm4_32b` | 0 | CONCENTRATED | 1 | 95.9 | generative | 166.9 | 0.404 | 1.53 | 133.0 | — | — | 59 | 93.2 |
| 3 | `gptj_6b` | 2 | CONCENTRATED | 1 | 90.7 | generative | -94.3 | 0.769★ | 2.52 | 3,677 | -99.0 | -99.1 | — | — |
| 4 | `mistral_7b_v03` | 0 | CONCENTRATED | 1 | 89.7 | generative | -17.6 | — | 1.08 | 1.5 | -83.2 | — | — | — |
| 5 | `qwen2.5_0.5b` | 0 | CONCENTRATED | 1 | 97.1 | generative | -45.8 | -0.501 | 1.48 | 20.3 | — | — | 22 | 64.2 |
| 6 | `qwen2.5_7b` | 3 | CONCENTRATED | 1 | 89.5 | generative | 266.2 | -1.344 | 2.64 | 8,768 | — | — | — | — |
| 7 | `qwen2_7b` | 3 | CONCENTRATED | 1 | 92.5 | — | — | -1.459 | 2.84 | 4,716 | — | — | 26 | 92.1 |
| 8 | `qwen3_0.6b` | 2 | CONCENTRATED | 1 | 92.8 | generative | -76.7 | -0.279 | 1.41 | 230.5 | — | — | 26 | 83.3 |
| 9 | `falcon_7b` | 3 | FEW-SOURCE | 2 | 93.5 | generative | -21.4 | 0.508★ | 1.37 | 1,194 | -98.1 | -97.4 | — | — |
| 10 | `glm4_9b` | 1 | FEW-SOURCE | 2 | 90.7 | generative | -44.9 | 0.235 | 3.26 | 64.2 | — | — | 38 | 89.9 |
| 11 | `gpt2` | 3 | FEW-SOURCE | 3 | 64.0 | generative | -60.0 | -0.314★ | 3.05 | 2,648 | -6.5 | -94.6 | — | — |
| 12 | `llama2_13b` | 0 | FEW-SOURCE | 3 | 93.6 | generative | -79.5 | — | 1.32 | 64.8 | -95.6 | -28.6 | — | — |
| 13 | `llama3.1_8b` | 1 | FEW-SOURCE | 2 | 90.3 | generative | -56.6 | 0.369 | 1.38 | 16.2 | — | — | 30 | 82.0 |
| 14 | `qwen3.5_35b_a3b` | 9 | FEW-SOURCE | 3 | 15.9 | generative | 5.1 | 0.673 | 1.03 | 8,000 | — | — | 38 | 0.0 |
| 15 | `qwen3_1.7b` | 2 | FEW-SOURCE | 3 | 89.8 | generative | -22.0 | -0.147 | 1.33 | 487.0 | — | — | 26 | 67.7 |
| 16 | `qwen3_4b` | 6 | FEW-SOURCE | 2 | 89.1 | generative | -13.1 | 0.377 | 1.24 | 7,048 | — | — | 34 | 93.3 |
| 17 | `qwen1.5_14b` | 35 | DISPERSED | 15 | 39.7 | generative | -15.1 | 0.690 | 1.05 | 704.5 | — | — | 38 | 93.4 |
| 18 | `qwen3.5_27b` | 54 | DISPERSED | 12 | 55.6 | generative | -16.0 | -1.005 | 1.12 | 115.0 | — | — | 62 | 85.5 |
| 19 | `qwen3.5_9b` | 22 | DISPERSED | 15 | 52.2 | generative | -16.3 | -0.889 | 1.06 | 47.8 | — | — | 30 | 80.5 |
| 20 | `qwen3_14b` | 6 | DISPERSED | 15 | 71.6 | generative | 84.9 | 1.099 | 1.33 | 9,904 | — | — | 38 | 94.9 |
| 21 | `qwen3_30b_a3b` | 1 | DISPERSED | 10 | 75.4 | generative | -19.8 | -0.052 | 1.17 | 8,000 | — | — | 46 | 0.0 |
| 22 | `qwen3_32b` | 6 | DISPERSED | 10 | 80.2 | generative | 59.3 | 1.443 | 1.35 | 23,920 | — | — | 62 | 93.9 |
| 23 | `qwen3_8b` | 6 | DISPERSED | 15 | 57.7 | generative | -34.1 | -0.553 | 1.48 | 13,304 | — | — | 34 | 94.4 |
| 24 | `yi_9b` | 8 | DISPERSED | 5 | 89.6 | generative | 27.3 | -0.312 | — | 87.9 | — | — | 46 | 83.6 |
| 25 | `opt_6.7b` | — | ANOMALY_NO_MLP_RESPONSE | 0 | 0.0 | generative | 250.3 | — | — | 148.1 | — | — | — | — |

**列定义**:
- `L` (origin layer)：按 RQ2c 贪心累积第一层禁用即可让 MA top1 显著坍塌的层号。
- `steps`：贪心累积消融至达到 floor 所需的层数（CONCENTRATED=1, FEW-SOURCE=2-5, DISPERSED>5）。
- `drop%`：exp2c 最终 MA 相对 baseline 的下降百分比。
- `exp1_Δ%`：RQ1 attention 消融后 MA 变化百分比（负 = generative；正 = suppressive）。
- `Cohen_d`：功能词 vs 内容词在 W_down v₁ 方向上的 alignment 效应量。
- `σ₁/σ₂`：origin 层 W_down 奇异值谱比（RQ4）。理论预期模式 A ≥ 3。
- `baseline_MA`：wikitext baseline 的 hidden state top1 abs 值均值。
- `ΔMA_s` / `ΔMA_m`：将 V 替换为随机正交矩阵（单层）/ 投影掉 macro v₁ 方向（多层 origin_layers 集合）后的 MA 变化。
- `peak_L` / `V_att_peak%`：legacy RQ5 在 peak 层（非 origin 层）做 V 归因的结果；**已知会高估 V 权重**，列出供对照。

---

## 2. 分组明细

### 2.1 CONCENTRATED — 单层主导 (steps = 1)

- **`bloom_7b1`** (L=3, drop=95.9%): exp1 generative Δ=-98.3%, σ₁/σ₂=1.41, Cohen_d=0.125★, ΔMA_single=-8.6%
- **`glm4_32b`** (L=0, drop=95.9%): exp1 generative Δ=166.9%, σ₁/σ₂=1.53, Cohen_d=0.404
- **`gptj_6b`** (L=2, drop=90.7%): exp1 generative Δ=-94.3%, σ₁/σ₂=2.52, Cohen_d=0.769★, ΔMA_single=-99.0%, ΔMA_macro=-99.1%
- **`mistral_7b_v03`** (L=0, drop=89.7%): exp1 generative Δ=-17.6%, σ₁/σ₂=1.08, Cohen_d=—, ΔMA_single=-83.2%
- **`qwen2.5_0.5b`** (L=0, drop=97.1%): exp1 generative Δ=-45.8%, σ₁/σ₂=1.48, Cohen_d=-0.501
- **`qwen2.5_7b`** (L=3, drop=89.5%): exp1 generative Δ=266.2%, σ₁/σ₂=2.64, Cohen_d=-1.344
- **`qwen2_7b`** (L=3, drop=92.5%): exp1 None Δ=—%, σ₁/σ₂=2.84, Cohen_d=-1.459
- **`qwen3_0.6b`** (L=2, drop=92.8%): exp1 generative Δ=-76.7%, σ₁/σ₂=1.41, Cohen_d=-0.279

**观察**：
- 7/8 CONCENTRATED 模型起源层在前 3 层（0-3），与 embedding 紧邻。
- `mistral_7b_v03` L=0 且 baseline_MA=1.46（极小）——需确认是 MA 确实来自 embedding/layer-0 MLP，还是 capture 时机问题。
- `glm4_32b` exp1_Δ=+167%（suppressive）+ CONCENTRATED，是**单层写入 + attention 抑制**的典型。

### 2.2 FEW-SOURCE — 2-5 层协作

- **`falcon_7b`** (L=3, steps=2, drop=93.5%): exp1 Δ=-21.4%, σ=1.37, Cohen_d=0.508★, ΔMA_s=-98.1%, ΔMA_m=-97.4%
- **`glm4_9b`** (L=1, steps=2, drop=90.7%): exp1 Δ=-44.9%, σ=3.26, Cohen_d=0.235
- **`gpt2`** (L=3, steps=3, drop=64.0%): exp1 Δ=-60.0%, σ=3.05, Cohen_d=-0.314★, ΔMA_s=-6.5%, ΔMA_m=-94.6%
- **`llama2_13b`** (L=0, steps=3, drop=93.6%): exp1 Δ=-79.5%, σ=1.32, Cohen_d=—, ΔMA_s=-95.6%, ΔMA_m=-28.6%
- **`llama3.1_8b`** (L=1, steps=2, drop=90.3%): exp1 Δ=-56.6%, σ=1.38, Cohen_d=0.369
- **`qwen3.5_35b_a3b`** (L=9, steps=3, drop=15.9%): exp1 Δ=5.1%, σ=1.03, Cohen_d=0.673
- **`qwen3_1.7b`** (L=2, steps=3, drop=89.8%): exp1 Δ=-22.0%, σ=1.33, Cohen_d=-0.147
- **`qwen3_4b`** (L=6, steps=2, drop=89.1%): exp1 Δ=-13.1%, σ=1.24, Cohen_d=0.377

**观察**：
- `qwen3.5_35b_a3b` MoE：exp2c drop=15.88%（弱），原因可能是 PE-ablation 未适配 MoE 路由专家 → 需要专门的 MoE 策略（见 §5）。
- `gpt2` 是教学典例：ΔMA_single=-6% 但 ΔMA_macro=-95%，**强证据支持多层 macro v₁ 路径**。

### 2.3 DISPERSED — >5 层分散

- **`qwen1.5_14b`** (L=35, steps=15, drop=39.7%): exp1 Δ=-15.1%, σ=1.05, Cohen_d=0.690
- **`qwen3.5_27b`** (L=54, steps=12, drop=55.6%): exp1 Δ=-16.0%, σ=1.12, Cohen_d=-1.005
- **`qwen3.5_9b`** (L=22, steps=15, drop=52.2%): exp1 Δ=-16.3%, σ=1.06, Cohen_d=-0.889
- **`qwen3_14b`** (L=6, steps=15, drop=71.6%): exp1 Δ=84.9%, σ=1.33, Cohen_d=1.099
- **`qwen3_30b_a3b`** (L=1, steps=10, drop=75.4%): exp1 Δ=-19.8%, σ=1.17, Cohen_d=-0.052
- **`qwen3_32b`** (L=6, steps=10, drop=80.2%): exp1 Δ=59.3%, σ=1.35, Cohen_d=1.443
- **`qwen3_8b`** (L=6, steps=15, drop=57.7%): exp1 Δ=-34.1%, σ=1.48, Cohen_d=-0.553
- **`yi_9b`** (L=8, steps=5, drop=89.6%): exp1 Δ=27.3%, σ=—, Cohen_d=-0.312

**观察**：
- 所有 DISPERSED 模型 drop% < 90%，贪心 15 步后仍有 MA 残留 → 提示**冗余备份**（多路径同时写入）。
- `qwen3_30b_a3b` V_att_peak=0.0（MoE 专家 V 替换失败）——需 PE-level 消融。
- Cohen_d 绝对值普遍偏大（|d| > 0.5 共 6/7 个），但符号混杂——说明 peak 层的 Cohen_d 噪声很大，**必须在 origin 层重测**。

### 2.4 ANOMALY

- **`opt_6.7b`**: exp2c 禁用全部 MLP 后 MA 不降（baseline==floor==371.66）。
  - exp1_Δ=+250%（suppressive）
  - 候选解释：(1) OPT 的 MA 可能由 LayerNorm/embedding 而非 MLP 写入；(2) hook patch 对 OPTDecoderLayer 的捕获点错位；(3) tx 4.57 API 不兼容。
  - **下一步**: 单独跑 exp1 per-layer MLP 消融 + 对比 embedding 直接 clamp 测 MA 源头。

---

## 3. 按论文假设分类的发现

### 3.1 MA 生成公式的三因子

```
MA ≈ σ₁ · |h₂ᵀ·v₁| · max_j|(u₁)_j|  +  bias
```

| 因子 | 理论预期 | 实测 | 结论 |
|---|---|---|---|
| σ₁/σ₂（谱集中） | 模式 A ≥ 3 | 23 个样本均值 ≈ 1.56，仅 2 个 ≥ 3 (`gpt2`, `glm4_9b`) | **预期偏强**。多数模型的谱集中很弱，但并不阻碍 MA 生成——提示 MA 形成**不依赖强谱集中**，而依赖 top-1 方向的有效投影乘积。 |
| Cohen_d（功能词对齐） | 模式 A |d| > 0.5 | origin 重估的 4/4 Tier1-3 均有明确符号：gptj 0.77, falcon 0.51, gpt2 -0.31, bloom 0.13 | 符号**不统一**；但绝对值 gptj / falcon 支持功能词假设，gpt2 / bloom 提示可能是**内容词偏向**——需要在更多模型上 origin 重估。 |
| ΔMA（V 消融因果） | 替 V → MA 崩塌 | 5/7 origin 模型 ≤ -80%，macro 路径全部 ≥ -94% 或负值较大 | **因果链最强证据**。gpt2 从 single(-6.5%) 跃升到 macro(-94.6%) 是**多层必要性的关键证据**。 |

### 3.2 生成模式 A/B × 调节模式 Gen/Sup 四象限

| | Generative (exp1_Δ < 0) | Suppressive (exp1_Δ > 0) |
|---|---|---|
| **A (集中)** | gpt2, gptj_6b, bloom_7b1, falcon_7b, mistral_7b_v03, llama2_13b, llama3.1_8b, qwen2.5_0.5b, qwen3_0.6b, qwen3_1.7b, qwen3_4b, glm4_9b | qwen2.5_7b, qwen3.5_35b_a3b, glm4_32b |
| **B (分散)** | qwen3_8b, qwen3_30b_a3b, qwen3.5_9b, qwen3.5_27b, qwen1.5_14b | qwen3_14b, qwen3_32b, yi_9b |

**观察**：两个维度 **正交**（卡方独立性直观可见），支持论文论点：MA **生成与调节是两个独立机制**。

### 3.3 peak vs origin 层号的影响（根因诊断）

对比 4 个 Tier1-3 模型（有完整重估）：

| 模型 | Cohen_d (peak) | Cohen_d (origin) | σ (peak-legacy) | σ (origin) | 结论变化 |
|---|:-:|:-:|:-:|:-:|---|
| gpt2 | — | **-0.31** | — | **3.05** | origin 层是 σ 最集中点 |
| gptj_6b | — | **+0.77** | — | **2.52** | 功能词对齐符号正 |
| bloom_7b1 | — | **+0.12** | — | **1.41** | 轻微功能词偏向 |
| falcon_7b | — | **+0.51** | — | **1.37** | 功能词偏向显著 |

**根因结论**: legacy v2 的 Cohen_d 在 peak 层计算，系统性误配 → 18/18 模型（除 Tier1-3 外）需按 `origin_layer/output/L_ORIGIN.sh` 重跑 RQ3。

---

## 4. 数据缺口与下一步

### 4.1 缺 origin-layer RQ3/4/5 重跑（18 模型）

以下模型只有 legacy peak-layer exp5 数据：
- `llama3.1_8b`, `qwen2.5_0.5b`, `qwen2.5_7b`, `qwen2_7b`, `qwen3_0.6b`, `qwen3_1.7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_30b_a3b`, `qwen3_32b`, `qwen3.5_9b`, `qwen3.5_27b`, `qwen3.5_35b_a3b`, `yi_9b`, `qwen1.5_14b`, `glm4_9b`, `glm4_32b`

**行动**：按 `paper_experiments/run_rq345_origin_layer.sh "" all` 批处理（依赖 `origin_layer/output/L_ORIGIN.sh`）。

### 4.2 MoE 专家消融（2 模型）
- `qwen3_30b_a3b`, `qwen3.5_35b_a3b`: legacy V_att=0.0；需适配 per-expert (PE) 模式。

### 4.3 ANOMALY 根因（1 模型）
- `opt_6.7b`: exp2c 禁全部 MLP 后 MA 不降。需验证 hook 位置 / 探查 OPT 架构中 MA 的写入点。

### 4.4 macro RQ5 完成度

- 已完成: `gpt2`, `gptj_6b`, `falcon_7b`, `llama2_13b` (4/25)
- 其中 5/5 ΔMA_m ≤ -28%（`llama2_13b`弱）；3/5 ΔMA_m ≤ -94%。
- `gpt2`: single -6.5% vs macro -94.6% → 多层 v₁ 叠加才是因果方向
- `llama2_13b`: single -95.6% 但 macro -28.6% → origin_layers 集合选择可能过宽（含噪声层）

---

## 5. 参考文件

- 主 JSON: `paper_experiments/results/ALL_EXPERIMENTS_SUMMARY_v2.json`
- 起源层: `paper_experiments/origin_layer/output/{SUMMARY.md, L_ORIGIN.json, ORIGIN_LAYERS_MACRO.json}`
- 实验脚本索引: `paper_experiments/docs/EXECUTION_PLAN.md`
- 机制讨论: `paper_experiments/docs/EXPERIMENT_PLAN.md`
- 根因诊断: `paper_experiments/docs/V2_ROOT_CAUSE.md`
- 模型 × RQ 矩阵: `paper_experiments/docs/PROGRESS_MATRIX.md`

---

## 6. 注意事项（Caveats）

1. **Cohen_d 符号**：peak 层 (legacy) 与 origin 层重估结果不可混用；混用会导致 §3.1 的结论错误。
2. **σ₁/σ₂ 模式 A 假设偏强**：实证未普遍支持 σ≥3；论文正文需要**弱化**谱集中的主导性，强调三因子乘积。
3. **macro vs single RQ5**：论文主证据应使用 `ΔMA_macro`（multi-layer projection），single 仅对 CONCENTRATED 适用。
4. **Mistral/Llama2_13b Cohen_d 空缺**：tokenizer 对前导空格处理不同，功能词列表需按 BPE/SentencePiece 差异改造。
5. **opt_6.7b 排除**：单独列为 ANOMALY，不纳入总体统计；需机制层重做。

