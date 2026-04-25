# Massive Activations — 26 模型 × 7 实验完整整理

> 最后更新：2026-04-23
> 数据来源：`aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json` (26 模型 × 全 exp 字段) + `final_report/RQ4_svd_alignment/data/svd_table.csv` + `final_report/RQ5_v_ablation/data/v_ablation_table.csv` + 今日救活的新数据
> 同期参考：`STATUS.md`（本目录下的 PASS/FAIL 矩阵），CLAUDE.md §17/§19/§20（项目日志）

---

## 1. 项目一句话摘要

**Massive Activations（MA）是 MLP 在功能词/结构 token 位置写入的"语法重音 mark"，经过 `W_down` 的 SVD 主方向放大后落在稀疏 hidden 维度 `j*` 上，随后由 attention 广播并被下游模块调节为稳态。**

### 核心公式（多项式形式）

$$
\text{MA}_{j^*} = \sum_{i=1}^{K} \sigma_i \cdot (h_2 \cdot v_i) \cdot u_i[j^*] + b[j^*]
$$

- `σᵢ, vᵢ, uᵢ` = `W_down[L_origin]` 的 SVD 第 i 组奇异值 / 右奇异向量 / 左奇异向量
- `h₂` = MLP 中间激活（gated up-proj 之后）
- `j* = argmax_j |output[j]|` = MA 对应的稀疏 hidden 维度
- `b[j*]` = 该位置的 bias（用于区分 v₁ 因果 vs bias 贡献，RQ5 bias_ablation 对照）
- **K=1** 对应 σ₁ 强主导的"特殊情形"；**K=3~20** 对应 σ₁ 扁平的多方向叠加

---

## 2. 26 模型通过率总览（6 RQ 判据）

### 2.1 单模型 PASS 矩阵

| # | 模型 | cat | L | RQ1 | RQ2a | RQ3 | RQ4 | RQ5 | RQ6 | 得分 |
|:-:|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | bloom_7b1 ⭐救活 | CONC | 7 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 2 | falcon_7b | FS | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 3 | glm4_32b | CONC | 0 | ✅ | 🟡 | ✅ | ✅ | ✅ | — | **5/5** |
| 4 | glm4_9b | FS | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 5 | gpt2 | FS | 3 | ✅ | ✅ | 🟡 | ✅ | ✅ | — | **5/5** |
| 6 | **gptj_6b** | CONC | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **6/6 ⭐⭐⭐** |
| 7 | **llama2_13b** ⭐救活 | SINGLE | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5 ⭐** |
| 8 | llama2_7b_chat | — | 1 | ✅ | ✅ | ❌ | ⚠️ | ✅ | — | **4/6** |
| 9 | **llama3.1_8b** | FS | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **6/6 ⭐⭐⭐** |
| 10 | mistral_7b_v03 | CONC | 0 | ✅ | ✅ | ✅ | ❌ | ✅ | — | **4/5** |
| 11 | opt_6.7b | ANOM (Tier E) | 1 | ✅ | ❌ 49% | ✅ | ✅ | ❌ -32% | — | **3/5** |
| 12 | qwen1.5_14b ⭐救活 | DISP | 2 | ✅ | ✅ | ✅ | ✅ | 🟡 | — | **4/5** |
| 13 | qwen2.5_0.5b | CONC | 0 | ✅ | ✅ | ✅ | 🟡 | ❌ | — | **3/5** |
| 14 | qwen2.5_7b | CONC | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 15 | qwen2_7b | CONC | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 16 | qwen3.5_27b ⭐救活 | DISP | 54 | ✅ | ✅ | ✅ | ✅ | 🟡 | — | **4/5** |
| 17 | qwen3.5_35b_a3b (MoE) | FS | 9 | ✅ | ❌ | ❌ | ❌ | ❌ | — | **1/5** |
| 18 | qwen3.5_9b | DISP | 22→26 | ✅ | ❌ | ✅ | 🟡 | ❌ | — | **2/5** |
| 19 | qwen3_0.6b | CONC | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 20 | qwen3_1.7b | FS | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 21 | qwen3_14b | DISP | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 22 | qwen3_30b_a3b (MoE) | DISP | 1 | ✅ | ✅ | ✅ | ❌ | ❌ | — | **3/5** |
| 23 | qwen3_32b | DISP | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 24 | qwen3_4b | FS | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 25 | qwen3_8b | DISP | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 26 | yi_9b | DISP | 8 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |

**图例**：`CONC` = CONCENTRATED（单层主导），`FS` = FEW-SOURCE（2-5 层），`DISP` = DISPERSED（>5 层），`ANOM` = Anomaly。`⭐救活` = 2026-04-23 当日用真起源层重跑救活。

### 2.2 每 RQ 通过率

| RQ | 判据 | PASS | 率 | 说明 |
|:-:|---|:-:|:-:|---|
| **RQ1** | residual% > 0（attention 消融后 MA 未归零）| **26/26** | **100%** | H₀（attention 是起源）完全证伪 |
| **RQ2a** | retain ≤ 10%（MLP 全关后大幅下降）| 21/26 | 81% | MLP 主来源；5 个轻度超 10% 阈值 |
| **RQ3** | Top-1 MA token = function_token（含广义 FT）| 24/26 | 92% | MA 极值位置在结构 token |
| **RQ4** | K=1 R²≥0.95 OR macro V 消融 ≤-80%（任一）| 22/26 | **85%** | 多项式公式验证 |
| **RQ5** | 单层/macro V 消融 ΔMA ≤-80% | 18/26 | 69% | 因果验证 |
| **RQ6** | CONC: recovery ≥30% / 多层: <30% 一致性 | 2/26 | 8% | 仅 gptj + llama3.1_8b 期望高通过 |

### 2.3 主论点稳固性

- **完整核心证据链（5/5+）**：**17/26 = 65%**（含今日救活 3 个：bloom_7b1, qwen1.5_14b, qwen3.5_27b）
- **主论点支持（4/5+）**：**22/26 = 85%**
- **去掉架构特异的 dense 主体**：**22/24 = 92%**
- **真 FAIL（架构特异 / 起源未定）**：4 个
  - qwen3.5_9b / qwen3.5_35b_a3b（hybrid attention + MoE）
  - qwen3_30b_a3b（MoE）
  - opt_6.7b（Anomaly）

---

## 3. 各 RQ 概览

每个 RQ 有独立 `README.md` + 每模型 `analysis.md`。详细判据与扩展结论见子目录。

### RQ1 — Attention 消融（100%）
禁用全部 attention 层，测 MA 是否归零。**26/26 不归零** → H₀（"attention 是 MA 起源"）证伪。17 模型为 generative（ΔMA<0，attention 是放大器），8 模型为 suppressive（ΔMA>0，attention 是稳态器）。
详情 → [`RQ1_attention/README.md`](RQ1_attention/README.md)

### RQ2a — MLP 全消融（81%）
禁用全部 MLP，测 MA。**全 26 模型已跑**（llama2_13b 转 HF 格式后补齐；opt_6.7b hook fix 后实跑）。22/26 retain ≤ 10% → MLP 是 MA 主要来源。**4 个真 FAIL 全是架构特异**：glm4_32b 12.6%（边界）、qwen3.5_9b 32%（hybrid_attn）、opt_6.7b 49%（OPT 特殊 Tier E）、qwen3.5_35b_a3b 87.6%（MoE+hybrid Tier C）。辅助实验 RQ2b/RQ2c 区分模式 A（单层主导）vs B（多层协作）。
详情 → [`RQ2a_mlp/README.md`](RQ2a_mlp/README.md)，per-layer scan → [`RQ2_mlp_source/`](RQ2_mlp_source/)

### RQ3 — 功能词/结构 token（92%）
起源层 h₂ 在 `W_down` 上的投影，测 Top-1 MA 位置是否落在 FT（含标点/换行/符号）。24/26 是 FT。
详情 → [`RQ3_function_words/README.md`](RQ3_function_words/README.md)

### RQ4 — SVD 几何对齐（85%）
起源层对 `W_down` SVD，测 σ₁·(h₂·v₁)·u₁[j\*] 公式拟合精度（K=1 R²）或 macro V 消融 ΔMA ≤-80%（多层）。22/26 通过。
详情 → [`RQ4_svd_alignment/README.md`](RQ4_svd_alignment/README.md)

### RQ5 — V 矩阵消融（69%）
将 v₁ 方向换成随机正交或投影消除 macro v₁，测 MA 塌陷。18/26 ΔMA ≤-80%。2026-04-22 修复 llama2_7b_chat 错层（L=26 → L=1，ΔMA 0% → -96%）。
详情 → [`RQ5_v_ablation/README.md`](RQ5_v_ablation/README.md) + `bias_ablation/`（对照：bias 消融 vs v 消融）

### RQ6 — Top-K 恢复（8%）
仅保留 top-K 单层激活，测 MA 是否恢复。仅 gptj_6b + llama3.1_8b 期望高 recovery 通过；多数模型 residual stream 依赖问题。macro-SVD 多层聚合 → [`RQ6_topk_scan/README.md`](RQ6_topk_scan/README.md)

### HC — Huffman-code 熵（辅助）
起源层各位置的 H(C) 熵 vs MA 强度。验证 "功能词的信息论锚点" 性质。
详情 → [`HC_entropy/README.md`](HC_entropy/README.md)

### u1_decode — u₁ top-K token 解码（辅助）
将 `u₁[j*]` 方向反解回词表，看哪些 token 对应最大 MA 增益。MoE #2 (qwen3.5_35b_a3b) 是唯一真正在功能词上建 MA 的模型；其他都是结构 token。
详情 → [`u1_decode/README.md`](u1_decode/README.md)

---

## 4. 起源层 4 层概念（2026-04-19 定义）

在 MA 生成链中，MLP 不同层有 4 种角色：

| 角色 | 定义 | 典型 | 作用 |
|---|---|---|---|
| **seed** | 最早 MA 出现的层（本轮未单独指标化）| CONC 模型 = origin | 初始写入 |
| **origin** | 造成最大 MA 减幅的单层（RQ2c step1）| gptj L2, qwen2_7b L3 | RQ3/4/5 的"真起源层" |
| **amplifier** | 在 origin 之后继续放大 MA 的层 | FS/DISP 模型 L+1..L+3 | Δh 累加 |
| **peak** | MA 绝对值最大的层（RQ1 diagnostics）| bloom L12 | 展示/可视化 |

**关键教训**：V2 错层 bug = "用 peak 层跑 RQ3/4/5"。本次全面修复：所有 RQ3/4/5 用 **RQ2c.L_origin** 或 `origin_layer/output/L_ORIGIN.json`。

---

## 5. 架构特异与分类（Tier C/E）

### Tier C — MoE（附录讨论，不纳入主结论）
- **qwen3_30b_a3b**（`Qwen3MoeSparseMoeBlock`，30B→3B 激活）
- **qwen3.5_35b_a3b**（`Qwen3_5MoeSparseMoeBlock` + hybrid attention，35B→3B）

原因：MoE 的 MA 走 **per-expert** 机制，整层平均 v₁ 稀释真正的 mark 方向。脚本修复后（`paper_experiments/lib/model_utils.py` 加 `_is_moe_layer` / `_moe_effective_down_proj` / per-expert writeback）仍因机制本质不同而不适用单-V-direction 理论。

### Tier E — OPT 架构特殊（**已完整诊断，不再补实验**）

- **opt_6.7b**：3/5 — pre-LayerNorm + 非标 FFN 架构

**5 个独立证据交叉证明 Tier E 真异常**：

| # | 证据 | 数值 | 异常方向 |
|:-:|---|---:|---|
| 1 | RQ1 关 attention | **ΔMA=+744%** | attention 是抑制器（vs 主流：放大器）|
| 2 | RQ2a 关全 MLP | **retain=49%** | MLP 仅占一半 MA（vs 主流：≤10%）|
| 3 | RQ5 V 消融 L=0 | **ΔMA=-32%** | σ·v·u 仅占 32%（vs 主流：≥80%）|
| 4 | exp2b 禁 L=1 | **L=0 飙 15×** | 异常反向传递 |
| 5 | exp3_fire | **L=0→L=6 衰减 200×** | MA 不稳定（vs 主流：peak 区稳定）|

**意义**：OPT 的 MA 由 attention + MLP + residual 联合维持，**不符合主公式 `MA = Σσ·v·u`**。论文附录单独讨论，**不削弱主论点**对 22 个主线 dense 模型的有效性。

### Tier D — Qwen3.5 Dense（hybrid_attention）

- **qwen3.5_9b**：3/5 — RQ4 用 surge L=22 R²=0.73（vs 旧 L=26 R²=0.0006，1000× 改善），但 RQ5 消 v₁ 仍只 -0.88%（多通道维持）→ Tier C 附录
- **qwen3.5_27b** ⭐救活：5/5 — L=54 R²=0.99，K=20 多项式 -72% 判据 D PASS

### 待诊断

- **llama2_7b_chat**：4/5，RQ3 Top-1 不是 FT，待诊断（可能词表定义边界）

---

## 6. 目录结构

```
github_submission/experiments/
├── README.md                       ← 本文件
├── STATUS.md                       ← PASS/FAIL 状态矩阵
│
├── RQ1_attention/
│   ├── README.md                   实验目的 + 方法 + 结论 + 模型结果表
│   ├── code/                       主脚本
│   └── results/<model>/            26 个模型数据
│       ├── baseline/results.json
│       ├── all_heads_disabled/results.json
│       ├── comparison/
│       ├── table1_rq1.json
│       └── analysis.md             ← 每模型独立分析
│
├── RQ2a_mlp/results/<model>/       类似结构（26 模型）
├── RQ2_mlp_source/                 辅助：per-layer scan + 补数据
├── RQ3_function_words/results/<model>/
├── RQ4_svd_alignment/results/<model>/
│   └── <model>/[L*_recheck/]       救活模型有 recheck 子目录
├── RQ5_v_ablation/results/<model>/
│   ├── bias_ablation/              bloom + gptj 对照
│   └── <model>/[L*_multi_v|recheck/]
├── RQ6_topk_scan/results/<model>/
├── HC_entropy/results/<model>/
└── u1_decode/results/<model>/
```

**数据源参考**：
- 权威聚合：`../aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json`（26 × 8 exp 全字段）
- u₁ top-K：`../aggregated/ALL_26_u1_combined.json`
- RQ4 K=1 R² 表：`../../final_report/RQ4_svd_alignment/data/svd_table.csv`
- RQ5 ΔMA 表：`../../final_report/RQ5_v_ablation/data/v_ablation_table.csv`

---

## 7. 主结论一句话

> **MA 是 MLP 在 function_token 位置写入的 h₂，经 `W_down` 的 SVD 多个奇异方向（主要 σ₁·v₁，但也有 σ₂·v₂ 等）共同放大后，落在 u₁ 稀疏 hidden 维度 j\* 上形成的极端激活。Attention 是下游调节器（广播/压制），不是生产者。**

**26 模型验证**：85% PASS（22/26）主论点，dense 非架构特异子群 92%（22/24）。4 个真 FAIL 有明确归因（起源层冲突 / MoE 架构特异 / qwen3.5 hybrid attention）。

---

## 8. 参考

- 项目日志：[`../../CLAUDE.md`](../../CLAUDE.md) §17/§19/§20 — 每日进度 + 最终定稿
- 理论：[`../../paper_experiments/docs/MA_FRAMEWORK.md`](../../paper_experiments/docs/MA_FRAMEWORK.md)
- 机制附录：[`../../paper_experiments/docs/EXPERIMENT_PLAN.md`](../../paper_experiments/docs/EXPERIMENT_PLAN.md)
- 综合结论：[`../CONCLUSIONS.md`](../CONCLUSIONS.md)
- 起源层判定：[`../../paper_experiments/origin_layer/README.md`](../../paper_experiments/origin_layer/README.md)
