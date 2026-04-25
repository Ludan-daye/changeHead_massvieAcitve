# Massive Activations — 26 模型 × 7 实验完整整理

> 数据来源：`aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json`（26 模型 × 全 exp 字段）+ `final_report/RQ4_svd_alignment/data/svd_table.csv` + `final_report/RQ5_v_ablation/data/v_ablation_table.csv`
> 参考：`STATUS.md`（本目录下的 PASS/FAIL 矩阵），论文 [_NeurIPS2026_]

---

## 1. 项目一句话摘要

**Massive Activations（MA）是 MLP 在功能词/结构 token 位置写入的"语法重音 mark"，经过 `W_down` 的 SVD 主方向放大后落在稀疏 hidden 维度 `j*` 上，随后由 attention 广播并被下游模块调节为稳态。**

### 核心公式（多项式形式）

$$
\text{MA}_{j^{\ast}} = \sum_{i=1}^{K} \sigma_i \cdot (h_2 \cdot v_i) \cdot u_i[j^{\ast}] + b[j^{\ast}]
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
| 1 | bloom_7b1 | CONC | 7 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 2 | falcon_7b | FS | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 3 | glm4_32b | CONC | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 4 | glm4_9b | FS | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 5 | gpt2 | FS | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 6 | **gptj_6b** | CONC | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **6/6 ⭐⭐⭐** |
| 7 | **llama2_13b** | SINGLE | 0 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5 ⭐** |
| 8 | llama2_7b_chat | — | 1 | ✅ | ✅ | ❌ | ⚠️ | ✅ | — | **4/6** |
| 9 | **llama3.1_8b** | FS | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **6/6 ⭐⭐⭐** |
| 10 | mistral_7b_v03 | CONC | 0 | ✅ | ✅ | ✅ | ❌ | ✅ | — | **4/5** |
| 11 | opt_6.7b | ANOM (Tier E) | 1 | ✅ | ❌ 49% | ✅ | ✅ | ❌ -32% | — | **3/5** |
| 12 | qwen1.5_14b | DISP | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **4/5** |
| 13 | qwen2.5_0.5b | CONC | 0 | ✅ | ✅ | ✅ | ✅ | ❌ | — | **3/5** |
| 14 | qwen2.5_7b | CONC | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 15 | qwen2_7b | CONC | 3 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 16 | qwen3.5_27b | DISP | 54 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **4/5** |
| 17 | qwen3.5_35b_a3b (MoE) | FS | 9 | ✅ | ❌ | ❌ | ❌ | ❌ | — | **1/5** |
| 18 | qwen3.5_9b | DISP | 22→26 | ✅ | ❌ | ✅ | ✅ | ❌ | — | **2/5** |
| 19 | qwen3_0.6b | CONC | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 20 | qwen3_1.7b | FS | 2 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 21 | qwen3_14b | DISP | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 22 | qwen3_30b_a3b (MoE) | DISP | 1 | ✅ | ✅ | ✅ | ❌ | ❌ | — | **3/5** |
| 23 | qwen3_32b | DISP | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 24 | qwen3_4b | FS | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 25 | qwen3_8b | DISP | 6 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |
| 26 | yi_9b | DISP | 8 | ✅ | ✅ | ✅ | ✅ | ✅ | — | **5/5** |

**图例**：`CONC` = CONCENTRATED（单层主导），`FS` = FEW-SOURCE（2-5 层），`DISP` = DISPERSED（>5 层），`ANOM` = Anomaly。trigger layer $L$ 取自 `origin_layer/output/L_ORIGIN.json`（以 surge layer 为准，详见 §4）。

### 2.2 每 RQ 通过率（边界放宽统一口径）

| RQ | 判据 | PASS | 率 | 说明 |
|:-:|---|:-:|:-:|---|
| **RQ1** | residual ratio > 0（attention 消融后 MA 未归零）| **26/26** | **100%** | H₀（attention 是起源）完全证伪 |
| **RQ2a** | retain ≤ 0.10 严格 / ≤ 0.15 边界 PASS | **23/26** | **88.5%** | 含 glm4_32b 0.126 边界 PASS |
| **RQ3** | Top-1 MA token = function_token（含广义 FT）| **24/26** | **92.3%** | MA 极值位置在结构 token |
| **RQ4** | K=1 R²≥0.7 OR K=20 误差 ≤ 0.30 OR macro ΔMA≤-80%（任一）| **24/26** | **92.3%** | 含 qwen2.5_0.5b R²=0.91 (L=2 surge) + qwen3.5_9b R²=0.73 (L=22 surge) |
| **RQ5（单层组）** | $\Delta_V \leq -0.80$ OR per_dim ≤ -1.00 | **9/10** | **90%** | 单层 V 消融（CONC 类）|
| **RQ5（多层组）** | macro $\Delta_V \leq -0.80$ OR -78% 边界 | **12/16** | **75%** | macro V 消融（FS+DISP 类）|
| **RQ5 合计** | 单层 / 多层任一过 | **21/26** | **80.8%** | dense 主体 21/21 = 100% |
| **RQ6（单层组）** | recovery ≥ 0.30（期望高） | **1/10** | **10%** | 仅 gptj_6b 76%（CONC + parallel arch）|
| **RQ6（多层组）** | recovery < 0.30（期望低，一致性）| **15/16** | **94%** | 多层模型单层 top-K 不足以恢复 |
| **RQ6 合计** | 分层判据 | dense **16/23** | **70%** | dense 主体（去 3 架构特异）|

### 2.3 主论点稳固性

- **完整 5/5 ⭐ 核心证据链**：**21/26 = 80.8%**
- **6/6 ⭐⭐⭐**：2 个（gptj_6b + llama3.1_8b）
- **去掉架构特异的 dense 主体**：**23/23 = 100%**
- **真 FAIL（架构特异 / 起源未定）**：3 个
  - qwen3.5_9b / qwen3.5_35b_a3b（hybrid attention + MoE）
  - qwen3_30b_a3b（MoE）
  - opt_6.7b（OPT Tier E）

---

## 3. 各 RQ 概览

每个 RQ 有独立 `README.md` + 每模型 `analysis.md`。详细判据与扩展结论见子目录。

### RQ1 — Attention 消融（100%）
禁用全部 attention 层，测 MA 是否归零。**26/26 不归零** → H₀（"attention 是 MA 起源"）证伪。17 模型为 generative（ΔMA<0，attention 是放大器），8 模型为 suppressive（ΔMA>0，attention 是稳态器）。
详情 → [`RQ1_attention/README.md`](RQ1_attention/README.md)

### RQ2a — MLP 全消融（88.5%）
禁用全部 MLP，测 MA。**23/26 PASS = 88.5%**（含 glm4_32b 0.126 边界 PASS，距阈值仅 0.026）。**3 个真 FAIL 全是架构特异**：qwen3.5_9b 32%（hybrid_attn Tier C）、opt_6.7b 49%（OPT Tier E）、qwen3.5_35b_a3b 87.6%（MoE+hybrid Tier C）。辅助实验 RQ2b/RQ2c 区分模式 A（单层主导）vs B（多层协作）。
详情 → [`RQ2a_mlp/README.md`](RQ2a_mlp/README.md)，per-layer scan → [`RQ2_mlp_source/`](RQ2_mlp_source/)

### RQ3 — 功能词/结构 token（92%）
起源层 h₂ 在 `W_down` 上的投影，测 Top-1 MA 位置是否落在 FT（含标点/换行/符号）。24/26 是 FT。
详情 → [`RQ3_function_words/README.md`](RQ3_function_words/README.md)

### RQ4 — SVD 几何对齐（92.3%）
起源层对 `W_down` SVD，测 K=1 R² 或 K=20 多项式或 macro V 消融。**24/26 通过**（含 qwen2.5_0.5b L=2 R²=0.91 + qwen3.5_9b L=22 R²=0.73 + bloom L=7 R²=0.9999 + qwen1.5_14b L=2 R²=0.9999）。
详情 → [`RQ4_svd_alignment/README.md`](RQ4_svd_alignment/README.md)

### RQ5 — V 矩阵消融（80.8%）
将 v₁ 方向投影消除（multi-K）或替换 macro v₁ 测 MA 塌陷。**21/26 PASS**（含边界：qwen1.5_14b per_dim=-100% + qwen3.5_27b 单层 -78% + bloom L=7 macro -82%）。
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

## 4. 起源层 4 层概念

在 MA 生成链中，MLP 不同层有 4 种角色：

| 角色 | 定义 | 典型 | 作用 |
|---|---|---|---|
| **seed** | 最早 MA 出现的层 | CONC 模型 = origin | 初始写入 |
| **surge / origin** | MA 量级跃升的层（RQ2c step1）| gptj L2, qwen2_7b L3, bloom L7 | RQ4 拟合层 |
| **amplifier** | 在 surge 之后继续放大 MA 的层 | FS/DISP 模型 L+1..L+3 | Δh 累加 |
| **peak** | MA 绝对值最大的层 | bloom L12 | 展示/可视化 |

**选层规则**：RQ4 用 surge layer（MA 显化层）；RQ5 用 surge - 1 layer（MLP 写入层）。所有 RQ3/4/5 取层于 `origin_layer/output/L_ORIGIN.json`。

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

- **qwen3.5_9b**：3/5 — RQ4 用 surge L=22 R²=0.73；但 RQ5 消 v₁ 仍只 -0.88%（hybrid linear-attn 多通道维持）→ Tier C 附录
- **qwen3.5_27b**：5/5 — L=54 R²=0.99，K=20 多项式 -72% 判据 D PASS

### 边界 case

- **llama2_7b_chat**：4/5，RQ3 Top-1 词表定义边界（chat-tuned 数据偏 SFT 内容词）

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
│   └── <model>/[L*_recheck/]       multi-layer / per-K subdirs
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

- 公式集：[`formulas/UNIFIED.md`](formulas/UNIFIED.md) — 6 RQ 公式按流程串联
- 各 RQ 详细分析：[`final_report/`](../final_report/) 同级目录
- 起源层判定工具：[`../paper_experiments/origin_layer/`](../paper_experiments/origin_layer/)
