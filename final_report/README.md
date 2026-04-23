# Massive Activations (MA) 研究 — 最终报告

> **项目**：LLM 中 Massive Activations (MA) 的机制研究
> **时间**：2026-04 / 最终整理：2026-04-23
> **样本**：26 个 LLM 模型（Pre-Llama × 5 + Llama-family × 5 + Qwen × 12 + GLM × 2 + Yi × 1 + MoE × 2）

## 研究目标

理解 LLM 中出现的极端激活值（hidden state 中某些维度的激活值比中位数高 300-3000×）——**MA**——的生成机制。

## 核心论点（五步机制链）

```
(1) MLP 在特定位置写入带方向性的 h₂       [RQ2 + RQ3]
                  ↓
(2) h₂ 沿 W_down 的主奇异方向 v₁ 被 σ₁ 放大   [RQ4]
                  ↓
(3) 放大后落在 u₁ 稀疏维度 j*              [RQ4 + u₁ 分析]
                  ↓
(4) MA 在 j* 处形成（单层 CONCENTRATED 或多层 macro 协作）  [RQ2c 分类]
                  ↓
(5) Attention 把 MA 广播到所有 token（调节器而非生产者）[RQ1]
```

**MA 的主公式**：
$$
\text{MA}_{j^*} \approx \beta \cdot (h_2 \cdot v_1) + b, \quad \beta = \sigma_1 \cdot u_1[j^*]
$$

多奇异向量严格展开：$\text{MA}_{j^*} = \sum_{i=1}^{r} \sigma_i \cdot (v_i \cdot h_2) \cdot u_i[j^*]$

## 目录结构

```
final_report/
├── README.md                   ← 本文（总览 + 导航）
├── OVERVIEW.md                 ← 机制链详解 + 主论点
├── RQ1_attention_ablation/     ← Attention 消融（H₀ 证伪）
├── RQ2_mlp_source/             ← MLP 消融（H₁ 验证 + 起源层判定）
├── RQ3_function_token/         ← FT 位置 MA 定位（RQ4 中间证据）
├── RQ4_svd_alignment/          ← SVD 公式验证（MA 生成机制核心）
├── RQ5_v_ablation/             ← V 消融因果验证
├── RQ6_topk_recovery/          ← 单层/多层 MLP 恢复分析
├── HC_entropy/                 ← token 熵分布（双重稀疏假说）
├── u1_sparsity/                ← u₁ 方向稀疏度分析
└── aggregated/                 ← 跨 RQ 汇总 + 总表
```

## 每个 RQ 目录结构

```
RQ{N}/
├── ANALYSIS.md                 ← 实验目的/方式/判据/数据/结论/异常原因
├── data/                       ← 每模型一行的数据表（CSV/JSON）
└── analysis/                   ← 辅助分析脚本 + 中间结果
```

**实际代码和结果文件**在仓库的 `github_submission/experiments/RQ*/` 和 `paper_experiments/RQ*/` 下（不重复拷贝）。

## 26 模型完整清单

### 按架构家族分类

| 家族 | 模型 | 数量 |
|---|---|:-:|
| Pre-Llama | gpt2, gptj_6b, opt_6.7b, bloom_7b1, falcon_7b | 5 |
| Llama-base | llama2_13b, llama3.1_8b, mistral_7b_v03 | 3 |
| Llama-RLHF | llama2_7b_chat | 1 |
| Yi | yi_9b | 1 |
| GLM4 | glm4_9b, glm4_32b | 2 |
| Qwen 1.5/2/2.5 | qwen1.5_14b, qwen2_7b, qwen2.5_0.5b, qwen2.5_7b | 4 |
| Qwen3 dense | qwen3_0.6b, qwen3_1.7b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_32b | 6 |
| Qwen3.5 dense (hybrid-attn) | qwen3.5_9b, qwen3.5_27b | 2 |
| **MoE** | qwen3_30b_a3b (30B→3B), qwen3.5_35b_a3b (35B→3B) | 2 |

### 按 RQ2c 模式分类

| 模式 | 数量 | 模型 |
|---|:-:|---|
| CONCENTRATED (单层主导) | 8 | bloom_7b1, glm4_32b, gptj_6b, mistral_7b_v03, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b |
| FEW-SOURCE (2-5 层) | 8 | falcon_7b, glm4_9b, gpt2, llama2_13b, llama3.1_8b, qwen3.5_35b_a3b, qwen3_1.7b, qwen3_4b |
| DISPERSED (>5 层) | 8 | qwen1.5_14b, qwen3.5_27b, qwen3.5_9b, qwen3_14b, qwen3_30b_a3b, qwen3_32b, qwen3_8b, yi_9b |
| ANOMALY | 1 | opt_6.7b (MLP 消融也无效) |
| 无分类 | 1 | llama2_7b_chat |

## 各 RQ 核心结论（一句话）

| RQ | 结论 | PASS 率 |
|:-:|---|:-:|
| **RQ1** | attention 不是 MA 起源（关 attention 后 MA 都有残留）| 26/26 (100%) |
| **RQ2** | MLP 是 MA 主要来源（retain ≤ 10%）| 21/24 dense (87%) |
| **RQ3** | MA 极值位置 Top-1 是 function_token | 24/26 (92%) |
| **RQ4** | MA = β·(h₂·v₁) + b 公式成立（分层判据）| 14-17/26 (54-65%) |
| **RQ5** | 消除 v₁ 方向后 MA 塌陷（因果验证）| 17/26 (65%) |
| **RQ6** | 保留起源层能恢复部分 MA（recovery ≥ 30%）| ~5/26 (20%) |

## 关键文档

- **[OVERVIEW.md](OVERVIEW.md)** — 完整机制链 + 公式推导
- **[RQ1_attention_ablation/ANALYSIS.md](RQ1_attention_ablation/ANALYSIS.md)** — attention 调节器
- **[RQ2_mlp_source/ANALYSIS.md](RQ2_mlp_source/ANALYSIS.md)** — MLP 起源 + 模式分类
- **[RQ3_function_token/ANALYSIS.md](RQ3_function_token/ANALYSIS.md)** — FT 位置锚点
- **[RQ4_svd_alignment/ANALYSIS.md](RQ4_svd_alignment/ANALYSIS.md)** — MA 生成公式
- **[RQ5_v_ablation/ANALYSIS.md](RQ5_v_ablation/ANALYSIS.md)** — 因果验证
- **[RQ6_topk_recovery/ANALYSIS.md](RQ6_topk_recovery/ANALYSIS.md)** — 恢复分析
- **[aggregated/SUMMARY_26_MODELS.md](aggregated/SUMMARY_26_MODELS.md)** — 跨 RQ 汇总表

## 数据完整度

| RQ | 完整度 | 备注 |
|:-:|:-:|---|
| RQ1 | **26/26** | ✅ |
| RQ2a | 24/26 | ⏳ llama2_13b, opt_6.7b 补跑中 |
| RQ3 | 24/26 (exp5 alignment) + 26/26 (RQ4 sample) | ⚠️ qwen3.5_9b/27b RQ3 脚本 bug |
| RQ4 | **26/26** | ✅ |
| RQ5 | **26/26** | ⚠️ llama2_7b_chat hook 失效待修 |
| RQ6 | **26/26** | ✅ baseline 错层 bug 已修 |
| HC | 26/26 | ✅ |
| u₁ | 26/26 (token list) / 16/26 (W_down SVD) | ⚠️ 11 模型缺 W_down SVD |

## 代码与原始数据位置

- **代码**：`paper_experiments/RQ{1-6}_*/` + `RQ{N}_.../exp*.py`
- **修复后代码**：`paper_experiments/fixes/` (B1-B19 bug 修复版本)
- **V2 JSON 汇总**：`github_submission/aggregated/ALL_EXPERIMENTS_SUMMARY_v2.json`
- **26 × 7 实验真数据**：`github_submission/experiments/RQ*/results/<model>/`

## 里程碑

| 日期 | 事件 |
|:-:|---|
| 2026-04-17 | 启动 MA 研究，构建基础框架 |
| 2026-04-19 | 起源层自动判定工具上线 |
| 2026-04-20 | 发现 V2 错层根因 + RQ2c 贪心累积消融 |
| 2026-04-21 | 7 个脚本 bug（B1-B7）统一修复 |
| 2026-04-22 | 26 × 7 实验全真数据补齐（MoE hook 修复）|
| 2026-04-23 | **最终整理**：RQ1-6 结论定稿 + 公式优化 V3 |

---

**最后更新**：2026-04-23
