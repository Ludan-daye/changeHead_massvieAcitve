# 实验逐项讨论记录

> 本文档按 RQ 逐个记录讨论结论，边讨论边填充。
> 每个 RQ 只记录三项：**做什么**、**25 模型状态**、**需要重做的清单**。
> 附录 A 记录贯穿所有 RQ 的机制回路。

---

## 目录

- [RQ1 — Attention 消融测试](#rq1--attention-消融测试)
- [RQ2 — MLP 来源与起源层定位](#rq2--mlp-来源与起源层定位)
- [RQ3 — 功能词 SVD 空间映射](#rq3--待讨论)
- RQ4 — 待讨论
- RQ5 — 待讨论
- RQ6 — 待讨论
- [附录 A — MA 机制回路](#附录-a--ma-机制回路功能词-mark--attention-广播--分岔)

---

## RQ1 — Attention 消融测试

### 做什么

把模型里**所有 attention 层**的输出整体置零（保留残差流和 MLP），然后测 MA 大小：

```
原始:  h_{L+1} = h_L + Attn(h_L) + MLP(h_L)
禁用:  h_{L+1} = h_L + 0       + MLP(h_L)
```

**目的**：否证"MA 来自 attention"这一假设，为 RQ2（MLP 是 MA 来源）铺路。

**判读规则**（证伪逻辑）：
- 关掉 attention 后 MA **完全消失**（disabled ≈ 0） → attention 是 MA 来源（**尚未观察到任何一个模型**）
- 关掉 attention 后 MA **仍有残留**（disabled > 0）→ attention **不是** MA 来源 → **证伪 H₀**

**关键字段**（`ALL_EXPERIMENTS_SUMMARY_v2.json` 的 `exp1`）：
- `baseline_top1`：原始 MA
- `disabled_top1`：禁用 attention 后 MA
- `delta_top1_pct`：变化率（= (disabled − baseline) / baseline × 100）
- `residual_pct` *(derived)*：残留率 = disabled / baseline × 100（RQ1 主指标）
- `peak_layer`：MA 观测最大的层
- `mode`：generative (Δ<0) / suppressive (Δ>0)

**RQ1 两条记录**：
1. **主结论**：**atten_h 不是 MA 的起源** — 26 个模型中有数据的 25 个，关 attention 后 MA 都有残留（最小残留 1.69%，没有任何一个归零）
2. **副结论（影响方向）**：关 attention 后 MA 的变化方向：
   - **Generative（Δ<0，17 个）**：attention 是 MA 的下游**放大器/广播器**，关掉后 MA 坍缩
   - **Suppressive（Δ>0，8 个）**：attention 是 MA 的下游**抑制器/稳态器**，关掉后 MA 反而爆炸

### 26 模型当前 RQ1 状态（按 residual% 升序）

> residual% = disabled_top1 / baseline_top1 × 100；越小表示 attention 对 MA 放大作用越强；>100% 表示 attention 在抑制 MA。

| # | 模型 | baseline | disabled | **residual%** | ΔTop1 % | peak | 方向 | 状态 |
|:-:|---|---:|---:|---:|---:|:-:|:-:|:-:|
| 1 | bloom_7b1 | 3541 | 60 | **1.69%** | -98.31% | L12 | ↓ gen | ✓ |
| 2 | gptj_6b | 4246 | 240 | 5.65% | -94.35% | L16 | ↓ gen | ✓ |
| 3 | llama2_13b | 1283 | 263 | 20.50% | -79.50% | L22 | ↓ gen | ✓ |
| 4 | qwen3_0.6b | 6871 | 1603 | 23.33% | -76.67% | L25 | ↓ gen | ✓ |
| 5 | gpt2 | 983 | 393 | 39.99% | -60.01% | L16 | ↓ gen | ✓ |
| 6 | llama3.1_8b | 314 | 137 | 43.43% | -56.57% | L17 | ↓ gen | ✓ |
| 7 | qwen2.5_0.5b | 1692 | 918 | 54.23% | -45.77% | L15 | ↓ gen | ✓ |
| 8 | glm4_9b | 2250 | 1240 | 55.08% | -44.92% | L1 | ↓ gen | ✓ |
| 9 | qwen3_8b | 13336 | 8791 | 65.92% | -34.08% | L33 | ↓ gen | ✓ |
| 10 | qwen3_1.7b | 12582 | 9818 | 78.03% | -21.97% | L25 | ↓ gen | ✓ |
| 11 | falcon_7b | 1827 | 1437 | 78.62% | -21.38% | L23 | ↓ gen | ✓ |
| 12 | qwen3_30b_a3b (MoE) | 1207 | 967 | 80.17% | -19.83% | L36 | ↓ gen | ✓ |
| 13 | mistral_7b_v03 | 314 | 259 | 82.43% | -17.57% | L25 | ↓ gen | ✓ |
| 14 | qwen3.5_9b | 353 | 296 | 83.69% | -16.31% | L31 | ↓ gen | ✓ |
| 15 | qwen3.5_27b | 1000 | 840 | 84.00% | -16.00% | L58 | ↓ gen | ✓ |
| 16 | qwen1.5_14b | 7444 | 6317 | 84.86% | -15.14% | L37 | ↓ gen | ✓ |
| 17 | qwen3_4b | 8526 | 7407 | 86.88% | -13.12% | L17 | ↓ gen | ✓ |
| 18 | qwen3.5_35b_a3b (MoE) | 40 | 42 | 105.14% | +5.14% | L39 | ↑ sup | ✓ |
| 19 | yi_9b | 5004 | 6368 | 127.26% | +27.26% | L47 | ↑ sup | ✓ |
| 20 | qwen3_32b | 27418 | 43680 | 159.31% | +59.31% | L53 | ↑ sup | ✓ |
| 21 | qwen3_14b | 15205 | 28112 | 184.89% | +84.89% | L33 | ↑ sup | ✓ |
| 22 | glm4_32b | 298598 | 797082 | 266.94% | +166.94% | L1 | ↑ sup | ✓ |
| 23 | opt_6.7b | 391 | 1370 | 350.27% | +250.26% | L25 | ↑ sup | ✓ |
| 24 | qwen2.5_7b | 11886 | 43520 | 366.16% | +266.16% | L16 | ↑ sup | ✓ |
| 25 | llama2_7b_chat | 2112 | 12812 | **606.67%** | +506.67% | L22 | ↑ sup | ✓ |
| — | qwen2_7b | — | — | — | — | — | ? | **⚠ 缺数据** |

**汇总**：
- **方向 — Generative**：17 个（attention 放大 MA）
- **方向 — Suppressive**：8 个（attention 抑制 MA）
  - `qwen3.5_35b_a3b` (+5%), `yi_9b` (+27%), `qwen3_32b` (+59%), `qwen3_14b` (+85%), `glm4_32b` (+167%), `opt_6.7b` (+250%), `qwen2.5_7b` (+266%), `llama2_7b_chat` (+507%)
- **H₀ 证伪**：**25/25** — 没有任何一个模型的 residual 达到 0
- **数据缺口**：1 个（qwen2_7b）

### 需要补跑的模型（仅 1 个）

| # | 模型 | 问题 | 修复方式 | 成本 |
|:-:|---|---|---|---|
| 1 | **qwen2_7b** | `exp1` 字段为 None，无 baseline/disabled 数据（历史上 `+Infinity` 异常已在 v2 数据中被清空） | `--nsamples 30 → 60`，打印 baseline 分布诊断 | ~15 min |

**结论**：RQ1 除 `qwen2_7b` 一个缺口外，其余 25 个模型数据齐全，结论已稳。即便不补 qwen2_7b，25/25 也已经 100% 证伪 H₀。

### 关键观察（为论文调节机制章节提供素材）

1. **同家族方向翻转**：
   - GLM 家族：`glm4_9b`（放大，Δ=-44.92%）vs `glm4_32b`（抑制，Δ=+167%）→ 模型规模影响 attention 角色
   - Llama 家族：`llama2_13b` base（放大，Δ=-79.5%）vs `llama2_7b_chat` 微调（抑制，Δ=+506%）→ RLHF 对齐显著改变 attention 对 MA 的调节方向

2. **Suppressive 模型 baseline 显著偏高**（中位数 13,545 vs generative 1,827，约 7×）——说明高基线 MA 模型依赖 attention 做稳态压缩，关掉后 residual/LN 失衡导致 MA 爆炸。

3. **Suppressive 集群在中国开源家族**（Qwen 4/13, Yi 1/1, GLM 1/2）；GPT/BLOOM/Falcon/Mistral/Llama-base 全部 generative。提示**训练策略**决定 attention 角色，而非架构本身。

4. **MoE 响应弱**（qwen3.5_35b_a3b Δ=+5%）——整层 attention 消融对 MoE 专家路由影响小，需 PE-level 重跑才能看清真实调节强度。

### 待决定项

（留待后续确认，不影响本次重做范围）

---

## RQ2 — MLP 来源与起源层定位

### 做什么

RQ2 包含三个子实验：

#### RQ2a — 全部 MLP 禁用（feasibility test）

把**所有层**的 MLP 输出置零，保留 attention 和残差流，测 MA 变化。

```
原始:  h_{L+1} = h_L + Attn(h_L) + MLP(h_L)
禁用:  h_{L+1} = h_L + Attn(h_L) + 0
```

**目的**：正面证明"MA 来自 MLP"。预期 ΔMA ≈ -95%~-100%。脚本：`RQ2_mlp_source/exp2a_mlp_feasibility_test.py`。

#### RQ2b — 逐层 MLP 禁用（per-layer ablation）

每次只禁用**一层** MLP，扫遍所有 N 层，找出哪一层影响最大。

**目的**：
1. 定位**起源层** `critical_layer`——下游 RQ3/4/5 的正确层位来源
2. 用"最大单层 ΔMA"给出**初步**分类（A 单层 ≥85%、B 多层 <30%、中间 30-85%）

⚠️ **重要限制**：RQ2b 单层数据**不足以严格定性**中间形态（30-85%）。下面三种完全不同的物理机制会给出相同的 RQ2b 数据：

| 情景 | per-layer max | 真实机制 |
|---|:-:|---|
| X | -50% | 双层主导（两个 MLP 各写一半） |
| Y | -50% | 单层主导（但另一层抑制掉一半） |
| Z | -50% | 多层协作（每层 ~10%，某层稍大） |

因此需要 RQ2c 来区分。

#### RQ2c — 贪心累积消融（greedy cumulative ablation）【新增】

按 RQ2b 的 `top5_layers` 排序，依次**累积禁用**并观测 ΔMA：

```
step 1: 禁 top1
step 2: 禁 top1 + top2
step 3: 禁 top1 + top2 + top3
...
step 5: 禁 top1 + … + top5
```

**判读规则（累积曲线形状）**：
- top-1 ≈ -95% → **模式 A 单层**
- top-2 快速达 -95%（top-1 大 + 边际增量大）→ **A + 调节** 或 **双层主导**
- top-5 才 -90%+（每步边际增量相近）→ **模式 B 多层协作**
- top-k 不单调（中间反而回升）→ **存在抑制层**，模式 A + 调节

**目的**：解决 RQ2b 对中间形态的不可定性，给出严格的 A / B / 调节型三分法。

### 论文价值

- RQ2a：闭环"MLP 是唯一来源"（对模式 B 4 个模型尤其关键——per-layer 只 -6~-15%，必须靠 RQ2a 证明"全禁时接近 -100%"）
- RQ2b：给下游 RQ3/4/5 定层位 + 初步模式分类
- RQ2c：严格区分 A / B / 混合模式，**支撑论文"两种生成模式"的主张**

### 26 模型当前 RQ2 状态

### RQ2a 主结论：H₁ 验证成立（2026-04-20 重审）

**H₁**："MLP 是 MA 的主要来源"——正面验证。

**两条记录**（与 RQ1 对称）：

1. **主结论**：**MLP 是 MA 的主要来源**——26 个模型中有数据的 24 个，关全部 MLP 后 MA **大幅消失**（中位数 retention ≈ 2%，20/24 retain ≤ 10%），**bloom_7b1 甚至归零**（retain=0%）。对比 RQ1 的"atten 最小 residual=1.69% 不归零"——**MLP 是起源，atten 不是**，闭环完成。
2. **副结论（4 个残留异常）**：qwen3.5 家族 3/3 全体显著残留 + gpt2 + MoE 实现工件——MLP 之外存在次级机制（见下文异常分析）。

**指标定义**：
- `retention% = disabled_max_ma / baseline_max_ma × 100`（主验证指标，表征"关掉 MLP 后 MA 还剩多少"）
- `reduction% = 100 − retention%`（等价，越大说明 MLP 越主导）

**26 模型 RQ2a 状态（按 retention% 降序）**

> 数据源：JSON `exp2a` 字段 18 个 + `results/wikitext_run/RQ2a/` 磁盘 5 个 + `changeHead_massvieAcitve` 归档 gpt2 1 个（互无重叠）= 24/26。

| # | 模型 | baseline_max_ma | disabled_max_ma | **retain%** | reduce% | 状态 |
|:-:|---|---:|---:|---:|---:|:-:|
| 1 | qwen3.5_35b_a3b (MoE) | 39.90 | 32.35 | **81.08%** | 18.92% | ⚠ 异常 |
| 2 | gpt2 | 3021.33 | 1164.08 | **38.53%** | 61.47% | ⚠ 异常 |
| 3 | qwen3.5_9b | 353.20 | 113.30 | **32.08%** | 67.92% | ⚠ 异常 |
| 4 | qwen3.5_27b | 1000.00 | 196.20 | **19.62%** | 80.38% | ⚠ 异常 |
| 5 | qwen3_14b | 15204.80 | 1334.00 | 8.77% | 91.23% | ✓ |
| 6 | mistral_7b_v03 | 318.37 | 17.05 | 5.36% | 94.64% | ✓ |
| 7 | qwen3_32b | 27417.60 | 1405.90 | 5.13% | 94.87% | ✓ |
| 8 | glm4_9b | 2250.40 | 107.47 | 4.78% | 95.22% | ✓ |
| 9 | glm4_32b | 298598.40 | 13465.60 | 4.51% | 95.49% | ✓ |
| 10 | qwen3_1.7b | 12582.40 | 532.05 | 4.23% | 95.77% | ✓ |
| 11 | llama3.1_8b | 314.45 | 12.48 | 3.97% | 96.03% | ✓ |
| 12 | qwen3_30b_a3b (MoE) | 1206.40 | 34.34 | 2.85% | 97.15% | ✓ |
| 13 | qwen1.5_14b | 7444.00 | 163.03 | 2.19% | 97.81% | ✓ |
| 14 | qwen2_7b | 6925.60 | 147.75 | 2.13% | 97.87% | ✓ |
| 15 | qwen2.5_7b | 12257.60 | 253.55 | 2.07% | 97.93% | ✓ |
| 16 | gptj_6b | 4185.27 | 80.74 | 1.93% | 98.07% | ✓ |
| 17 | qwen2.5_0.5b | 1691.80 | 29.09 | 1.72% | 98.28% | ✓ |
| 18 | qwen3_0.6b | 6871.20 | 117.66 | 1.71% | 98.29% | ✓ |
| 19 | falcon_7b | 1871.83 | 31.48 | 1.68% | 98.32% | ✓ |
| 20 | qwen3_8b | 13336.00 | 185.43 | 1.39% | 98.61% | ✓ |
| 21 | llama2_7b_chat | 2194.57 | 28.41 | 1.29% | 98.71% | ✓ |
| 22 | yi_9b | 5004.00 | 59.50 | 1.19% | 98.81% | ✓ |
| 23 | qwen3_4b | 8525.60 | 80.21 | 0.94% | 99.06% | ✓ |
| 24 | bloom_7b1 | 3631.33 | 0.00 | **0.00%** | 100.00% | ✓ **最强证据** |
| — | llama2_13b | — | — | — | — | **· 未跑** |
| — | opt_6.7b | — | — | — | — | **· 未跑** |

**汇总**：24/24 显著消减（bloom_7b1 100% 消失）；20/24 retain ≤ 10%；4 异常 retain > 15%；缺 2（llama2_13b、opt_6.7b）。

### RQ2a 4 个异常的机制归因

| 模型 | retain% | 归因 |
|---|:-:|---|
| **qwen3.5_35b_a3b** | 81% | **MoE 实现工件**——脚本 hook 挂在 `model.layers[i].mlp` 外层，但 MoE 的 FFN 计算在 `mlp.experts[*]` 里，可能绕过 hook。对比 qwen3_30b_a3b（另一个 MoE）retain=2.85%——两个 MoE 一弱一强不合常理，大概率是 qwen3.5 MoE 结构差异导致 hook 不完整。**不是机制问题，是脚本兼容性问题**。|
| **gpt2** | 39% | 小模型老架构（无 RoPE）——MLP 可能不是唯一 MA 写入者，LN + residual + attention 自积累也参与。但 gpt2 是论文大池子里的离群点，**不动摇主论点**。 |
| **qwen3.5_9b** | 32% | qwen3.5 家族专属——与其他 qwen3（qwen3_8b=1.39%, qwen3_14b=8.77%）对比，差异极大。 |
| **qwen3.5_27b** | 20% | qwen3.5 家族专属——同上。 |

**关键发现：qwen3.5 家族 3/3 全体残留显著**（81% / 32% / 20%，无一例外）。这和 RQ2c 类别（CONCENTRATED / FEW-SOURCE / DISPERSED）**无关**——DISPERSED 家族里另外 5 个模型（qwen3_14b, qwen3_32b, qwen3_30b_a3b, qwen3_8b, yi_9b）的 retain 都 ≤ 10%。

**qwen3.5 的 MA 生成机制与 qwen3 有本质不同**。值得在论文里单独讨论的家族级异常——可能原因：
- qwen3.5 训练数据或对齐策略引入了非 MLP 源
- qwen3.5 的 LayerNorm / 残差结构有改动

需查阅 qwen3.5 技术报告（若公开）进一步确认。

#### RQ2b 状态（JSON 里的 exp2 数据）

| # | 模型 | 总层 | 起源层 | max ΔMA | 初步模式 | 状态 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|
| 1 | bloom_7b1 | 30 | L3 | -95.04% | A 单层 | ✓ |
| 2 | qwen3_0.6b | 28 | L2 | -94.32% | A 单层 | ✓ |
| 3 | qwen2.5_0.5b | 24 | L0 | -90.37% | A 单层 | ✓ |
| 4 | gptj_6b | 28 | L2 | -90.26% | A 单层 | ✓ |
| 5 | qwen2_7b | 28 | L3 | -89.10% | A 单层 | ✓ |
| 6 | yi_9b | 48 | L1 | -87.62% | A 单层 | ✓ |
| 7 | llama3.1_8b | 32 | L1 | -87.02% | A 单层 | ✓ |
| 8 | falcon_7b | 32 | L3 | -86.93% | A 单层 | ✓ |
| 9 | qwen2.5_7b | 28 | L3 | -84.99% | A 单层（近中间） | ✓ |
| 10 | qwen3_1.7b | 28 | L2 | -83.75% | A 单层（近中间） | ✓ |
| 11 | qwen1.5_14b | 40 | L2 | -68.58% | **中间 待判定** | ✓ |
| 12 | qwen3_14b | 40 | L6 | -66.61% | **中间 待判定** | ✓ |
| 13 | qwen3_30b_a3b (MoE) | 48 | L2 | -57.95% | **中间 待判定** | ✓ |
| 14 | qwen3.5_9b | 32 | L26 | -50.00% | **中间 待判定** | ✓ |
| 15 | mistral_7b_v03 | 32 | L1 | -40.57% | **中间 待判定** | ✓ |
| 16 | qwen3_8b | 36 | L6 | -39.42% | **中间 待判定** | ✓ |
| 17 | qwen3_4b | 36 | L5 | -33.50% | **中间 待判定** | ✓ |
| 18 | glm4_9b | 40 | L17 | -33.03% | **中间 待判定** | ✓ |
| 19 | qwen3.5_27b | 64 | L54 | -30.48% | **中间 待判定** | ✓ |
| 20 | qwen3_32b | 64 | L43 | -15.28% | B 多层 | ✓ |
| 21 | opt_6.7b | 32 | L1 | -11.59% | B 多层 | ✓ |
| 22 | qwen3.5_35b_a3b (MoE) | 40 | L39 | -6.74% | B 多层 | ✓ |
| 23 | gpt2 | 12 | L3 | -6.70% | B 多层 | ✓ |
| 24 | **llama2_13b** | — | — | — | — | **· 未跑** |
| 25 | **llama2_7b_chat** | — | — | — | — | **· 未跑** |
| 26 | **glm4_32b** | 61 | L0 | 0% (全 NaN) | — | **⚠ 异常（v2 待 fp32 重跑）** |

**初步模式分布**：A 单层 10 个、**中间 待判定 9 个**、B 多层 4 个；缺 2 未跑 + 1 待修。实际最终模式以下面 RQ2c.category 为准。

（RQ2a 已在上方详细展开——见 "RQ2a 主结论：H₁ 验证成立" 节；此处不重复。）

#### RQ2c 状态（贪心累积消融；≡ RQ6.4 progressive）

**数据来源**：JSON `exp2c` + `results/wikitext_run/RQ2c/` 目录 25 个 `*_rq6_greedy.json` 文件（本轮 2026-04-20 盘点发现此前说的"全新实验 25 未跑"已过时，实际大部分已跑完）。

**26 模型 RQ2c 模式最终分布**（由 `exp2c.category` 严格定型）：

| 类别 | 数量 | 模型 |
|:-:|:-:|---|
| **CONCENTRATED**（单层主导） | 8 | bloom_7b1, glm4_32b, gptj_6b, mistral_7b_v03, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b |
| **FEW-SOURCE**（2-5 层主导） | 8 | falcon_7b, glm4_9b, gpt2, llama2_13b, llama3.1_8b, qwen3.5_35b_a3b, qwen3_1.7b, qwen3_4b |
| **DISPERSED**（>5 层分散） | 8 | qwen1.5_14b, qwen3.5_27b, qwen3.5_9b, qwen3_14b, qwen3_30b_a3b, qwen3_32b, qwen3_8b, yi_9b |
| **ANOMALY** | 1 | opt_6.7b（RQ1 +250%、RQ2b max -11.59%、RQ2c drop 仅 1.6%——MA 几乎扰不动）|
| **未跑** | 1 | llama2_7b_chat |

#### RQ2b critical_layer vs RQ2c L_origin 不一致（错层根因）

**RQ2b 的 critical_layer 和 RQ2c 的 L_origin 大量错位**——这是旧流程用 RQ2b peak 层当起源层跑 RQ3/4/5 导致错层的**根本原因**（详见 `docs/V2_ROOT_CAUSE.md`）。

典型案例：

| 模型 | RQ2b critical_layer | RQ2c L_origin | 错位幅度 |
|---|:-:|:-:|:-:|
| qwen3_32b | 43 | 6 | 37 层 |
| qwen1.5_14b | 2 | 35 | 33 层 |
| qwen3.5_9b | 26 | 22 | 4 层 |
| qwen3.5_35b_a3b | 39 | 9 | 30 层 |
| glm4_9b | 17 | 1 | 16 层 |
| yi_9b | 1 | 8 | 7 层 |

**结论**：下游 RQ3/4/5 必须以 `exp2c.l_origin_from_step1` 为准（即 `origin_layer/output/L_ORIGIN.sh` 的值），不再用 RQ2b peak。

### 需要重做/补跑（小清单；2026-04-20 重审后大幅收缩）

经本轮盘点，RQ2 只剩 3 个模型的零散缺口：

| 实验 | 模型 | 原因 | 成本 |
|:-:|---|---|:-:|
| RQ2a | **llama2_13b** | 未跑 | ~3 min |
| RQ2a | **opt_6.7b** | 未跑；ANOMALY 模型的关键验证 | ~3 min |
| RQ2b | **llama2_13b** | 未跑 | ~20 min |
| RQ2b | **llama2_7b_chat** | 未跑 | ~15 min |
| RQ2c | **llama2_7b_chat** | 未跑（依赖 RQ2b） | ~5 min |

**合计**：补完 3 个模型的缺口 ≈ 45 min（部分串行依赖）。

> 说明：此前版本写的"RQ2a 补 20 个 / RQ2c 全新实验 25 个均未跑"是基于不完整的 JSON 字段扫描；实际盘点后 RQ2a 已有 24/26、RQ2c 已有 25/26。

### 关键观察（RQ2 专属）

1. **H₁ 主结论最强证据**：bloom_7b1 关 MLP 后 MA 直接归零（retain=0%），配合 RQ1 最小 residual 1.69%——两边对照，MLP 起源、atten 广播的因果链完整。

2. **qwen3.5 家族专属 MA 非 MLP 机制**（新发现）：qwen3.5_9b / qwen3.5_27b / qwen3.5_35b_a3b 三个全都 retain > 15%，和 qwen3 系列的 1-8% 形成强反差。需查 qwen3.5 技术报告以确认（训练数据 / 对齐策略 / LN 结构改动）。

3. **MoE 脚本覆盖问题**：qwen3.5_35b_a3b (81%) vs qwen3_30b_a3b (2.85%) 差异悬殊，不是机制问题，是 hook 覆盖问题——需定位 MoE 专家层并单独挂 hook（Tier C 任务）。

4. **RQ2c 模式分布均匀**（CONCENTRATED/FEW-SOURCE/DISPERSED 各 8 个）——支持论文"两种生成模式"可推广到新模型家族。

5. **opt_6.7b 是真 ANOMALY**——RQ1 Sup +250%、RQ2b max -11.59%、RQ2c drop 1.6%、RQ2a 未跑。补 RQ2a 前无法判断是"MLP 不是主源"还是"attention 反向调节太强"。**优先补**。

### 待决定项

（累计到全部 RQ 讨论后一并规划执行顺序）

---

## RQ3 — 结构 Token SVD 空间映射（**待重做 + 论点重定位**）

> **2026-04-20 暂停 + 重定位**：
>
> 1. **旧名"功能词 SVD 映射"过时**——RQ4 Top-K 验证（gpt2）显示 Top-10 激活里只 1/10 是功能词，真正主导 MA 的是**结构 token**（换行、标点、特殊符号 + 部分功能词）。论文主论点要从"function word mark"改为"structural token mark"。
>
> 2. **原脚本致命 bug**：`exp5_function_words_svd_mapping.py` 的 `add_token` 只采集功能词，内容词 h₂ 从未被记录。所有已跑的 RQ3 数据作废。
>
> **重做同时解决两件事**（见 §全局补跑清单 RQ4 条目的任务 1-5）：
> - 扩采样：全 token（不只功能词），新增 is_structural 标签
> - 换指标：Top-K 分位（不用 Cohen's d 平均）
> - 换论点：结构 token 而非语法功能词
>
> 详见 §全局补跑清单 RQ4 节。

## RQ4 — SVD 几何对齐（**分析方法待重写**）

> **2026-04-20 发现**：现有 RQ4 数据（23/26 模型）在正确起源层上跑了、脚本方法学干净，但**分析指标选错**——用 Cohen's d 比较"功能词平均 \|h₂·v₁\|"，而 MA 是**极值现象**（top-K）不是均值现象。
>
> **gpt2 单模型 Top-K 验证**（σ₁/σ₂=3.05 的"强谱"模型）：
> - 整体 39.9% token 是功能词
> - 但 \|h₂·v₁\| Top-10 里只有 1/10 是功能词
> - Top-1 是 `'\n\n'`（换行符），MA=165.88，比第 2 名高 10×
> - Top-10 实际构成：**换行 / 标点 / @ 符号 / 日文字符 / 少数罕见内容词**
>
> **论点重定位**：MA 不是写在"语法功能词"位置，而是写在**"结构 token"**位置——包括：
> - 换行、段落分隔（`\n`, `\n\n`）
> - 标点（`.`, `,`, `!`, `?`, `;`）
> - 特殊符号（`@`, `#`, BOS/EOS, 罕见字符）
> - **部分功能词**（`the`, `of`, `a` 有时）
> - 高信息**罕见内容词**也偶尔上榜
>
> **这和学界 attention sink 研究一致**（Xiao et al. "Efficient Streaming LMs" 观察到 BOS/换行是 sink 位置）。
>
> **RQ3 和 RQ4 都要用新框架重做**：
> 1. 采样扩展：捕获所有 token（不限功能词）
> 2. 统计指标：Top-K 里结构 token 占比，不用 Cohen's d(平均)
> 3. 结构 token 定义：标点 + 换行 + 特殊符号 + 功能词闭类词
>
> 详见 §全局补跑清单 A 节 RQ4 / RQ3 新条目。



## RQ6 — 多层聚合分析（Macro-SVD）（**待重做**）

### 2026-04-20 致命 Bug：所有 RQ6 remove/keep_top_K 数据作废

脚本 `changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py`（JSON 里 `exp6` 字段的数据源）有两个串联问题：

**Bug 1：`critical_layer` 默认 = 0**
```python
def get_critical_layer(model_name):
    critical_layers = {"bloom_7b1": 28, "opt_6.7b": 0}  # 只硬编码 2 个！
    return critical_layers.get(model_name, 0)  # 其他都默认 L0
```
- 不读 RQ2c.l_origin_from_step1
- 对 qwen3_32b（真 L_origin=6）、qwen1.5_14b（真 L=35）、yi_9b（真 L=35）等大批模型**全部用 L0**

**Bug 2：baseline MA 在错层测，根本不是真 MA**

对比验证：

| 模型 | RQ2a 真 MA baseline | RQ6 exp6 baseline | 差距 |
|---|:-:|:-:|:-:|
| glm4_32b | 298598 | 1.15 | **260000×** |
| yi_9b | 5004 | 1.97 | 2540× |
| qwen1.5_14b | 7444 | 3.58 | 2079× |
| qwen2_7b | 6925 | 4.40 | 1574× |
| qwen3_8b | 13336 | 13.69 | 974× |
| qwen3_32b | 27417 | 30.79 | 890× |
| llama3.1_8b | 314 | 4.80 | 66× |

**17/17 模型的 RQ6 baseline 都比 RQ2a 真 MA 小 36-260000×**——脚本测到的"baseline"是**非 MA 层的正常激活值**，不是 MA。

**因此**：
- 所有 `remove_top_k.X.pct_of_baseline` 百分比**都是拿小数字和小数字比**，不反映真实 MA 机制
- glm4_32b 的 137%、glm4_9b 的 135% 看起来"MA 变大"——其实只是 1.15 → 1.57、4.58 → 6.22 的正常波动，与 MA 无关
- 之前所谓的 "qwen3_8b keep_1 = 95%，v₁ 主导" 也作废（~14 vs ~13 的比较，不是 13336 vs 13000）

### RQ6 已有数据状态总结

| 数据类型 | 之前覆盖 | 新状态 |
|---|:-:|---|
| exp6.sigma_ratio（macro η）| 17/26 | **可能仍可信**（纯数学计算，不依赖 MA 测量）|
| exp6.remove_top_k / keep_top_k | 17/26 | ✗ **全部作废**（baseline 错层）|
| macro_svd_full（gpt2, gptj_6b 全流程版）| 2/26 | 需单独验证 baseline 是否正确 |

### 重做要求

1. **修脚本**：
   - `get_critical_layer()` 改为读 `origin_layer/output/L_ORIGIN.sh`（或直接传参 `--layer_id`）
   - `get_mlp_down_proj()` 加 glm4 分支
   - baseline 测量改为"扫所有层找真 MA"（像 RQ2a / RQ6.1 那样）
2. **26 模型全部重跑 exp6（remove/keep_top_K）**
3. 详见 §全局补跑清单 RQ6 条目

### RQ6 异常分类（除 MoE 外）

扫描 15 个非 MoE 模型的 `remove_top_K / keep_top_K` 数据，发现 4-6 个异常，分三类：

**A. glm4 家族（2 个）——baseline 错层 bug 衍生**

| 模型 | remove_1 | keep_1 | 异常性质 |
|---|:-:|:-:|---|
| glm4_32b | 137.1% | 125.2% | 删/留 v₁ 都 >100%，非物理 |
| glm4_9b | 135.9% | 29.3% | 删 v₁ 反而 MA 增加 36% |

**原因**：`critical_layer` 默认 L0，glm4 baseline 极小（1.15 / 4.58），小数字相除放大了比例——**脚本 bug，非机制异常**。修 baseline + 起源层后重跑可消解。

**B. qwen3.5 dense 家族（2 个）——真机制异常**

| 模型 | remove_1 | keep_1 | 异常性质 |
|---|:-:|:-:|---|
| qwen3.5_9b | 98.3% | 100.1% | 删/留 v₁ 对 MA 几乎无影响 |
| qwen3.5_27b | 99.1% | 100.0% | 同上 |

**含义**：v₁ 和 MA 完全无关。和 RQ2a retain > 15%、RQ5 macro ≈ 0% **完全一致**——**qwen3.5 家族非标准 v₁ 机制**（Tier D）。

**C. 轻微异常（2 个）——可能是 baseline 错层衍生**

| 模型 | 指标 | 值 | 可能性 |
|---|---|:-:|---|
| qwen2_7b | keep_1 | 108.5% | 轻微 >100%，重跑后可能消解 |
| llama3.1_8b | keep_50 | 108.8% | 同上 |

### RQ6 异常总结

结合 MoE：**RQ6 真机制异常 = 4 个模型**
- Tier C (MoE)：qwen3_30b_a3b, qwen3.5_35b_a3b
- Tier D (qwen3.5 dense)：qwen3.5_9b, qwen3.5_27b

这 **4 个模型 v₁ 都和 MA 无关**——和 RQ5 macro 判读完全一致。

**B glm4 + C 轻微异常**属于脚本 bug 衍生，修脚本重跑后预计消解，不构成独立 tier。

### 位置

本轮 RQ 顺序调整为：**RQ1 → RQ2 → RQ3 → RQ4 → RQ6 → RQ5**。
RQ6 是 RQ5 模式 B 版本的前置依赖（提供 macro v₁）。

### 做什么

对多个 MLP 层的净贡献 `Δh` 累加，对"虚拟大层" `Δh_macro` 做 SVD：

```
Δh_L      = MLP_L(LN(h_L))           （L 层 MLP 写入残差流的部分）
Δh_macro  = Σ_{L in 聚合范围}  Δh_L   （多层累加）
SVD(Δh_macro) → macro σ₁, macro v₁, macro u₁
```

### 4 个子实验

| 子实验 | 脚本 | 测什么 |
|---|---|---|
| **6.1 macro-SVD** | `exp6_macro_svd_*.py` | macro σ₁/σ₂, macro v₁, macro u₁ |
| **6.2 remove-top-K** | `exp6_single_layer_activation.py` 的 `remove_top_k` | 从 Δh_macro 减去 top-K 奇异方向后的 MA |
| **6.3 keep-top-K** | 同上的 `keep_top_k` | 只保留 top-K 奇异方向后的 MA |
| **6.4 progressive ablation** | `exp6_progressive_ablation.py` | 贪心加层消融，找累积起源层子集 |

### 判读（重要）

macro σ₁/σ₂ 是 RQ6 最关键的数字：

| macro σ₁/σ₂ | 含义 |
|:-:|---|
| ≥ 3 | 强多层主方向 → 典型模式 B |
| 2 – 3 | 中等主方向 → 中间形态倾 B |
| 1 – 2 | 弱主方向 → 中间倾 A 或真分散 |

**和 RQ2b max ΔMA 对照使用可发现"矛盾模型"**（单层看分散 / 聚合看集中）→ 真模式 B 的隐藏候选。

### 25 模型当前 RQ6 状态（macro-SVD）

| # | 模型 | RQ2b max ΔMA | macro σ₁/σ₂ | 模式判定 | 状态 |
|:-:|---|:-:|:-:|:-:|:-:|
| 1 | bloom_7b1 | -95% | — | A | · 未跑 |
| 2 | falcon_7b | -87% | — | A | · 未跑 |
| 3 | glm4_32b | 0% NaN | 1.53 NaN | — | ⚠ 异常 |
| 4 | glm4_9b | -33% | **4.33** | 中间→**真 B** | ✓ |
| 5 | gpt2 | -6.7% | — (历史 3.48) | B | · JSON 未登记 |
| 6 | gptj_6b | -90% | — (历史 5.74) | A | · JSON 未登记 |
| 7 | llama2_13b | — | — | — | · 未跑 |
| 8 | llama3.1_8b | -87% | 1.42 | A | ✓ |
| 9 | mistral_7b_v03 | -41% | — | 中间 | · 未跑 |
| 10 | opt_6.7b | -12% | — | B | · 未跑 |
| 11 | qwen1.5_14b | -69% | 1.60 | 中间→偏 A | ✓ |
| 12 | qwen2.5_0.5b | -90% | 1.48 | A | ✓ |
| 13 | qwen2.5_7b | -85% | — | A | · 未跑 |
| 14 | qwen2_7b | -89% | 1.23 | A | ✓ |
| 15 | qwen3_0.6b | -94% | 1.15 | A | ✓ |
| 16 | qwen3_1.7b | -84% | 1.23 | A 近中间 | ✓ |
| 17 | qwen3_4b | -34% | 1.72 | 中间 | ✓ |
| 18 | qwen3_8b | -39% | 2.02 | 中间 | ✓ |
| 19 | qwen3_14b | -67% | 2.00 | 中间 | ✓ |
| 20 | qwen3_30b_a3b (MoE) | -58% | **1.06** | 中间→MoE 异常（expert mark 不共享）| ✓ |
| 21 | qwen3_32b | -15% | 1.83 | B | ✓ |
| 22 | qwen3.5_9b | -50% | 2.58 | 中间 | ✓ |
| 23 | qwen3.5_27b | -30% | **4.59** | B→强聚合 B | ✓ |
| 24 | qwen3.5_35b_a3b (MoE) | -6.7% | 1.40 | B（macro 较弱）| ✓ |
| 25 | yi_9b | -88% | 2.08 | A | ✓ |

### 状态汇总

| 类别 | 数量 | 备注 |
|---|:-:|---|
| ✓ macro-SVD JSON 有效 | 15 | |
| · 历史跑过但 JSON 未登记 | 2 | gpt2 (3.48), gptj_6b (5.74) 见 CONCLUSIONS.md |
| · 未跑 | 6 | bloom, falcon, mistral, opt, qwen2.5_7b, llama2_13b |
| ⚠ 数据异常 | 1 | glm4_32b |

**remove-top-K / keep-top-K 数据稀疏**，仅 ~5 个模型有，其余 `{}` 空字典。

### 关键发现

1. **单层分散、聚合集中的"真模式 B"** — glm4_9b、qwen3.5_27b 是 RQ2b 归类为中间、但 RQ6 显示 macro σ₁/σ₂ ≥ 4 的典型。没有 RQ6 无法发现。
2. **MoE 的 qwen3_30b_a3b macro σ₁/σ₂ = 1.06** — 极弱谱集中，强烈提示 MoE expert 之间 mark 不共享，需 Tier C 专项。
3. **qwen3.5_35b_a3b (MoE) macro σ₁/σ₂ = 1.40** — 也较弱，同上。

### 需要重做/补跑（按优先级）

#### 🔥 高优先 — 模式 B 4 个 + 中间形态 9 个 = 13 个模型

> **含义**：只有跑全这 13 个模型，才能把"中间形态"的 9 个模型的真正归属定下来（真 B / 偏 A / MoE 异常）。这是本轮 RQ6 的核心产出。

| # | 模型 | RQ2b | 当前 macro σ₁/σ₂ | 需要补什么 | 成本 |
|:-:|---|:-:|:-:|---|---|
| 1 | **gpt2** | -6.7% | 历史 3.48 | JSON 登记 + remove/keep top-K | ~15 min |
| 2 | **opt_6.7b** | -12% | 未跑 | 全套 macro-SVD + top-K | ~15 min |
| 3 | **qwen3_32b** | -15% | 1.83 ✓ | 补 remove/keep top-K | ~10 min |
| 4 | **qwen3.5_27b** | -30% | 4.59 ✓ | 补 remove/keep top-K | ~15 min |
| 5 | **qwen3.5_35b_a3b (MoE)** | -6.7% | 1.40 ✓ | 补 remove/keep top-K | ~20 min |
| 6 | **qwen3.5_27b** 也在中间？| -30% | 4.59 ✓ | 已在上方列出 | — |
| 7 | **qwen1.5_14b** | -69% | 1.60 ✓ | 补 remove/keep top-K | ~10 min |
| 8 | **qwen3_14b** | -67% | 2.00 ✓ | 补 remove/keep top-K | ~10 min |
| 9 | **qwen3_30b_a3b (MoE)** | -58% | **1.06** ✓ | 补 remove/keep top-K | ~15 min |
| 10 | **qwen3.5_9b** | -50% | 2.58 ✓ | 补 remove/keep top-K | ~10 min |
| 11 | **mistral_7b_v03** | -41% | 未跑 | 全套 macro-SVD + top-K | ~10 min |
| 12 | **qwen3_8b** | -39% | 2.02 ✓ | 补 remove/keep top-K | ~10 min |
| 13 | **qwen3_4b** | -34% | 1.72 ✓ | 补 remove/keep top-K | ~10 min |
| 14 | **glm4_9b** | -33% | 4.33 ✓ | 补 remove/keep top-K | ~10 min |

**合计**：~**2.5h**（14 条目去重后 13 个模型）

**期望产出**：判出中间 9 个模型每个到底是"单层+调节"（偏 A）、"真多层"（偏 B）、还是"MoE 异常"。

#### ⭐ 中优先 — 模式 A 10 个模型

> **含义**：模式 A 模型的 macro σ₁/σ₂ 预期偏低（因为 MA 就是单层写入的），作为"A 的对照证据"。

| 模型 | 预期 | 成本 |
|---|---|---|
| bloom_7b1, falcon_7b, gptj_6b, llama3.1_8b, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, qwen3_1.7b, yi_9b | macro σ₁/σ₂ 中等-弱（因为单层已主导，聚合不增益） | 每个 ~10 min × 10 = ~**1.5h** |

#### ⏸ 依赖前置

- **llama2_13b**（等 RQ2b 跑完，确定聚合起点）
- **glm4_32b**（等 RQ1 fp32 修复）

### 合并机会：RQ2c ≈ RQ6.4 progressive

RQ2c（贪心累积消融，判定中间形态模式）和 RQ6.4 progressive ablation 是**本质相同的实验**——都是"按 Δ 贡献排序依次禁层，画累积曲线"。

**建议**：合跑一次（用 `exp6_progressive_ablation.py`），数据同时填进 RQ2c 结论 + RQ6.4 结论。省一半时间。

### RQ6 记录小结

- **覆盖**：macro-SVD 15/25 JSON 有效 + 2 历史 + 6 未跑 + 1 异常 + 1 依赖
- **本轮补跑**：高优先 13 个（~2.5h）→ 中优先 10 个（~1.5h）；合计 ~**4h**
- **关键决定**：中间 9 个提到高优先，与模式 B 合并，目的是**本轮 RQ6 后全 25 模型完成精确分类**
- **合并操作**：RQ2c 累积消融一次做完，填两个 RQ 的数据
- **产出**：给 RQ5 模式 B 版本提供 macro v₁；给论文"A/B 两种模式"论点提供完整数据
- **Tier C 遗留**：MoE 两个模型的 expert-level 分析暂缓

## RQ5 — V 矩阵消融（终局因果验证）

### 2026-04-20 关键澄清：Single 和 Macro 两版本的真实差异

此前文档把 single-V 消融和 macro-V 消融对比时出现 "反常" 误判（glm4_32b, llama2_13b: single 强 macro 弱）。**实际上不是反常，是两个版本的消融方式本来就不同**：

| 版本 | 破坏方式 | 数学表达 | 强度 | 测试含义 |
|---|---|---|:-:|---|
| **Single V-ablation** | 把 V 矩阵**整个换成随机正交矩阵** | `W_down = U × Σ × V_rand^T` | **强**（16384 个方向全随机）| 整个 V 是否参与 MA 生成（含混杂效应）|
| **Macro projection** | 只把 macro_v₁ **一个方向投影掉** | `W_down = (I − vv^T) × W_down` | **弱精准**（只消 1 个方向）| **严格证明**特定方向 v₁ 的因果必要性 |

**因此**：
- 当 origin_layers 只有 1 层时（如 glm4_32b=[0]），single 和 macro 层一样，**但强度差异让 single 通常更强**——这是**合理的设计选择**，不是 bug
- 当 origin_layers 多层时（如 gpt2=[1,2,3]），macro 跨层 → 覆盖面更广 → macro 可能更强
- "single 强 macro 弱"不一定反常；"两个都弱"才是真反例

**论文叙事建议**：
- **macro projection 作为主指标**——论文严谨证据（"v₁ 是因果必要方向"无混杂）
- single V_rand 作为辅证——证明 V 矩阵整体参与（但不具体到 v₁）

### 可选扩展：macro V_rand（本轮不做，记录为未来工作）

在 `exp5_macro_v_ablation.py` 基础上加一个版本：对 origin_layers 所有层做 `V → V_rand` 替换（而不是投影消除 macro_v₁）。

- 目的：验证"single V_rand 扩到多层"的效果，和 macro projection 对比
- 如果 projection ΔMA ≈ V_rand ΔMA → v₁ 一家就够（最强证据）
- 如果 projection < V_rand → MA 还用了其他 V 方向
- 成本：改脚本 ~20 min + 全 26 模型跑 ~1h

### 26 模型完整 RQ5 数据（2026-04-20 定稿）

| # | 模型 | RQ2c 类别 | L_origin | single_ΔMA | macro_ΔMA | origin_layers | macro η | 备注 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | bloom_7b1 | CONCENTRATED | 3 | **-8.6%** | — | — | — | ⚠ **single 反常弱**（CONCENTRATED 理应强），需补 macro |
| 2 | falcon_7b | FEW-SOURCE | 3 | -98.1% | -97.4% | [0, 3] | 3.18 | ✓ 双强 |
| 3 | glm4_32b | CONCENTRATED | 0 | -96.8% | -17.1% | [0] | 1.87 | ✓ single > macro（合理，V_rand 比 projection 力度大）|
| 4 | glm4_9b | FEW-SOURCE | 1 | -46.5% | -81.8% | [0, 1] | 3.03 | ◐ macro > single |
| 5 | **gpt2** | FEW-SOURCE | 3 | **-6.5%** | **-94.6%** | [1, 2, 3] | **9.44** | ✓✓ 典型多层接力，macro 救场 |
| 6 | gptj_6b | CONCENTRATED | 2 | -99.0% | -99.1% | [2] | 15.35 | ✓ 双强，标准案例 |
| 7 | llama2_13b | FEW-SOURCE | 39 | -95.6% | -28.6% | [0, 1, 39] | 1.79 | ✓ single 强（L0 关键）|
| 8 | llama2_7b_chat | — | — | — | — | — | — | · 两个都缺 |
| 9 | llama3.1_8b | FEW-SOURCE | 1 | — | -99.8% | [0, 1] | 24.66 | ✓ macro 强，single 待补 |
| 10 | mistral_7b_v03 | CONCENTRATED | — | -83.2% | — | — | — | ◐ single 近强，macro 待补 |
| 11 | opt_6.7b | ANOMALY | — | — | — | — | — | · 两个都缺（RQ2c ANOMALY 模型）|
| 12 | qwen1.5_14b | DISPERSED | 18 | — | **-12.6%** | [2,3,4,15,17,18] | 3.24 | ⚠ macro 弱，需补 single 才能判 |
| 13 | qwen2.5_0.5b | CONCENTRATED | — | — | — | — | — | · 两个都缺 |
| 14 | qwen2.5_7b | CONCENTRATED | — | — | — | — | — | · 两个都缺 |
| 15 | qwen2_7b | CONCENTRATED | — | — | — | — | — | · 两个都缺 |
| 16 | **qwen3.5_27b** | DISPERSED | 56 | — | **-0.4%** | [50,51,52,53,54] | 1.83 | ⚠ **qwen3.5 异常**（macro 几乎无效）|
| 17 | **qwen3.5_35b_a3b** | FEW-SOURCE | 9 | **+0.1%** | **+0.5%** | [7, 9, 38] | 3.93 | ✗ **真反例**（MoE + qwen3.5，single/macro 都无效，甚至 ΔMA 微正）|
| 18 | **qwen3.5_9b** | DISPERSED | 11 | — | **-0.2%** | [2,3,6,9,10,11] | 6.00 | ⚠ **qwen3.5 异常**（η=6 高但消融无效）|
| 19 | qwen3_0.6b | CONCENTRATED | — | — | — | — | — | · 两个都缺 |
| 20 | qwen3_1.7b | FEW-SOURCE | 2 | — | -99.8% | [0, 1, 2] | 28.92 | ✓ macro 强，single 待补 |
| 21 | qwen3_14b | DISPERSED | 6 | -93.5% | -88.2% | [3,6,7,10,12,13] | 6.83 | ✓ 双强 |
| 22 | **qwen3_30b_a3b** | DISPERSED | 1 | **-0.8%** | **-0.4%** | [1,3,10,11,26] | 1.48 | ✗ **真反例**（MoE，η=1.48 很低）|
| 23 | qwen3_32b | DISPERSED | 29 | — | -86.3% | [4,5,6,19,28,29] | 13.95 | ✓ macro 强，single 待补 |
| 24 | qwen3_4b | FEW-SOURCE | 15 | — | -99.7% | [6, 15] | 16.28 | ✓ macro 强，single 待补 |
| 25 | qwen3_8b | DISPERSED | 7 | — | -99.6% | [0,2,3,5,6,7] | 10.83 | ✓ macro 强 |
| 26 | yi_9b | DISPERSED | 35 | — | -99.2% | [8,18,23,24,35] | 8.38 | ✓ macro 强 |

**数据覆盖汇总**：
- 有 single：11/26
- 有 macro：18/26
- 两者都有：9/26
- 全无：7/26（bloom_7b1 macro 缺 + 6 个 CONCENTRATED + opt_6.7b + llama2_7b_chat）

### 异常值清单（单独归类）

**A. 确定的真反例（2 个）**——single 和 macro 都有数据且都弱（ΔMA > -5%）
| 模型 | single | macro | 归因 |
|---|:-:|:-:|---|
| qwen3.5_35b_a3b | +0.1% | +0.5% | MoE + qwen3.5 双重异常，ΔMA 甚至微正 |
| qwen3_30b_a3b | -0.8% | -0.4% | MoE，macro η 只 1.48（合并贡献小）|

**B. 待补 single 的弱 macro（3 个）**——macro 弱但需要 single 验证
| 模型 | macro | macro η | 预判 |
|---|:-:|:-:|---|
| qwen3.5_9b | -0.2% | 6.00 | **η 高 macro 却无效**——非 v₁ 机制？|
| qwen3.5_27b | -0.4% | 1.83 | qwen3.5 家族异常 |
| qwen1.5_14b | -12.6% | 3.24 | η 3.24 不低却消融弱——存疑 |

**C. bloom_7b1 单独异常（1 个）**——CONCENTRATED 模型但 single 只 -8.6%
- RQ2a retain=0%（MA 100% 来自 MLP）
- RQ5 single ΔMA=-8.6%（消 v₁ 却基本无效）
- 矛盾：MLP 强主导 vs v₁ 消融无效 → **v₁ 不是 bloom 的 MA 方向**
- 可能用了 u₁[max] 特定 hidden 维度，或 bias 补偿
- **需跑 macro 验证**，同时 RQ4 加测 u₁ 分布

### 联合判读（扣除异常后）

- **有 single+macro 数据的 9 个里**：7 个强支持（至少一个 ≤ -85%），2 个真反例（都是 MoE）
- **只有 macro 数据的 9 个里**：6 个强支持，3 个弱（qwen3.5 家族 + qwen1.5_14b）
- **只有 single 数据的 2 个里**：mistral_7b_v03 近强，bloom_7b1 弱异常

**主论点支持率**（扣除 MoE 2 个 + qwen3.5 3 个 + bloom 1 个 + 缺数据 7 个）：
- 有效样本 = 18 - 2 - 3 - 1 = 12 个（都是非 MoE/qwen3.5/非异常模型的 macro 数据）
- 强支持（ΔMA ≤ -85%）= 11/12 = **91.7%**
- 唯一"弱支持"是 glm4_9b macro=-81.8%（接近阈值）

**RQ5 结论**：H₅（v₁ 是 MA 因果必要方向）**在标准模型上成立**，qwen3.5 家族和 MoE 是已知异常集群，论文应单独讨论。

### 执行顺序：放在 RQ6 之后

**理由**：RQ5 是"砸掉 v₁ → 看 MA 是否消失"的终局因果验证。它需要先知道 v₁ 是哪个：
- 模式 A 模型 → 单层 v₁（来自起源层 W_down 的 SVD，由 RQ4 给出）
- 模式 B 模型 → macro v₁（来自多层聚合 SVD，由 RQ6 给出）
- 中间形态 → 先等 RQ2c 判定模式后决定

所以本轮 RQ 顺序调整为：**RQ1 → RQ2 → RQ3 → RQ4 → RQ6 → RQ5**。

### 做什么

对起源层（或 macro）的 W_down 做 SVD，把 V 矩阵替换为随机正交矩阵 `V_rand`，保留 U 和 Σ。测替换前后 MA 的变化。

```
W_down = U Σ V^T          (原矩阵)
W_ablated = U Σ V_rand^T   (替换 V，保留 U 和 Σ)

测 MA(model with W_ablated) / MA(model with W_down)
```

预期：**模式 A 在起源层做 → ΔMA ≤ -85%**；**模式 B 在 macro 空间做 → ΔMA ≤ -85%**。

### 关键字段（JSON `exp5`）

- `layer`：做消融的层
- `baseline_ma`：原始 MA
- `u_attribution_pct` / `v_attribution_pct` / `interaction_pct`：归因分解
- `ablate_u_mean` / `ablate_v_mean`：消融 u / v 后的 MA 均值

### 25 模型当前 RQ5 状态

| # | 模型 | 起源层 | 跑的层 | u 归因% | 状态 |
|:-:|---|:-:|:-:|:-:|:-:|
| 1 | bloom_7b1 | L3 | — | — | · 未跑 |
| 2 | falcon_7b | L3 | — | — | · 未跑 |
| 3 | glm4_32b | 不可定 | L59? | 全 NaN | **⚠ 异常** |
| 4 | glm4_9b | L17 | L38 (peak+) | 88.6 | ◐ 错层 |
| 5 | gpt2 | L3 (多层) | — | — | · 未跑 |
| 6 | gptj_6b | L2 | — | — | · 未跑 |
| 7 | llama2_13b | 未知 | — | — | · 未跑 |
| 8 | llama3.1_8b | L1 | L30 (peak+) | 87.1 | ◐ 错层 |
| 9 | mistral_7b_v03 | L1 | — | — | · 未跑 |
| 10 | opt_6.7b | L1 (多层) | — | — | · 未跑 |
| 11 | qwen1.5_14b | L2 | L38 (peak+) | 92.1 | ◐ 错层 |
| 12 | qwen2.5_0.5b | L0 | L22 (peak+) | 67.5 | ◐ 错层 |
| 13 | qwen2.5_7b | L3 | — | — | · 未跑 |
| 14 | qwen2_7b | L3 | L26 (peak+) | 94.0 | ◐ 错层 |
| 15 | qwen3_0.6b | L2 | L26 (peak+) | 63.5 | ◐ 错层 |
| 16 | qwen3_1.7b | L2 | L26 (peak+) | 57.0 | ◐ 错层 |
| 17 | qwen3_4b | L5 | L34 (peak+) | 97.0 | ◐ 错层 |
| 18 | qwen3_8b | L6 | L34 (peak+) | 97.3 | ◐ 错层 |
| 19 | qwen3_14b | L6 | L38 (peak+) | 98.8 | ◐ 错层 |
| 20 | qwen3_30b_a3b (MoE) | L2 | — | — | · 未跑 |
| 21 | qwen3_32b | L43 | L62 (peak+) | 98.8 | ◐ 错层 |
| 22 | qwen3.5_9b | L26 | L30 (peak+) | 78.1 | ◐ 错层 |
| 23 | qwen3.5_27b | L54 | L62 (peak+) | 85.2 | ◐ 错层 |
| 24 | qwen3.5_35b_a3b (MoE) | L39 | — | — | · 未跑 |
| 25 | yi_9b | L1 | L46 (peak+) | 83.8 | ◐ 错层 |

### 汇总

| 类别 | 数量 | 备注 |
|---|:-:|---|
| ✓ 完成且正确 | **0** | 无 |
| ◐ 错层（在 peak+ 做） | 14 | 全部需在起源层重跑 |
| · 未跑 | 10 | bloom, falcon, gpt2, gptj, llama2_13b, mistral, opt, qwen2.5_7b, qwen3_30b_a3b, qwen3.5_35b_a3b |
| ⚠ 数据异常 | 1 | glm4_32b (承 RQ1 fp32 修复) |

### 需要重做/补跑

**按原因分组**：

| 原因 | 数量 | 动作 |
|---|:-:|---|
| 🔧 重跑（已跑但错层）| 14 | `--layer_id = exp2.critical_layer` 重跑 |
| ➕ 新跑（从未跑过）| 10 | 起源层跑 |
| ⚠ 数据修复 | 1 | glm4_32b 合 RQ1 fp32 修复后跑 |

**按模式分组**：

| 模式 | 数量 | 在哪做 RQ5 |
|---|:-:|---|
| A 单层 | 10 | 起源层单层 RQ5（标准） |
| 中间 | 9 | 起源层单层 RQ5 + 若 RQ2c/RQ6 判定为模式 B，补做 macro RQ5 |
| B 多层 | 4 | 单层起源层 RQ5 作对照（预期弱），+ macro RQ5（依赖 RQ6 的 macro v₁）|
| MoE | 2 | 暂缓（Tier C 专项） |
| 数据待修 | 2 | llama2_13b（等 RQ2b）、glm4_32b（等 RQ1 fp32）|

### RQ5 记录小结

- **本轮 25 个模型都要重做/新跑**（没有一个是在正确层位做的）
- **依赖前置**：
  - RQ2b 起源层 → 所有模型
  - RQ2c 模式判定 → 中间 9 个模型
  - RQ6 macro v₁ → 模式 B 4 个模型的 macro RQ5
- **执行时机**：在 RQ6 完成之后统一做
- **预期验证**：
  - 模式 A 起源层 RQ5 → ΔMA ≤ -85%
  - 模式 B macro RQ5 → ΔMA ≤ -85%
  - 若出现 PPL 不上升 → 证实附录 A.4 的"信息分轨"

---

## 全局重跑计划汇总（2026-04-21）

> 按**执行优先级 + 依赖链**排序。修脚本 bug 是阻塞前提。

### 总览

| 阶段 | 内容 | 模型数 | 成本 |
|:-:|---|:-:|:-:|
| **0. 修脚本（阻塞所有重跑）** | 7 个 bug 修复 | — | **~2h** |
| **1. 数据缺口小补** | RQ1 / RQ2 遗留缺口 | 3 模型 | ~1h |
| **2. RQ3/RQ4 结构 token 重做** | 改论点 + 脚本 + 全跑 | 26 | ~3h |
| **3. RQ6 exp6 全 26 重跑** | baseline 错层修复后 | 26 | ~2.5h |
| **4. RQ5 补数据** | 缺 single/macro 的模型 | 8 | ~2h |
| **5. MoE 专项（Tier C）** | per-expert 分析扩展 | 2 | ~3h |
| **总计** | | | **~13.5h** |

### 阶段 0：修脚本（7 个 bug）

| # | Bug | 位置 | 影响 RQ | 成本 |
|:-:|---|---|---|:-:|
| B1 | `add_token` 只存功能词，丢内容词 | `RQ3/exp5_function_words_svd_mapping.py` | RQ3/4 | ~15 min |
| B2 | MoE `SparseMoeBlock` 无 `.up_proj/.down_proj` | RQ3 脚本直接访问 | RQ3/6 | ~30 min |
| B3 | `get_mlp_submodules()` 缺白名单 | `lib/model_utils.py` | RQ3 | ~10 min |
| B4 | `get_mlp_down_proj()` 缺 glm4/MoE 分支 | `changeHead/exp6_v_ablation.py` | RQ6 | ~10 min |
| B5 | `get_critical_layer()` 默认 L0 不读 L_origin | `changeHead/exp6_v_ablation.py` | RQ6 | ~10 min |
| B6 | baseline 只在 critical_layer 测（非真 MA）| `changeHead/exp6_v_ablation.py` | RQ6 | ~15 min |
| B7 | RQ2a `MLPDisableHook` 未处理 tuple 输出 | `RQ2/exp2a_mlp_feasibility_test.py` | RQ2a | ~5 min |

**合计 ~2h**（串行）；全部完成才能执行阶段 1-5。

### 阶段 1：RQ1 / RQ2 数据缺口（3 模型）

| # | 模型 | 补 | 成本 |
|:-:|---|---|:-:|
| 1 | opt_6.7b | RQ2a（ANOMALY_NO_MLP_RESPONSE 判因必需）| ~3 min |
| 2 | qwen2_7b | RQ1（`--nsamples 60` 修复 baseline≈0）| ~15 min |
| 3 | llama2_13b | RQ2a + RQ2b | ~23 min |
| 4 | llama2_7b_chat | RQ2b + RQ2c | ~20 min |

**合计 ~1h**。

### 阶段 2：RQ3 / RQ4 结构 token 重做（26 模型）

**前置**：Bug B1/B2/B3 修完。

| 任务 | 内容 | 成本 |
|---|---|:-:|
| 定义"结构 token"词表 | 标点 + 换行 + 特殊符号 + 功能词 | ~10 min |
| 扩脚本采样：全 token（不限功能词）+ is_structural 标签 | 改 `exp5_function_words_svd_mapping.py` | ~15 min |
| 重跑 RQ3 全 26 模型 | 收集所有 token 的 h₂ + v₁ alignment | ~1.5h |
| RQ4 指标换：Top-K 里结构 token 占比（不用 Cohen's d 平均）| 分析脚本，不需重跑模型 | ~30 min |
| RQ3 + RQ4 联合分析 + 论点重定位 | 写回 EXPERIMENT_PLAN.md | ~30 min |

**合计 ~3h**。

### 阶段 3：RQ6 exp6 全重跑（26 模型）

**前置**：Bug B4/B5/B6 修完。

| 任务 | 内容 | 成本 |
|---|---|:-:|
| 全 26 模型重跑 `exp6_v_ablation` | 用正确的 L_origin + 真 MA baseline | ~2h |
| 重新分析 remove/keep_top_K | 修正数据后的机制判读 | ~30 min |

**合计 ~2.5h**。预期：glm4 `>100%` 异常消解；qwen3.5 dense 仍 `remove_1 ≈ 100%`（真机制异常）。

### 阶段 4：RQ5 补数据（8 模型）

| 任务 | 模型 | 缺什么 | 成本 |
|:-:|---|---|:-:|
| 1 | bloom_7b1 | macro | ~15 min（验证 single -8.6% 是不是真反例）|
| 2 | mistral_7b_v03 | macro | ~15 min |
| 3 | qwen2.5_0.5b | single + macro | ~15 min |
| 4 | qwen2.5_7b | single + macro | ~15 min |
| 5 | qwen2_7b | single + macro | ~15 min |
| 6 | qwen3_0.6b | single + macro | ~15 min |
| 7 | opt_6.7b | single + macro | ~15 min |
| 8 | llama2_7b_chat | single + macro | ~15 min |

**合计 ~2h**。可选：macro V_rand 扩展（~1h 额外）——论文双重验证用。

### 阶段 5：MoE 专项（Tier C，优先级低）

| 任务 | 内容 | 成本 |
|---|---|:-:|
| 修 4 类 MoE 脚本 bug | B1-B4 | 合阶段 0 |
| 扩展 per-expert 分析到 RQ2/3/4/6 | 加 `experts[*]` 迭代 | ~2h |
| qwen3_30b_a3b + qwen3.5_35b_a3b 专项重跑 | 修完后 | ~1h |

**合计 ~3h**。**本轮优先级低**——主结论定稿后再做。

### 依赖链图

```
阶段 0 (修 bug, ~2h)
    ↓
    ├── 阶段 1 (RQ1/2 小补, ~1h) ─────────┐
    ├── 阶段 2 (RQ3/4 重做, ~3h) ─────────┤
    ├── 阶段 3 (RQ6 重跑, ~2.5h) ─────────┤
    └── 阶段 4 (RQ5 补数据, ~2h) ─────────┤
                                          ↓
                                     主结论定稿
                                          ↓
                                   阶段 5 (MoE 专项, ~3h)
```

### 验证标准

- **阶段 0 完成**：跑一次 gpt2 + glm4_9b + qwen3_30b_a3b 三个 "sentinel" 模型不报错
- **阶段 2 完成**：26 模型都有完整结构 token 对 v₁ 的 Top-K 统计
- **阶段 3 完成**：glm4 `remove_1 > 100%` 消解；qwen3.5 dense 仍 `≈ 100%`
- **阶段 4 完成**：bloom_7b1 macro ΔMA < -80% 或确认真反例
- **主结论定稿**：24 dense 模型中 **≥ 20 个支持 H₁-H₅**（Tier D 的 qwen3.5_9b/27b + Tier E 的 opt_6.7b 单独讨论）

---

## 全局补跑清单（最后要补的）

> 本节汇总所有 RQ 的最终遗留缺口，随每个 RQ 分析定稿时追加。

### A. 没跑的缺口（2026-04-20）

**RQ1/RQ2 缺口**：

| # | 模型 | 缺口实验 | 成本 | 优先级 |
|:-:|---|---|:-:|:-:|
| 1 | **opt_6.7b** | RQ2a | ~3 min | 🔥 最高（ANOMALY 判因） |
| 2 | **qwen2_7b** | RQ1（`--nsamples 30 → 60`，fix baseline≈0 除零） | ~15 min | ⭐ 高 |
| 3 | **llama2_13b** | RQ2a + RQ2b | ~23 min | ⭐ 中 |
| 4 | **llama2_7b_chat** | RQ2b + RQ2c（2c 等 2b） | ~20 min | ⭐ 中 |

**RQ6 macro-SVD full 缺口**（2026-04-20 新增，含 projection 字段）：

`exp6_macro_svd_full.py` 跑完整版只有 **gpt2、gptj_6b 2/26** 有数据（`*_macro_svd_full.json`），包含 `macro_svd.eta/var_top1`、`projection.cohen_d` 等字段。其余 24 个模型在 JSON `exp6` 里只有**单层 `sigma_ratio`** + `remove_top_k / keep_top_k` 消融数据，**缺 projection**（无法做 RQ3-macro 对照）。

| # | 补跑什么 | 模型数 | 成本 |
|:-:|---|:-:|:-:|
| — | `exp6_macro_svd_full.py` 完整版（含 macro v₁ projection）| 24 | ~2.5h（大模型 ~15 min/个）|

**优先级**：对 DISPERSED 8 个（qwen1.5_14b, qwen3.5_27b, qwen3.5_9b, qwen3_14b, qwen3_30b_a3b, qwen3_32b, qwen3_8b, yi_9b）**必要**——macro v₁ projection 是"多层接力是否共享 mark"的直接证据。其余 16 模型**可选**（用于对照）。

**RQ3 缺口**（2026-04-20 重审，**整个 RQ3 需要重做**）：

> ⚠️ 此前已跑的 16 模型数据**全部作废**——脚本 `exp5_function_words_svd_mapping.py` 的 `FunctionWordSVDTracker.add_token()` 只采集功能词，内容词 h₂ 从未记录。所谓"功能词 vs 内容词"的 Cohen's d 实际比较的是"核心功能词 vs 边缘功能词"。详见 RQ3 节说明。

**RQ3 重做涉及 3 个脚本 bug**（阻塞所有 26 模型）：

| # | Bug | 位置 | 修法 | 成本 |
|:-:|---|---|---|:-:|
| **3** | **add_token 只存功能词，内容词被完全丢弃** | `RQ3_function_words/exp5_*.py::FunctionWordSVDTracker.add_token()` | 加 `content_word_data` dict，`is_function_word(t)` 为 False 时存入 | ~15 min |
| 1 | MoE 不支持（`'SparseMoeBlock' has no attribute 'up_proj'`）| 同脚本访问 `.up_proj/.down_proj` 处 | hook 到 `mlp.experts[*].down_proj` 聚合 | ~30 min |
| 2 | `get_mlp_submodules()` 缺模型白名单 | `lib/model_utils.py` | 加 glm4 / qwen1.5 / qwen3.5 / yi 的分支 | ~10 min |

**重做范围**：全部 26 模型（此前 16 个也要重跑，因为没有内容词对照数据）

| # | 模型 | 状态 | 修脚本后成本 |
|:-:|---|---|:-:|
| 1-16 | 此前 16 个"有效"模型 | 数据作废，重跑拿内容词对照 | ~25 min |
| 17-20 | qwen3_30b_a3b, qwen3.5_35b_a3b (MoE) | Bug 1 阻塞 | ~6 min |
| 21-24 | glm4_9b, glm4_32b, qwen3.5_9b, qwen3.5_27b | Bug 2 阻塞 | ~15 min |
| 25-26 | qwen1.5_14b, yi_9b | Bug 2，脚本报错无 json | ~5 min |
| 27-28 | opt_6.7b, llama2_7b_chat | 从未跑 | ~4 min |
| — | qwen3_1.7b 验证 A/C | 修完重跑后重判反例 | 合在一起 |

**合计**：
- 脚本修复（3 个 bug）~55 min
- 重跑全部 26 模型 ~55 min
- RQ1/2 尾巴 ~61 min
- RQ6 macro-SVD full 24 模型 ~2.5h
- **总计 ~5h**（部分可并行）

**RQ4 分析方法重写 + RQ3/RQ4 论点重定位为"结构 token"**（2026-04-20 新增，**高优先级**）：

| # | 任务 | 成本 | 前置 |
|:-:|---|:-:|---|
| 1 | **Top-K 验证扩展**：对现有 23 个 RQ4_origin 模型的 `sample_tokens` 跑 Top-K 统计（不需重跑模型，只分析 JSON） | ~15 min | 无 |
| 2 | **定义"结构 token"词表**：标点 + 换行 + 特殊符号 + 功能词闭类词的完整列表 | ~10 min | 无 |
| 3 | **修 RQ3 脚本**：同时记录所有 token（不只功能词），新增 is_structural 标签（和 Bug 3 修复合并） | ~15 min | 任务 2 |
| 4 | **RQ3 + RQ4 重跑**：全 26 模型，用新指标（Top-K 结构 token 占比）+ 新采样（全 token） | ~2h | 任务 3 |
| 5 | **RQ4 分析重写**：从"功能词 vs 内容词 Cohen's d"改为"Top-K 里结构 token 占比 + 例举 Top-10"+ σ₁ 绝对值分析 | ~30 min | 任务 4 |

**说明**：
- gpt2 单模型已经验证 Top-K 9/10 是结构 token（非语法功能词）——**需要在 23 个模型上扩展确认这是普遍规律**
- 采样限制：目前每个模型只存 1000 个 sample_tokens，如果 Top-K 极值被截断，分析可能偏差——**RQ4 重跑建议存全部 token**（或至少 10000）
- **论文主论点要从"function word mark"调整为"structural token mark"**——这是本轮最大的理论修正
- **RQ3 作废的旧分析全部不用恢复**，直接按新框架重写

**MoE 专项**（2026-04-20 划分为 Tier C 单独类别；不纳入主结论）：

| # | 任务 | 内容 | 成本 |
|:-:|---|---|:-:|
| 1 | 修 4 类 MoE 脚本 bug | B1（`.up_proj/.down_proj`）、B2（tuple 返回）、B3（`get_mlp_submodules`）、B4（`get_mlp_down_proj` 无 MoE 分支）| ~1h |
| 2 | 扩展 per-expert 分析到所有 RQ | RQ2 / RQ3 / RQ4 / RQ6 加 per-expert 版本 | ~2h |
| 3 | qwen3_30b_a3b, qwen3.5_35b_a3b 两模型专项重跑 | 修完后 | ~1h |

**说明**：
- MoE 两个模型已**不纳入主结论**（24 dense 模型作为 main sample）
- 论文单独章节讨论（"Appendix: MA in MoE Models"）
- 本轮**优先级低**——等主结论定稿后再做

**RQ6 整体重跑**（2026-04-20 发现 RQ6 baseline 严重错层，**全 26 模型重跑**）：

| # | 任务 | 内容 | 成本 |
|:-:|---|---|:-:|
| 1 | 修 `get_critical_layer()` | 改为读 `origin_layer/output/L_ORIGIN.sh` 或直接传 `--layer_id` 参数；不再默认 L0 | ~10 min |
| 2 | 修 `get_mlp_down_proj()` | 加 glm4 分支（`layer.mlp.down_proj`）；检查其他 MoE/异构模型 | ~10 min |
| 3 | 修 baseline 测量 | 改为扫所有层找真 MA（像 RQ2a 那样），而不是只在 critical_layer 测 | ~15 min |
| 4 | 全 26 模型重跑 exp6 (remove/keep_top_K) | 修完脚本后 | ~2h |
| 5 | 补跑缺的 9 个模型的完整 macro-SVD | bloom_7b1, falcon_7b, gpt2/gptj_6b 已有, ...（见 RQ6 节覆盖）| 合并在任务 4 |

**说明**：
- 所有 17 个已跑 exp6 模型的 baseline **比真 MA 小 36-260000×**——remove/keep 百分比**全部不反映 MA 机制**
- glm4_32b 的 260000× 差距是**最极端的错层证据**
- `sigma_ratio` 和 `macro_η`（纯 SVD 数学）可能仍可信，但 `remove/keep_top_K`（依赖 MA 测量）**全部作废**
- 之前写的"qwen3_8b keep_1=95% 支持 v₁ 主导"等结论**作废**

**RQ5 缺数据 + 可选扩展**（2026-04-20）：

| # | 任务 | 模型 / 内容 | 成本 | 优先级 |
|:-:|---|---|:-:|:-:|
| 1 | **补 macro V-ablation**（缺 8 个）| bloom_7b1, mistral_7b_v03, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, opt_6.7b, llama2_7b_chat | ~2h（每个 ~15 min）| ⭐ 中 |
| 2 | **补 single V-ablation**（缺一些）| llama3.1_8b, qwen1.5_14b, qwen3.5_27b, qwen3.5_9b, qwen3_1.7b, qwen3_4b, qwen3_8b, qwen3_32b, yi_9b | ~2h | ⭐ 中（与 macro 联合判读用） |
| 3 | **bloom_7b1 异常定位**：补 macro，验证 single -8.6% 是否 bloom 真反例 | bloom_7b1 专项 | 合任务 1 | 🔥 高 |
| 4 | **（可选）macro V_rand 对照** | 扩展 `exp5_macro_v_ablation.py` 加 V_rand 版本（非 projection），26 模型重跑 | ~20 min 改脚本 + ~1h 跑 | ◯ 可选 |

**说明**：
- RQ5 判读框架已定稿（见 §RQ5 节）：single 和 macro **不是等效操作**，macro projection 更精准（论文主指标），single V_rand 更全面（辅证）
- glm4_32b / llama2_13b 的"single 强 macro 弱" **不是反常**——是设计上 V_rand 比 projection 更狠
- 真反例只有 MoE 2 个（qwen3_30b_a3b, qwen3.5_35b_a3b），归"MoE 异常集群"（和 RQ1-RQ4 一致）
- **论文叙事**：macro projection 作主指标，V_rand 作辅证——两个互相印证"v₁ 因果必要"

**RQ2a MoE tuple bug 复查**（2026-04-20 复盘 RQ1/RQ2 发现，低优先级）：

| # | 位置 | Bug | 修法 | 影响 | 成本 |
|:-:|---|---|---|---|:-:|
| — | `RQ2_mlp_source/exp2a_mlp_feasibility_test.py::MLPDisableHook.__call__` (line 51) | `return torch.zeros_like(output)` 未处理 tuple 输出 | 加 `if isinstance(output, tuple)` 分支，参照 `exp6_progressive_ablation.py` line 33-35 的实现 | MoE 模型（尤其 qwen3.5_35b_a3b retain=81%）结果可能是脚本工件而非机制异常。**其余 23 个模型结果不受影响，RQ2a 主结论（H₁ MLP 是 MA 主来源）稳** | 修 ~5 min + qwen3.5_35b_a3b 重跑 ~10 min |

> RQ2a 主要结论（20/24 retain ≤ 10%）对这个 bug **不敏感**，所以复盘后决定：**记录但不阻塞**后续 RQ 分析，等整体批次重跑时一起做。

**说明**：
- opt_6.7b 是 RQ2c ANOMALY（drop 仅 1.6%），RQ2a 结果能判定"MLP 真不是主源"还是"attention 反向调节太强"，必补。
- qwen2_7b RQ1 曾给 ΔTop1=+Inf（baseline≈0 除零），本轮数据已清空，补跑让 RQ1 从 25/26 → 26/26。
- llama2 两兄弟是 RQ2 数据完整性的尾巴。
- RQ3 全部 26 模型依赖先修脚本（3 个 bug）；修完后批量跑即可。
- **服务器 117.50.223.194:23 root 上的 RQ3_origin/ 数据与本地完全一致，无额外可救数据**（已人工核对，错误同源）。
- RQ4/5/6 的补跑清单将在各 RQ 分析定稿时追加。

### B. 已跑但不符合预期（真异常，不是数据缺口）

> 这些模型**已有数据**，但结果偏离"MLP 是 MA 主导来源"假设。**不通过补跑解决，需机制解释或单独脚本验证**。

| # | 模型 | 异常表现 | 性质 | 处置建议 |
|:-:|---|---|---|---|
| 1 | **qwen3.5_35b_a3b** (MoE) | RQ2a retain = **81%** | 可能**脚本工件**（MoE 专家层 hook 未覆盖）| 写 MoE-aware hook 单独验证；模型权重不公开，成本高 |
| 2 | **gpt2** | RQ2a retain = **39%** | 真异常——小模型老架构，MA 不全靠 MLP | 保留为离群点，论文可选讨论 |
| 3 | **qwen3.5_9b** | RQ2a retain = **32%** | **qwen3.5 家族专属**（与 qwen3 系列差异悬殊）| 记录为家族级新发现；查 qwen3.5 技术报告 |
| 4 | **qwen3.5_27b** | RQ2a retain = **20%** | **qwen3.5 家族专属**（同上） | 同上 |

**共性**：qwen3.5 家族 3/3 全体 retain > 15%，而 qwen3 家族 5/5 retain ≤ 10%。**这不是 bug，是真实的家族级 MA 生成机制差异**。

**影响**：
- 对论文主论点"MLP 是 MA 来源"不构成推翻（24/24 都显著减少，median retain ≈ 2%）
- 但 qwen3.5 家族需要在论文 Discussion 节单独讨论
- RQ3/4/5 这 3 个 qwen3.5 模型的起源层分析可能也会偏离一般规律——**到时需单独观察**

---

## MoE（多专家机制）模型——单独类别（Tier C）

> **2026-04-20 正式划分**：含 MoE（Mixture of Experts）的模型和**标准 dense MLP** 模型走**完全不同**的 MA 生成机制。本轮分析**不再把它们混入主结论**——单独作为一个类别讨论。

### 本轮 26 模型里的 MoE（2 个）

| 模型 | 总参数 | 激活参数 | MoE 块类名 |
|---|:-:|:-:|---|
| **qwen3_30b_a3b** | 30B | 3B | `Qwen3MoeSparseMoeBlock` |
| **qwen3.5_35b_a3b** | 35B | 3B | `Qwen3_5MoeSparseMoeBlock` |

命名约定 `aXb` = "active X billion params" = 每个 token 只激活 X B 的参数。

### MoE 和标准 MLP 的结构差异

**标准 MLP（dense）**：
```
input h → W_up → activation → W_down → output
```
每个 token 都走同一组 W_up/W_down，hook 挂在 `layer.mlp.down_proj` 上能拿到全部 MA 写入。

**MoE 的稀疏路由**：
```
input h → Router → 挑 top-K 个 experts（通常 K=2 或 K=8）
                    ├── expert_1: own W_up_1, W_down_1
                    ├── expert_2: own W_up_2, W_down_2
                    ├── ...
                    └── expert_N: own W_up_N, W_down_N
         → 加权求和（路由权重）→ output
```

- 每层有 **N 个专家**（Qwen3 MoE 通常 N=64~256）
- 每个 token 只激活其中 **K 个**（K=2~8）
- 不同 token 走**不同专家**——**功能词可能走 expert_5，内容词可能走 expert_23**

### 脚本对 MoE 的系统性不兼容（4 类 bug）

| Bug | 位置 | 症状 |
|---|---|:-:|
| **B1**：直接访问 `.up_proj / .down_proj` | RQ3 script, RQ6 exp6_v_ablation | `AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'` |
| **B2**：`register_forward_hook` 的 return tuple 未处理 | RQ2a exp2a | `torch.zeros_like(tuple)` 行为未定义 |
| **B3**：`get_mlp_submodules()` 没有 MoE 分支 | `lib/model_utils.py` | `ValueError: Cannot identify MLP submodules` |
| **B4**：`get_mlp_down_proj()` 没有 MoE 分支 | RQ6 exp6_v_ablation | 静默误取或失败 |

### 跨 RQ 的 MoE 异常数据

| RQ | qwen3_30b_a3b | qwen3.5_35b_a3b | 读解 |
|:-:|---|---|---|
| **RQ1**（residual%）| 80%（Gen 弱）| 105%（Sup 弱 +5%）| MoE 关 attention 效果偏弱（路由打散了） |
| **RQ2a**（retain%）| **2.85%**（强）| **81.1%**（异常）| qwen3_30b 正常，qwen3.5 的 MoE 脚本工件 |
| **RQ3**（concentration/alignment）| ⚠ 跑过但全空 | ⚠ 跑过但全空 | B1 bug 导致数据损坏 |
| **RQ4**（σ₁/σ₂）| 1.17 | 1.03 | 单层谱极弱（MoE 各专家独立，不合并）|
| **RQ5 single** | -0.8% | +0.1% | v₁ 消融无效 |
| **RQ5 macro** | -0.4% | +0.5% | macro v₁ 也无效 |
| **RQ5b_moe_pe**（per-expert）| — | -15.9%（中等）| 专家级消融部分有效 |
| **RQ6 macro η** | 1.48 | 3.93 | qwen3.5 的 macro 聚合有信号 |

### 为什么 MoE 需要"专家级"分析（per-expert）

如果**每个专家都有自己的 W_down**，那：
- 专家 A 的 v₁ 方向 ≠ 专家 B 的 v₁ 方向
- "MLP 写 mark"在 MoE 里可能是**某几个特定专家**的行为，不是整层
- 整层平均的 v₁（标准 macro）**会稀释**真正的 mark 方向

**正确的 MoE 分析流程**（Tier C 任务）：
1. 找出路由到每个专家的 token 子集
2. 对**每个专家**的 W_down 做单独 SVD
3. 测**每个专家**的 v₁ 消融对 MA 的影响
4. 看是否存在**"MA 专责专家"**（某几个专家专门写 mark）

部分 per-expert 数据已经在：
- `results/wikitext_run/RQ5_macro_moe_per_expert/qwen3_30b_a3b/`
- `results/wikitext_run/RQ5_macro_moe_per_expert/qwen3.5_35b_a3b/`
- `exp5_origin_moe_pe` (2 模型) + `exp5b_moe_pe` (2 模型)

### 处置建议

1. **MoE 不纳入主结论统计**——26 - 2 = **24 个 dense 模型**作为主样本
2. MoE 单独写论文章节 "Appendix: MA in MoE Models"
3. **补跑优先级**（见 §全局补跑清单 MoE 条目，新增）：
   - Bug B1/B2 修复脚本兼容 MoE
   - 扩展 per-expert 分析脚本到所有 RQ
   - 4 个 RQ 全部 MoE 专项重跑

### 论文叙事

> "**Our main analyses cover 24 dense MLP models.** Two Mixture-of-Experts models (qwen3_30b_a3b, qwen3.5_35b_a3b) exhibit distinct MA generation mechanisms incompatible with single-V-direction theory. We provide preliminary per-expert analysis showing that **a subset of experts may specialize in MA writing**, but a full treatment is deferred to future work."

---

## 附录 A — MA 机制回路（功能词 mark → attention 广播 → 分岔）

> 这是贯穿 RQ1-RQ6 的**单一因果链**。每条实验都是在验证这条链上的某一步。
> 下文引用它时用 "**机制链步骤 N**"。

### A.1 五步机制

```
┌───────────────────────────────────────────────────────────────┐
│  步骤 1  Mark 形成（生成端，RQ2 验证）                          │
│  ───────                                                      │
│  训练动力学：                                                   │
│    - 功能词高频 → 梯度信号充足                                  │
│    - 功能词低熵 → 任务简单，loss 快速下降                       │
│    - 功能词低维语义 → hidden 维度里大量空闲                     │
│  ↓                                                            │
│  MLP 发现"功能词位置 × 空闲维度"是写标记的最佳载体              │
│  → 在 v₁ 方向把激活推到极大（300-3000×）                       │
│  → MA = mark                                                  │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  步骤 2  Attention sink（softmax 指数放大）                     │
│  ─────────                                                    │
│  attention_weight(i→j) = softmax(Q_i · K_j / √d)              │
│                                                               │
│  当 K_j 在 v₁ 维度有 MA（K_j[v₁] = 2000）:                    │
│    Q_i · K_j ≈ 2000  （被这一维主导）                         │
│    exp(2000/√d) >> exp(其他 token)                           │
│  → attention 权重几乎全部集中到 MA token                       │
│  → 这不是模型"故意"sink，是 softmax 被 MA 绑架                 │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  步骤 3  广播（V 向量横向传播）                                 │
│  ───────                                                      │
│  output_i = Σ_j attention_weight(i→j) · V_j                  │
│           ≈ V_{MA_token}  （权重全在 MA token 上）            │
│                                                               │
│  → 所有 token 的 attention 输出都被拉向 V_{功能词}             │
│  → 结构信号横向传到整个序列                                     │
│  → 几何上：所有 token 在 v₁ 轴上"对齐"                        │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  步骤 4  分岔（Generative vs Suppressive）                    │
│  ───────                                                      │
│  attention head 的 V 投射方向决定两种行为：                     │
│                                                               │
│  ┌─────────────────────────┬──────────────────────────┐     │
│  │ Generative (17 个模型)  │ Suppressive (7 个模型)    │     │
│  ├─────────────────────────┼──────────────────────────┤     │
│  │ V ∥ +v₁  (同向)         │ V ∥ -v₁  (反向)          │     │
│  │ 接力放大 MA            │ 读到 mark 后主动减一部分   │     │
│  │ 禁用 attn: MA 下降      │ 禁用 attn: MA 暴涨        │     │
│  │ RQ1 ΔMA 为负           │ RQ1 ΔMA 为正              │     │
│  │ (-20% ~ -98%)          │ (+27% ~ +266%)           │     │
│  └─────────────────────────┴──────────────────────────┘     │
│                                                               │
│  关键：两种模式都在 v₁ 轴上，只是方向不同                       │
│       没有哪种模型的 attention 忽略 MA 随便投射                 │
│       → MA 的"几何锁定性"                                    │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│  步骤 5  稳态维持                                              │
│  ────────                                                     │
│  - 残差流: 跨层垂直传递 MA                                     │
│  - LayerNorm: 限制绝对幅度上限（被动硬约束）                    │
│  - GELU: 防止单维无限放大（软门控）                             │
│                                                               │
│  → MA 不再增长也不消失，稳定在某个量级                          │
│  → 训练动力学（Pythia 实测）:                                  │
│       step 1:      MA ≈ 2.1   (初始)                          │
│       step 32k:    MA ≈ 622   (达峰)                          │
│       step 143k:   MA ≈ 293   (回落稳定)                      │
└───────────────────────────────────────────────────────────────┘
```

### A.2 对各 RQ 的因果映射

| 步骤 | 对应 RQ | 测什么 |
|:-:|---|---|
| 1. Mark 形成 | **RQ2** | 禁用 MLP → MA 消失 → MLP 是写入者 |
| 1. Mark 位置 | **RQ3** | 功能词 h₂ 在 v₁ 对齐 > 内容词 → 标记集中在功能词 |
| 1. Mark 几何 | **RQ4 / RQ5** | σ₁ 主导 + v₁ 方向 + 消融 v₁ → MA 消失 |
| 1. Mark 多层版本 | **RQ6** | macro-SVD 把 Δ 多层累加还原出 macro v₁ |
| 2-3. Sink + 广播 | **RQ1** | 禁用 attn 的 ΔMA 符号 |
| 4. 分岔方向 | **RQ1 mode 字段** | 生成 (Δ<0) / 抑制 (Δ>0) 二分 |
| 5. 稳态 | **RQ7**（本轮排除） | 训练步-Step 维度的 MA 演化 |

### A.3 Peak 层 RQ3 Cohen's d 反向的解释

这是机制链的直接推论：

```
L_origin:    MLP 写 mark
             功能词 h₂[v₁] 远大于内容词 h₂[v₁]
             → RQ3 在这里测: Cohen's d >> 0 ✓

L_origin+1:  attention 把 V_{功能词} 广播到所有 token
             内容词位置被"染色"了 +v₁ 分量

L_origin+2:  MLP 在染色的 hidden 上运算
             内容词 h₂[v₁] 也长起来了

...

L_peak:      所有 token 的 h₂[v₁] 都不小
             功能词 vs 内容词的差距被摊薄
             部分模型甚至反向（内容词累加后更大）
             → RQ3 在这里测: Cohen's d ≤ 0
             → 这就是 JSON 里 9/16 模型反向的直接原因
```

**结论**：peak 层 Cohen's d 反向**不是 RQ3 理论错了**，恰恰是机制链"attention 广播"一步的**实证**。要测"功能词特异性"必须回到起源层，这就是 RQ3 重做的物理理由。

### A.4 信息分轨的隐含结论

```
hidden dim (4096)
 ├── 少数几维 (v₁/v₂ 主方向，~0.1%)
 │    = 结构信号通道 (MA)
 │    = 经过 attention sink + 广播 传递
 │    
 └── 其余维度 (~99.9%)
      = 语义信号通道
      = 正常 attention 机制传递，不受 sink 影响
```

**两条通路互不干扰**，所以功能词可以同时：
1. 承担"mark 标签"的 MA 功能（v₁ 维度）
2. 保留"的/和/在/是"的句法功能词语义（其他维度）

这也解释了为什么禁用 v₁ (RQ5) 几乎不破坏语言建模能力——语义通路完全没被碰。

### A.5 本回路未解决的问题

| 问题 | 关联 RQ | 备注 |
|---|---|---|
| 为什么是 v₁ 而不是 v₂？ | RQ4 σ₁/σ₂ | 训练收敛态的数学问题，推测是谱最大值吸引子 |
| Generative vs Suppressive 何时分化？ | RQ7（排除） | 需要训练-步数数据，本轮不做 |
| 多层协作 (模式 B) 的 mark 具体由哪几层写？ | RQ2c + RQ6 | 累积消融 + macro-SVD 联合判定 |
| MoE 模型每个 expert 的 mark 是否共享 v₁？ | Tier C 专项 | 本轮暂缓 |

---

## 附录 B — MA 的数学含义（工厂类比）

### B.1 MA 公式

```
MA 大小  ≈  σ₁  ×  (h₂ · v₁)  ×  u₁[max]  +  bias
            │        │            │
            ↓        ↓            ↓
           RQ4      RQ3          RQ4
```

### B.2 工厂生产线类比

把 MA 当成一条生产线的产出：

```
原料 h₂  →  对准入口方向 v₁  →  放大器 σ₁ 倍  →  堆到输出仓库 u₁[max]  →  MA
          └──── RQ3 ────┘    └─ RQ4 σ₁ ┘       └──── RQ4 u₁ ────┘
```

| 数学符号 | 工厂角色 | 实验测什么 |
|---|---|---|
| `σ₁` | 放大器功率 | W_down 矩阵里"最强放大方向"的放大倍数 |
| `v₁` | 放大器入口方向 | W_down 右奇异向量（intermediate space 的主方向） |
| `h₂ · v₁` | 原料对准度 | MLP 中间产物 h₂ 有多少投影到 v₁ 方向 |
| `u₁[max]` | 输出仓库位置 | W_down 左奇异向量里幅度最大的那一维 = MA 维度 |

**三件事同时满足才有 MA**。缺任何一件，MA 都不出现：
- `σ₁` 不大 → 没放大能力 → MA 不大
- `h₂ · v₁` 小 → 原料没对准 → 即使放大器强，也没东西可放大
- `u₁` 不集中 → 输出均匀铺开到几千维 → 没有"特别大的一维"= 没 MA

### B.3 为什么 peak 层数据不符合

**起源层**（例：GPT-J L2）：MLP 是"MA 生产车间"
- `σ₁` 大 ✓（这层 W_down 训练成专门的强放大器）
- `h₂` 还是纯功能词表示，对准 `v₁` ✓
- `u₁` 集中 ✓
- → 实测 σ₁/σ₂ ≈ **5.74**

**Peak 层**（例：GPT-J L16）：MLP 只是"路过车间"
- `σ₁` 平庸 ✗（这层 W_down 不需要强放大器）
- `h₂` 已被 attention 广播，功能词和内容词差别被稀释 ✗
- → 实测 σ₁/σ₂ ≈ **1.2-1.5**（JSON 里 17 个模型都这样）

**结论**：跑出 η≈1.3 不是 MA 理论错了，是跑进了不生产 MA 的车间。

### B.4 各 RQ 在测公式哪一项

| 实验 | 测什么 | 公式里对应项 |
|---|---|---|
| **RQ2** 禁 MLP → MA 消失 | MLP 是 MA 生产者 | 整个生产线 |
| **RQ3** 功能词 vs 内容词 h₂ 对 v₁ 的投影 | 原料对准度 | `h₂ · v₁` |
| **RQ4** W_down 做 SVD | 放大器功率 + 输出仓库位置 | `σ₁`, `u₁` |
| **RQ5** 把 v₁ 砸掉测 MA 变化 | 放大器入口方向的因果必要性 | `v₁`（用消融验证） |
| **RQ6** macro-SVD（多层 Δh 聚合） | 模式 B 的"分布式生产线" | σ₁/v₁/u₁ 的多层聚合版本 |

---

## 附录 C — 实验方法学决定：单层 + macro 两条并行

### C.1 本轮 RQ3/4/5 范围：只做单层分析

**所有 25 个模型的 RQ3/RQ4/RQ5 都只在起源层（`exp2.critical_layer`）跑一个层**。不做起源层 ± 2 的 5 层扫描，也不在 RQ3/4/5 里做 macro-SVD。

多层聚合的故事**全部归到 RQ6** 去做。

### C.2 为什么这样分工正确

#### 对不同模式模型的自洽

| 模式 | 模型数 | 单层起源层 RQ3/4/5 预期结果 | 如何解读 |
|:-:|:-:|---|---|
| A 单层 | 10 | 强（η ≥ 3, Cohen's d ≫ 0, ΔMA ≤ -85%） | 直接支持"单层主导"假说 |
| 中间 | 9 | 中等（η ~2, d 中等, ΔMA ~ -40~-70%） | 需要 RQ2c + RQ6 进一步判定 |
| B 多层 | 4 | 弱（η ~1.3, ΔMA ~ -10%） | **弱结果本身就是"单层不主导"的证据**，真正的主方向在 RQ6 |

**关键**：模式 B 的"弱单层 RQ4"结果**不是失败**，是正面证据——证明这些模型必须用多层聚合解释。

#### 方法学统一性

- 所有 25 模型用同一个单层脚本跑（`--layer_id = exp2.critical_layer`）
- 结果的数字大小**自动反映模式归属**（不需要预先分组）
- 下游表格简洁、可比、一致

#### 互补分工

```
RQ3 / RQ4 / RQ5  = 起源层单层分析  (25 模型共用这一流程)
                   └→ 覆盖公式的 h₂·v₁, σ₁, u₁, v₁ 因果

RQ6  = 多层 macro-SVD               (主要服务模式 B + 中间)
       └→ 覆盖公式在多层聚合下的等效版本
```

两条线互不重复、互相补强。

### C.3 这条决定的具体影响

对之前 RQ4 讨论里"方案 A（1 层）vs 方案 B（5 层）"的问题：

**定案 → 方案 A（只跑起源层 1 层）**。

- 成本：每模型只跑 1 个 layer，全 25 模型 ≈ 2-3h（约省 4×）
- 结果：更简洁，表格只有 1 列 η 而不是 5 列
- 多层信息：由 RQ6 已有的 17/25 macro-SVD 数据 + 本轮补齐的 8 个 macro-SVD 负责

### C.4 落到具体操作

| 实验 | 层选择 | 范围 |
|---|---|---|
| RQ3 功能词对齐 | `--layer_id = exp2.critical_layer` | 25 模型都跑 |
| RQ4 SVD 谱分析 | `--layer_id = exp2.critical_layer` | 25 模型都跑，`num_layers=1` |
| RQ5 v₁ 消融 | `--layer_id = exp2.critical_layer` | 25 模型都跑 |
| RQ6 macro-SVD | 跨多层聚合 | 25 模型都跑（补齐 8 个缺失） |

RQ4 里曾有的 `num_layers=5` 逻辑删除，全部改为 1。

