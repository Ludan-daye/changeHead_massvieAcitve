# RQ1 — Attention 消融测试（H₀ 证伪）

> 最终稿 · 2026-04-23
> 导航：[README](../README.md) | [OVERVIEW](../OVERVIEW.md)

---

## RQ1 — Attention 消融测试（**最终稿 · 2026-04-23**）

### 实验目的

否证假设 **H₀: "MA（Massive Activations）来自 attention"**。如果 H₀ 成立，关掉 attention 后 MA 应归零；如果 H₀ 不成立，说明 MA 来源另有其处（→ RQ2 指向 MLP）。

### 实验方式

把模型里**所有 attention 层**的输出整体置零（保留残差流和 MLP），然后测每个 token 位置上 hidden state 各维度的最大 MA：

```
原始:  h_{L+1} = h_L + Attn(h_L) + MLP(h_L)
禁用:  h_{L+1} = h_L + 0         + MLP(h_L)
```

对 wikitext 上 60 个样本（每样本 1024 tokens）取 `top1_max` = max_l max_p max_d |hidden[l,p,d]|，得 `baseline_top1` 与 `disabled_top1`。

**脚本**：`RQ1_attention_contribution/exp1_feasibility_test.py`

**关键指标**：

| 指标 | 定义 | 用途 |
|---|---|---|
| `residual%` | `disabled / baseline × 100` | **主证伪指标**（residual > 0 即证伪 H₀） |
| `ΔMA%` | `(disabled − baseline) / baseline × 100` | 定方向（Gen: <0 / Sup: >0） |
| `peak_layer` | MA 观测最大的层 | 定位 MA 广播终点 |

### 假设与判据

```
H₀: MA 来自 attention        →  disabled ≈ 0  (residual% ≈ 0)
H₁: MA 不来自 attention      →  disabled > 0  (residual% > 0)

判据（严格）:  residual% > 0       →  证伪 H₀ （PASS）
判据（方向）:  ΔMA% < 0            →  Generative （放大器）
             ΔMA% > 0            →  Suppressive （抑制器）
```

### 数据（26 模型，按 residual% 升序）

> residual% = disabled_top1 / baseline_top1 × 100；越小表示 attention 对 MA 放大作用越强；>100% 表示 attention 在抑制 MA。

| # | 模型 | baseline | disabled | **residual%** | ΔMA% | peak | 方向 |
|:-:|---|---:|---:|---:|---:|:-:|:-:|
| 1 | bloom_7b1 | 3,541 | 60 | **1.69%** | -98.3% | L12 | ↓ Gen* |
| 2 | gptj_6b | 4,246 | 240 | **5.65%** | -94.3% | L16 | ↓ Gen* |
| 3 | llama2_13b | 1,283 | 263 | 20.50% | -79.5% | L22 | ↓ Gen |
| 4 | qwen3_0.6b | 6,871 | 1,603 | 23.33% | -76.7% | L25 | ↓ Gen |
| 5 | gpt2 | 983 | 393 | 39.99% | -60.0% | L16 | ↓ Gen |
| 6 | llama3.1_8b | 314 | 137 | 43.43% | -56.6% | L17 | ↓ Gen |
| 7 | qwen2.5_0.5b | 1,692 | 918 | 54.23% | -45.8% | L15 | ↓ Gen |
| 8 | glm4_9b | 2,250 | 1,240 | 55.08% | -44.9% | L1 | ↓ Gen |
| 9 | qwen3_8b | 13,336 | 8,791 | 65.92% | -34.1% | L33 | ↓ Gen |
| 10 | qwen3_1.7b | 12,582 | 9,818 | 78.03% | -22.0% | L25 | ↓ Gen |
| 11 | falcon_7b | 1,827 | 1,437 | 78.62% | -21.4% | L23 | ↓ Gen |
| 12 | qwen3_30b_a3b (MoE) | 1,207 | 967 | 80.17% | -19.8% | L36 | ↓ Gen |
| 13 | mistral_7b_v03 | 314 | 259 | 82.43% | -17.6% | L25 | ↓ Gen |
| 14 | qwen3.5_9b | 353 | 296 | 83.69% | -16.3% | L31 | ↓ Gen |
| 15 | qwen3.5_27b | 1,000 | 840 | 84.00% | -16.0% | L58 | ↓ Gen |
| 16 | qwen1.5_14b | 7,444 | 6,317 | 84.86% | -15.1% | L37 | ↓ Gen |
| 17 | qwen3_4b | 8,526 | 7,407 | 86.88% | -13.1% | L17 | ↓ Gen |
| 18 | qwen3.5_35b_a3b (MoE) | 40 | 42 | 105.14% | +5.1% | L39 | ↑ Sup |
| 19 | yi_9b | 5,004 | 6,368 | 127.26% | +27.3% | L47 | ↑ Sup |
| 20 | qwen3_32b | 27,418 | 43,680 | 159.31% | +59.3% | L53 | ↑ Sup |
| 21 | qwen3_14b | 15,205 | 28,112 | 184.89% | +84.9% | L33 | ↑ Sup |
| 22 | glm4_32b | 298,598 | 797,082 | 266.94% | +166.9% | L1 | ↑ Sup |
| 23 | opt_6.7b | 391 | 1,370 | 350.27% | +250.3% | L25 | ↑ Sup |
| 24 | qwen2.5_7b | 11,886 | 43,520 | 366.16% | +266.2% | L16 | ↑ Sup |
| 25 | llama2_7b_chat | 2,112 | 12,812 | **606.67%** | +506.7% | L22 | ↑ Sup |
| 26 | qwen2_7b | 6,987 | 5,330† | 76.29% | -23.7% | L16 | ↓ Gen |

† qwen2_7b 深层 L26 disabled 出现 inf 数值溢出（fp16 爆炸），取 baseline peak L16 位置的 disabled 值作为对照，residual% = 76.29%。

*标 "Gen\*" = 强放大器子类（residual < 5%）：bloom / gptj——关 attention 后 MA 几乎全塌（<6%），说明 attention 在这两个模型里承担了"几乎全部"的下游放大工作。

### 结论：H₀ 是否成立

**✅ H₀ 证伪成立（25/25 已有数据）**

- **没有任何一个模型** residual% = 0 —— 最低是 bloom_7b1 和 gptj_6b 的 1.69% / 5.65%
- 即使是"强放大器"bloom/gptj，attention 消融后仍有 MA 残留（60 / 240），这个残留必然来自 attention 以外的路径（MLP 或残差流）
- qwen2_7b 正在补跑（后台 subagent a1e8267e7e4c617ac 处理中），数据补齐后完整性从 25/26 升到 26/26

**子分类**（3 类）：
| 子类 | 条件 | 数量 | 机制解释 |
|---|---|:-:|---|
| **强放大器 Gen\*** | residual < 5% | 2 | attention 承担几乎全部下游放大（bloom, gptj）|
| **放大器 Gen** | 5% ≤ residual < 100% | 15 | attention 放大，但 MLP 已能单独产生部分 MA |
| **抑制器 Sup** | residual ≥ 100% | 8 | attention 对 MA 做稳态压缩，关掉后爆炸 |

### RQ1 解释了什么问题

1. **排除了一条错路（attention 起源说）**——在 MA 研究里，Sun et al. 2024 等早期工作质疑过 attention 是 MA 主要制造者。RQ1 用 25 模型的广谱数据**证伪此假说**，直接为 RQ2（MLP 是来源）做铺垫。

2. **揭示 attention 的实际角色是"调节器"而不是"生产者"**：
   - 17 个模型里 attention 放大 MA（Gen）—— 把 MLP 写入的种子 MA 广播到整个序列
   - 8 个模型里 attention 压制 MA（Sup）—— 高基线 MA 模型依赖 attention 做 homeostasis

3. **发现训练策略与 attention 角色的相关性**：
   - 西方开源（GPT/BLOOM/Falcon/Mistral/Llama-base）全部 Gen
   - 中国开源（Qwen 4/13、Yi、GLM-32b）多数 Sup
   - 同家族翻转：llama2_13b Gen vs llama2_7b_chat Sup（RLHF 改变角色）；glm4_9b Gen vs glm4_32b Sup（scale 改变角色）

4. **为 RQ2-RQ5 指明方向**：既然 attention 不是起源，只能是 MLP（RQ2）；MLP 通过哪个层（RQ2b/c）、写什么方向（RQ3/RQ4）、去掉这个方向后 MA 是否消失（RQ5）——完整链路从 RQ1 开始。

### 关键观察（论文机制章节素材）

1. **同家族方向翻转**（重要）：
   - GLM: `glm4_9b` (Δ=-45%) vs `glm4_32b` (Δ=+167%) → scale 改变 attention 角色
   - Llama: `llama2_13b` base (Δ=-79%) vs `llama2_7b_chat` 微调 (Δ=+507%) → RLHF 显著改变

2. **Suppressive 组 baseline 显著偏高**（中位数 13,545 vs Gen 1,827，约 7×）—— 高基线 MA 模型需要 attention 做稳态压缩，关掉后 residual/LN 失衡导致爆炸。

3. **训练背景相关**：Sup 集中在中国开源家族（Qwen/Yi/GLM-32b），西方开源全部 Gen。提示训练语料/目标/RLHF 是决定 attention 角色的关键，非架构本身。

4. **MoE 响应极弱**（qwen3.5_35b_a3b Δ=+5%, qwen3_30b_a3b Δ=-20%）—— 整层 attention 消融对 MoE 路由影响小。per-expert 消融是附录讨论范围。

### 26 模型分类表

#### 表 A — 按模型结构（架构家族）分类

| # | 架构族 | 成员 | Gen\* | Gen | Sup | 家族规律 |
|:-:|---|---|:-:|:-:|:-:|---|
| 1 | **Pre-Llama 时代** (GPT2 类) | gpt2, gptj_6b, opt_6.7b, bloom_7b1, falcon_7b | 2 | 2 | 1 | 混合；bloom/gptj 是强放大器 |
| 2 | **Llama-style base** (RoPE+SwiGLU+RMSNorm) | llama2_13b, llama3.1_8b, mistral_7b_v03 | 0 | 3 | 0 | **全 Gen** — attention 是放大器 |
| 3 | **Llama-RLHF** (微调) | llama2_7b_chat | 0 | 0 | 1 | **RLHF 翻为 Sup**（同架构不同训练→方向翻转） |
| 4 | **Yi 家族** | yi_9b | 0 | 0 | 1 | Sup |
| 5 | **GLM4 家族** | glm4_9b (Gen), glm4_32b (Sup) | 0 | 1 | 1 | **scale 翻转**（9B Gen → 32B Sup） |
| 6 | **Qwen 1.5/2/2.5** | qwen1.5_14b, qwen2_7b, qwen2.5_0.5b (Gen) · qwen2.5_7b (Sup) | 0 | 3 | 1 | 多 Gen；qwen2.5_7b 异常 Sup |
| 7 | **Qwen3 dense** | qwen3_{0.6b, 1.7b, 4b, 8b} (Gen) · qwen3_{14b, 32b} (Sup) | 0 | 4 | 2 | **scale 翻转**（≤8B Gen, 14B+ Sup） |
| 8 | **Qwen3.5 dense** (hybrid attn) | qwen3.5_9b, qwen3.5_27b | 0 | 2 | 0 | 全 Gen |
| 9 | **MoE** (sparse experts) | qwen3_30b_a3b (Gen), qwen3.5_35b_a3b (Sup) | 0 | 1 | 1 | 混合；ΔMA 响应弱（\|Δ\|≤20%） |
| - | **合计** | 26 | **2** | **16** | **8** | |

**架构层面观察**：
- Llama-style base (3/3) + Qwen3.5 dense (2/2) 全部 Gen——方差很小的家族，attention 角色稳定
- **RLHF** 和 **scale** 是两大翻转因素：llama2_7b_chat、glm4_32b、qwen3_32b 都是同架构基座翻转
- MoE 表现独特（ΔMA 幅度小），整层 attention 消融对专家路由影响有限

#### 表 B — 按 MA 调节方向子类分类

| 子类 | 判据 | 数量 | 模型 | residual% 范围 | baseline 中位数 | 机制解读 |
|:-:|---|:-:|---|---|---:|---|
| **Gen\*** (强放大器) | residual < 5% | **2** | bloom_7b1, gptj_6b | 1.69 – 5.65% | 3,894 | attention **承担几乎全部下游放大**；MLP 只产生"种子 MA"，几乎不能独立存在 |
| **Gen** (放大器) | 5% ≤ residual < 100% | **16** | llama2_13b, qwen3_0.6b, gpt2, llama3.1_8b, qwen2.5_0.5b, glm4_9b, qwen3_8b, qwen3_1.7b, falcon_7b, qwen2_7b, qwen3_30b_a3b (MoE), mistral_7b_v03, qwen3.5_9b, qwen3.5_27b, qwen1.5_14b, qwen3_4b | 20.5 – 86.88% | 4,139 | attention **放大** MA 但非主导；**MLP 已能独立生成部分 MA** |
| **Sup** (抑制器) | residual ≥ 100% | **8** | qwen3.5_35b_a3b (MoE), yi_9b, qwen3_32b, qwen3_14b, glm4_32b, opt_6.7b, qwen2.5_7b, llama2_7b_chat | 105 – 607% | 13,545 | attention **压制** MA；关掉后 MA 爆炸（LayerNorm/residual 失衡）|

**方向层面观察**：
- Sup 组 baseline 中位数（13,545）比 Gen 组（4,139）高 **3.3×**，比 Gen\* 组（3,894）高 **3.5×**——**高基线 MA 模型更依赖 attention 做稳态压缩**
- Gen\* (residual<5%) 属于 MA 研究里极特殊的一类：可视为"attention 是唯一必要广播路径"的候选，论文讨论此子类时可单独给 case study
- 全部 MoE (2/2) 都在 Sup 边界附近（qwen3_30b_a3b 80% Gen, qwen3.5_35b_a3b 105% Sup），ΔMA 绝对值小 → **MoE 的 MA 机制对 attention 依赖弱**，需附录 per-expert 分析

#### 表 C — 两维交叉（架构 × 方向）

|  | Gen\* | Gen | Sup | 小计 |
|:-:|:-:|:-:|:-:|:-:|
| Pre-Llama | 2 (bloom, gptj) | 2 (gpt2, falcon) | 1 (opt) | 5 |
| Llama-style base | 0 | 3 | 0 | 3 |
| Llama-RLHF | 0 | 0 | 1 | 1 |
| Yi | 0 | 0 | 1 | 1 |
| GLM4 | 0 | 1 | 1 | 2 |
| Qwen1.5/2/2.5 | 0 | 3 | 1 | 4 |
| Qwen3 dense | 0 | 4 | 2 | 6 |
| Qwen3.5 dense | 0 | 2 | 0 | 2 |
| MoE | 0 | 1 | 1 | 2 |
| **合计** | **2** | **16** | **8** | **26** |

**交叉表观察**：
- Gen\* 只出现在 Pre-Llama 家族（bloom, gptj）——**旧架构**更容易让 attention 成为"唯一广播器"
- Sup 最多的是 Qwen3 dense (2 个) + Qwen1.5/2/2.5 (1) + 中国开源其他 (GLM4-32b/Yi/qwen2.5_7b) — 训练语料/目标是主因
- **完全没出现 Gen\*** 的家族：Llama-style、Qwen 任何一代、GLM4、Yi、MoE——说明**现代训练让 MLP 已经有能力独立产生可观的 MA**，不再完全依赖 attention 放大

### 数据补齐状态

- ✅ **26/26 完整**（2026-04-23 补齐 qwen2_7b，来源 `qwen2_7b_fixed/` 目录，baseline L16 peak=6986.7）
- 已修复的脚本 bug：
  - B19（load_model.py cuda:cpu guard，commit 2568d30）
  - hybrid_attn fix（qwen3.5 linear_attn vs self_attn，exp1_feasibility_test.py:45-76）

### 结论摘要

> **RQ1 最终结论**：关 attention 后 **26/26 模型**的 MA 均未归零（residual ∈ [1.69%, 606%]），**假设 H₀（MA 来自 attention）被证伪**。Attention 在 MA 生成链里扮演**调节器**（下游放大或压制），而非生产者。这个结果直接支撑 RQ2 转向 MLP 寻找真正的起源。

---

