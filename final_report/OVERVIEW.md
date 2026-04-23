# MA 机制总览

## 1. 现象

LLM 推理时，某些 hidden state 维度的激活值会突然**爆炸**——比中位数大 300-3000 倍。这些"Massive Activations (MA)"：
- 只出现在**特定 token 位置**（通常是换行、标点、空格等 function token）
- 只出现在**特定 hidden 维度**（u₁ argmax 的 j\*）
- 只在**特定层之后**（L_origin 层的 MLP 后开始出现）

## 2. 五步机制链

```
Token 位置 x 的 MA 生成路径：

   [h_in(x)] ──LayerNorm──> [h_norm] ──MLP──> [h_output]
                                        ↓
                         (RQ2 证明: MLP 是起源)
                                        ↓
   MLP 内部:
     h_norm ──W_up──> [h₂]           ← intermediate state (RQ3 证明: FT 位置 h₂ 特殊)
     h₂ ─(element-wise act)─ [h₂_act]
     h₂_act ──W_down──> [h_output]   ← hidden state (这里产生 MA)

   W_down 的 SVD 分解:
     W_down = U Σ V^T
     h_output[j] = Σᵢ σᵢ · (vᵢ · h₂_act) · uᵢ[j]      (RQ4: 多奇异方向展开)

   MA 在 j* 维度 (u₁ argmax) 出现:
     MA[j*] = σ₁ · (v₁ · h₂_act) · u₁[j*] + 次级项     (RQ4 V3 公式)

   Attention 把 MA 广播到所有 token (RQ1: 调节器):
     每个 token 的 MA = residual 累加 + attention 放大/压制
```

## 3. 核心公式

### 3.1 单层 CONCENTRATED 模式

$$
\text{MA}(x) \approx \beta \cdot (h_2 \cdot v_1)(x) + b
$$

其中：
- **β = σ₁ · u₁[j\*]** — v₁ 方向的**有效增益**（由单次回归拟合得到）
- **b** — 截距，吸收次级奇异方向贡献 + 残差流 + 非 MLP 源

### 3.2 多层 FEW-SOURCE/DISPERSED 模式

$$
\text{MA}(x) \approx \beta_{\text{macro}} \cdot (\Delta h \cdot v_{\text{macro}})(x) + b
$$

其中 **v_macro** = 跨 origin_layers 累加的 Δh_macro 做 SVD 后的主方向。

### 3.3 严格多项式展开

$$
\text{MA}_{j^*} = \sum_{i=1}^{r} \sigma_i \cdot (v_i \cdot h_2) \cdot u_i[j^*]
$$

- r = rank(W_down) = min(hidden, intermediate)
- σ₁ 强主导时，只取 i=1 项（V3 近似）
- σ₁ 不主导时（扁平谱），需要 i=1..k 多项（k=2-10）

## 4. RQ 因果链

```
RQ1: attention 消融 → 证伪 H₀（MA 非来自 attention）
     ↓
RQ2: MLP 消融 → 验证 H₁（MLP 是起源）+ 分类模式 A/B
     ↓
RQ3: FT 定位 → 验证 MA 在 function_token 位置（Top-1 FT 92%）
     ↓
RQ4: SVD 验证 → MA = β·(h₂·v₁) + b 公式成立（14-17/26）
     ↓
RQ5: V 消融 → 因果验证（消 v₁ 后 MA 塌 17/26）
     ↓
RQ6: Recovery → 保留起源层单层恢复 MA
```

## 5. 三种模式

### 模式 A — CONCENTRATED（单层主导）

- **特征**：单一 W_down 层 σ₁ 主导；u₁ 稀疏（top1 ≥ 0.5）；v₁ 方向仅由 1-2 个 FT 投影
- **数量**：8/26 模型
- **代表**：gptj_6b (σ₁=14.4, R²=1), qwen2.5_7b (σ₁=17, R²=1), qwen2_7b, qwen3_0.6b
- **公式适用**：V3 单层公式精确（误差 < 1%）
- **因果验证**：单层 V 消融 ΔMA ≥ -93%

### 模式 B — FEW-SOURCE / DISPERSED（多层协作）

- **特征**：MA 在 5-15 层 MLP 分散生成；单层 σ₁ 小（5-40），但跨层累积 macro σ₁ 大（10³-10⁵）
- **数量**：16/26 模型
- **代表**：qwen3 dense 家族全部 DISPERSED；yi_9b, falcon_7b 等
- **公式适用**：macro V3 公式（macro σ₁·(Δh·v_macro)）
- **因果验证**：macro V 消融 ΔMA ≥ -80%（10/16）

### 模式 C — 架构异常

- qwen3.5 家族 (hybrid_attn 干扰)：macro σ₁ 大但 v_macro 消融无效
- MoE 2 个（专家级机制）
- opt_6.7b (ANOMALY_NO_MLP_RESPONSE)
- glm4_32b (fp32 数值特殊)

## 6. attention 角色（RQ1 副结论）

attention **不是 MA 生产者**，而是**调节器**：

| 子类 | residual% | 机制 |
|---|---|---|
| **Gen\*** (强放大器) | <5% | attention 承担几乎全部下游放大（bloom, gptj）|
| **Gen** (放大器) | 5-100% | attention 放大 MA，但 MLP 已能独立生成部分 |
| **Sup** (抑制器) | ≥100% | attention 对 MA 做稳态压缩，关掉后爆炸 |

**分布**：Gen\* 2 / Gen 16 / Sup 8。

## 7. 与学界的连接

- **attention sink**（Xiao et al. "Efficient Streaming LMs"）：观察到 BOS/换行是 sink 位置——本研究证明这就是**MA 位置**
- **massive activations 原始观察**（Sun et al.）：本研究给出了**生成机制的数学公式**
- **两种生成模式**：首次系统分析单层 vs 多层 MA，对 LLM 可解释性有启示

## 8. 论文主要贡献

1. **证伪 attention 起源说**（RQ1 26/26 模型残留）
2. **验证 MLP 起源**（RQ2 21/24 dense retain ≤ 10%）
3. **定位 MA 在 function_token**（RQ3 Top-1 FT 92%）
4. **给出 MA 生成公式**（RQ4 β·(h₂·v₁) + b，V3 误差 < 1% 在 4 模型）
5. **提出单层/多层两种模式**（RQ2c 分类）
6. **因果验证 v₁ 是 MA 方向**（RQ5 macro V 消融 17/26 PASS）
7. **揭示 attention 调节器角色**（RQ1 Gen/Sup 分类）

---

详细分析见各 RQ 目录的 `ANALYSIS.md`。
