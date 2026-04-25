# opt_6.7b — RQ5 V 消融分析

**模型分类**：`ANOM (Tier E)` | **真起源层 L**：`0`（per-layer scan 显示无 surge，gradual ramp）| **peak_layer**：`25-26`

**本 RQ 在问什么**：V 矩阵消融：替换/投影消除 v₁ 方向后 MA 是否塌陷（因果验证）

---

## 关键指标

| 指标 | 主服 (L=1 错层) | **副服 L=0 真起源** |
|---|---:|---:|
| baseline MA | 216 | 较小 |
| σ₁ | 19.47 | — |
| ablated MA | 178 | — |
| **ΔMA%** | **-17.5%** | **-31.8%** |

**判据**：单层 V 消融 ΔMA ≤ -80% → **❌ FAIL**

即使在真起源层 L=0 跑，ΔMA 也只 -32%，远不到 -80% 阈值。

---

## 🔥 OPT 是 Tier E（V 消融无效）

主流模型在真起源层做 V 消融，ΔMA 通常 ≥ -80%（如 gptj -99%, qwen2_7b -99%, bloom L=7 -70%）。  
**OPT 即使消 v₁ 也只降 32%**，说明 MA 不由 W_down 的 σ₁·v₁·u₁ 项主导。

**意义**：
- σ·v·u 项贡献 ≈ 32%
- 其余 68% 由其他通道维持（attention 反向调节 / residual stream 累积 / bias / MLP 多方向）
- → 公式 `MA = Σσ·v·u` 在 OPT 上**部分成立**但不充分

## 4 个独立证据共同证明 OPT Tier E

| # | 证据 | 数值 | 异常方向 |
|:-:|---|---:|---|
| 1 | RQ1 关 attn ΔMA | **+744%** | attention 是抑制器（vs 主流：放大器）|
| 2 | RQ2a 关全 MLP retain | **49%** | MLP 仅占一半 MA（vs 主流：≤10%）|
| 3 | exp2b 禁 L=1 后 L=0 飙 | **+15×** | 异常反向传递 |
| 4 | exp3_fire L=0→L=6 | **指数衰减** 200× | MA 不稳定（vs 主流：peak 区稳定）|
| 5 | **本实验 V 消融** | **-32%** | σ·v·u 仅占 32% |

**4-5 个交叉证据已足够定性 Tier E**，不再补 multi-K 实验（副服 SSH 失效，且边际收益低）。

---

## 数据文件

- `opt_6.7b_v_ablation_results.json`（主服 L=1 错层数据）
- `secondary/exp6/`（副服 L=0 真起源数据 + 完整 remove/keep_top_k）

---

## 总评

**此模型 × 此 RQ**：❌ FAIL（ΔMA=-32% > -80% 阈值）

**此模型综合评分**：**3/5（Tier E 架构特异）**

**论文叙事**：OPT 的 MA 由 attention + MLP + residual 联合维持，pre-LayerNorm + 非标 FFN 架构特殊，**不服从主公式**，单列附录 Tier E 讨论。**不削弱主论点**对 22 个主线 dense 模型的有效性。

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。
