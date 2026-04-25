# opt_6.7b — RQ1 attention 分析

**模型分类**：`ANOM (Tier E)` | **真起源层 L**：`0`（per-layer scan 显示 gradual ramp 无 surge）| **peak_layer**：`25-26`

**本 RQ 在问什么**：Attention 消融：禁用全部 attention 层后 MA 是否仍存在（证伪 'attention 是 MA 起源'）

---

## 关键指标

| 指标 | 主服 | **副服更全数据** |
|---|---:|---:|
| baseline_top1 | 391.06 | 154 (L=0) |
| disabled_top1（关 attn） | 1369.75 | **1304** (L=0) |
| **ΔMA%** | **+250.26%** | **+744%** ⭐ |
| mode | generative | **抑制器（异常）** |

**判据**：residual% > 0 → **✅ PASS**（attention 消融后 MA 未归零，证伪 H₀）

---

## 🔥 OPT 是 Tier E（架构特异）

OPT 的 attention **不是** MA 的生产者或放大器，而是**抑制器**：
- 关 attention → MA 不仅没消，反而 **暴增 7-8 倍**
- 主流模型（如 bloom、gpt-j、qwen 系列）关 attention → MA 大幅下降（attn 是放大器）
- → OPT 的 attention 在 MA 系统里起**反向调节**作用

## 为什么 OPT 是 Tier E

OPT 架构特殊：
1. **pre-LayerNorm**（vs 主流 post-LN）→ residual stream 流向不规范
2. **非标 FFN**：fc1/fc2 直接挂 layer，无 mlp 包装
3. **MA 由 attention + MLP + residual 联合维持**，不是 MLP 单独主导
4. exp2b: 禁某层让前层 MA 飙 15×（异常反向传递）
5. exp3_fire: MA L=0 (1147) → L=6 (5.5) **指数衰减**（vs 主流模型 MA 在 peak 区稳定）

→ **不符合主公式 `MA = Σσ·v·u`，归 Tier E 附录单独讨论**

---

## 数据文件

- `secondary/exp1_opt_6.7b/`（副服真数据：baseline + all_heads_disabled）
- 副服路径：`/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/archive/by_model/opt_6.7b/`

---

## 总评

**此模型 × 此 RQ**：✅ PASS（证伪 H₀）

**此模型综合评分**：**3/5（Tier E 真异常）**

不再补 multi-K V 消融实验：4 个独立证据已交叉证明 Tier E 真异常，论文附录单独讨论。

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。
