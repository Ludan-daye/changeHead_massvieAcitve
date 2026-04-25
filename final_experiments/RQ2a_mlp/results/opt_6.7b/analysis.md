# opt_6.7b — RQ2a mlp 分析

**模型分类**：`ANOM (Tier E)` | **真起源层 L**：`0` | **peak_layer**：`25-26`

**本 RQ 在问什么**：MLP 全消融：禁用全部 MLP 后 MA 是否归零（验证 'MLP 是 MA 起源'）

---

## 关键指标（hook fix 后实测）

| 指标 | 数值 |
|---|---:|
| baseline Top1 (Dim 138 @ L=25 peak) | **367.61** |
| disabled Top1 (关全部 MLP) | **181.48** |
| **ΔMA%** | -50.63% |
| **retention%** | **49.4%** |

**判据**：retention ≤ 10% → **❌ FAIL**

---

## 🔥 OPT 是 Tier E（MLP 仅占一半 MA）

主流模型关全 MLP 让 MA 几乎归零（retain ≤ 10%），**OPT 关全 MLP 仍保留 49% MA**。

**意义**：OPT 的 MA 不是 MLP 单独造的：
- 50% 由 MLP 贡献（部分符合 H₁）
- **50% 由非 MLP 通道维持**（attention + residual 联合）

→ MLP 不是 MA 唯一来源，**Tier E 真异常**

## 历史背景

之前 v2 JSON 标 `ANOMALY_NO_MLP_RESPONSE`（关全部 MLP MA 不动），那是 hook bug：
- 主服 `lib/model_utils.py` 加了 `get_mlp_module_for_hook()` 函数后修复（OPT 走 `layer.fc2` 而非 `layer.mlp`）
- 修复后 retain 从 100% (hook fail) → 49.4%（真实数据）

---

## 数据文件

- 主服 `paper_experiments/results/exp2a_mlp_feasibility_test/`（hook fix 后的结果）
- 副服 `secondary/exp2b_mlp_layer_ablation/`（32 层逐层数据，禁 L=1 后 L=0 MA 飙 15×）

---

## 总评

**此模型 × 此 RQ**：❌ FAIL（retain=49% > 10%）

**此模型综合评分**：**3/5（Tier E 架构特异）**

不再补实验：49% retain 已交叉证明非 MLP 主导，归附录单独讨论。

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。
