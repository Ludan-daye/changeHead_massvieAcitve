# BLOOM-7B1 - 综合分析报告

## 模型概览

- **模型名称**: bloom_7b1
- **关键层**: Layer N/A
- **最大激活值**: N/A

---

## RQ1: 巨量激活生成分析

### Exp1: Attention贡献分析

**实验方法**: 禁用所有Attention heads，对比激活变化

**关键发现**:

**✅ Attention Heads 产生 Massive Activation**

禁用后激活下降98%，证明BLOOM的MA由Attention产生，与Qwen（MLP产生）机制相反。

**数据路径**: [`RQ1_activation_generation/exp1_attention_contribution/`](RQ1_activation_generation/exp1_attention_contribution/data/)

### Exp2: 层级贡献分析

**关键层**: Layer N/A

**数据路径**: [`RQ1_activation_generation/exp2_layer_contribution/`](RQ1_activation_generation/exp2_layer_contribution/data/)

---

## RQ2: MLP层来源分析

**实验方法**: Hook Attention和MLP输出，对比最大激活值

| 指标 | 数值 |
|------|------|
| Attention输出Max | 26.97 |
| **MLP输出Max** | **92.50** |
| MLP/Attn比值 | 3.43x |

**结论**: ✅ MA来自MLP

**数据路径**: [`RQ2_mlp_source/verification.json`](RQ2_mlp_source/verification.json)

---

## RQ3: 功能词触发分析

**实验方法**: 统计Top5 MA位置的token类型（10样本x5=50个）

**无语义词占比**: **98.0%**

**Token类型分布**:

| 类型 | 数量 | 占比 |
|------|------|------|
| 功能词 | 26 | 52.0% |
| 标点符号 | 23 | 46.0% |
| 实义词 | 1 | 2.0% |

**结论**: ✅ MA主要出现在无语义词位置

**全局数据**: [`MA_POSITION_TOKEN_ANALYSIS.json`](../../MA_POSITION_TOKEN_ANALYSIS.json)

---

## RQ4: SVD对齐分析

**实验方法**: 
- Exp4: 对MLP down_proj进行SVD分解
- Exp4b: 计算MA方向与top-k右奇异向量的对齐度

**数据路径**: 
- [`RQ4_svd_alignment/exp4_svd/`](RQ4_svd_alignment/exp4_svd/)
- [`RQ4_svd_alignment/exp4b_alignment/`](RQ4_svd_alignment/exp4b_alignment/)

---

## RQ5: V矩阵消融分析

**实验方法**: 将MLP down_proj的V矩阵替换为随机正交矩阵

| 指标 | 数值 |
|------|------|
| Baseline MA | 394.20 |
| 消融后MA | 320.05 |
| **变化率** | **-18.8%** |

**结论**: ⚠️ V矩阵影响较弱

**数据路径**: [`RQ5_v_ablation/data/`](RQ5_v_ablation/data/)

---

## 总结

### 关键发现

1. **MA生成机制**: Attention主导
2. **MA来源**: MLP输出（3.4x于Attention）
3. **触发位置**: 98.0%出现在无语义词
4. **V矩阵依赖**: 弱（变化18.8%）

### 完整数据

所有原始实验数据保存在各RQ子目录中。

---

*生成时间: 2025-12-03*
