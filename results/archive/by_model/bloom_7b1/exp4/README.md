# 实验4: Attention SVD分析

## 实验目的
分析Attention dense层权重的SVD结构，寻找主导奇异方向

## 实验方法
1. 提取各层`self_attention.dense`权重矩阵
2. 进行SVD分解: W = UΣVᵀ
3. 计算σ₁/σ₂比值判断是否有主导方向

## 核心结果

| Layer | σ₁ | σ₂ | σ₁/σ₂ | 主导方向 |
|-------|-----|-----|-------|---------|
| Layer 0 | 10.05 | 3.84 | **2.62** | ✅ 有 |
| Layer 7 | 6.82 | 3.70 | 1.85 | ❌ 无 |
| Layer 12 | 2.97 | 2.89 | 1.03 | ❌ 无 |
| **Layer 28** | 18.48 | 15.70 | **1.18** | ❌ 无 |
| Layer 29 | 22.84 | 10.26 | **2.23** | ✅ 有 |

## 结论

**Layer 28 没有主导奇异方向！**

产生MA的Layer 28反而σ₁/σ₂=1.18，说明BLOOM的机制不是SVD主导方向。

## 文件列表
- `attention_svd.json` - 完整SVD数据
- `attention_svd_summary.json` - 分析总结
- `attention_svd.png` - 可视化图表
