# 实验4: Attention SVD分析

## 实验目的
分析Mistral-7B各层Attention o_proj权重的SVD结构

## 实验方法
1. 提取各层`self_attn.o_proj`权重矩阵(4096×4096)
2. 进行SVD分解
3. 计算σ₁/σ₂比值判断是否存在主导方向

## 核心结果

| Layer | σ₁ | σ₁/σ₂ | 主导方向 |
|-------|-----|-------|---------|
| **Layer 0** | 0.87 | **1.10** | ❌ 无 |
| Layer 1 | 0.97 | 1.42 | ❌ 弱 |
| Layer 16 | 0.71 | 1.10 | ❌ 无 |
| Layer 30 | 1.35 | 1.56 | ⚠️ 中等 |
| **Layer 31** | 2.96 | **2.19** | ✅ 有 |

## 结论

**❌ Layer 0（关键层）无主导奇异方向**

- Layer 0的σ₁/σ₂=1.10，奇异值分布均匀
- Layer 31的σ₁/σ₂=2.19最高，存在主导方向
- Mistral的MA机制可能不是简单的SVD几何对齐

### 与其他模型对比
| 模型 | 关键层 | σ₁/σ₂ |
|------|--------|--------|
| Qwen | Layer 3 | **14.5** |
| BLOOM | Layer 28 | 1.18 |
| **Mistral** | Layer 0 | **1.10** |

## 文件列表
- `attention_svd.json` - 完整SVD数据
- `attention_svd_summary.json` - 摘要（含分析）
- `attention_svd.png` - 可视化图
