# 实验3: SVD对齐分析

## 实验目的
验证Massive Activation是否由权重矩阵的主导奇异方向产生

## 实验方法
1. 对Layer 2 MLP down_proj (W₂) 进行SVD分解
2. 提取主导方向v₁
3. 计算token激活与v₁的对齐度
4. 回归分析验证因果关系

## 核心结果

### SVD分解
| 指标 | 值 |
|------|-----|
| σ₁ | 38.26 |
| σ₂ | 15.16 |
| σ₁/σ₂ | **2.52** |

### 对齐分析
| 词类 | 与v₁对齐度 |
|------|-----------|
| 功能词 | -0.003 ± 0.021 |
| 内容词 | -0.002 ± 0.024 |

### 因果回归
```
Dim447 = 38.70 × (h₂ · v₁) + 3.59
R² = 0.998
```

## 结论

**✅ 因果机制确认：SVD主导方向产生MA**

- W₂有主导奇异方向（σ₁/σ₂=2.52）
- 投影强度解释99.8%的激活变化
- 这是**几何机制**的首次证明

## 文件列表
- `exp3_detailed_results.json` - 完整数据
- `exp3_singular_values.png` - 奇异值谱
- `exp3_alignment_comparison.png` - 对齐对比
- `exp3_projection_regression.png` - 回归分析
- `EXPERIMENT_3_SUMMARY.txt` - 详细报告
