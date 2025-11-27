# 实验2c: MLP内部分析

## 实验目的
精确定位MLP内部哪个位置产生Massive Activation

## 实验方法
追踪Layer 2 MLP的4个检查点：
1. MLP Input (768-dim)
2. After Linear1 (3072-dim)
3. After GELU (3072-dim)
4. MLP Output (768-dim)

## 核心结果

| 检查点 | 维度 | Max激活 | Top1/Median |
|--------|------|---------|-------------|
| Input | 768 | 19.88 | 186× |
| After Linear1 | 3072 | 62.91 | 63× |
| After GELU | 3072 | 62.91 | 634× |
| **Output** | 768 | **2,342** | **8,596×** |

## 结论

**🔥 爆发点: MLP Output (Linear2/down_proj)**

- 从GELU后的63到Output的2342，增长**3623%**
- GELU对激活几乎无影响
- 证明down_proj权重矩阵是MA的数学来源

## 权重分析
- Linear2 max weight: 15.07
- Top贡献中间维度: dim 496, 681, 732

## 文件列表
- `exp2c_detailed_results.json` - 详细数据
- `exp2c_activation_flow.png` - 激活流图
- `exp2c_gelu_impact.png` - GELU影响分析
- `EXPERIMENT_2C_SUMMARY.txt` - 详细报告
