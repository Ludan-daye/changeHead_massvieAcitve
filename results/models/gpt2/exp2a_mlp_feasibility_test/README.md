# 实验2a: MLP可行性测试

## 实验目的
验证MLP层是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型
2. **All MLP Disabled**: 将所有12个MLP层的输出置零
3. **对比**: 分析激活值变化

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 3,021 (Layer 10) | 1,164 | **-61%** |
| Dim 447 | 3,021 | 1,164 | -61% |
| Dim 138 | 796 | 300 | -62% |

### 关键层变化
| Layer | Baseline | Disabled | 变化 |
|-------|----------|----------|------|
| Layer 2 | 2,475 | 55 | **-98%** |
| Layer 3 | 2,648 | 50 | -98% |

## 结论

**✅ MLP层是Massive Activation的主要来源！**

- 禁用MLP后激活下降61%
- Layer 2是MA爆发点（从0→2475）
- 证明MLP产生MA，Attention读取MA

## 文件列表
- `baseline/` - 基线测试结果
- `all_mlp_disabled/` - 禁用MLP测试结果
- `comparison/EXPERIMENT_2A_SUMMARY.txt` - 详细报告
