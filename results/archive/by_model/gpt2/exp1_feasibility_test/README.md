# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型，记录激活值
2. **All Heads Disabled**: 将所有144个Attention Heads的输出置零
3. **对比**: 分析激活值变化

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 983 (Layer 16) | 393 | **-60%** |
| Dim 447 | 12.89 | 9.43 | -27% |
| Dim 138 | 12.08 | 9.97 | -17% |

## 结论

**⚠️ Attention Heads 参与产生 Massive Activation**

禁用后激活下降60%，说明Attention对MA有贡献，但不是唯一来源。

## 文件列表
- `baseline/` - 基线测试结果
- `all_heads_disabled/` - 禁用头部测试结果  
- `comparison/exp1_top1_comparison.png` - 对比图
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告
