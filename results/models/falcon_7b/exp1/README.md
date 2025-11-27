# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证Falcon-7B的Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型，记录各层激活值
2. **All Heads Disabled**: 将所有32层×71个Attention Heads输出置零
3. **对比**: 分析激活值变化幅度

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 1827 (Layer 23) | 1437 | **-21%** |
| Dim 447 | 10.51 | 4.73 | -55% |
| Dim 138 | 8.20 | 6.09 | -26% |

### 层级分析
- Layer 3: 变化最大（-40%）
- Layer 4-30: 稳定下降约21%
- Layer 31: 下降15%

## 结论

**✅ Attention Heads产生MA**

- 禁用所有Attention后，激活下降21%
- 说明Attention贡献了MA的生成
- 与BLOOM、LLaMA、GPT-2机制一致

## 文件列表
- `baseline/results.json` - 基线测试数据
- `all_heads_disabled/results.json` - 禁用头部测试数据
- `comparison/exp1_top1_comparison.png` - 对比图
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告
