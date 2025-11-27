# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型，记录激活值
2. **All Heads Disabled**: 将所有Attention Heads的输出置零
3. **对比**: 分析激活值变化

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 1,927 | 7,067 | **+266%** ⬆️ |
| Dim 458 | 1,888 | 7,044 | +273% |

## 结论

**❗ Attention Heads 抑制 Massive Activation**

禁用后激活反而上升266%，证明Qwen的Attention起**抑制**作用，MA由MLP产生。

与BLOOM、LLaMA、GPT-2机制相反！

## 文件列表
- `baseline/results.json` - 基线测试结果
- `all_heads_disabled/results.json` - 禁用头部测试结果
- `comparison/exp1_top1_comparison.png` - 对比图
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告
