# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证LLaMA-2-7B-Chat的Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型
2. **All Heads Disabled**: 将所有Attention Heads输出置零
3. **对比**: 分析激活值变化

## 核心结果

见`comparison/EXPERIMENT_1_SUMMARY.txt`

## 结论

LLaMA-2-7B-Chat与LLaMA-2-13B机制类似。

## 文件列表
- `baseline/` - 基线测试结果
- `all_heads_disabled/` - 禁用头部测试结果
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告
