# 实验2: 层贡献分析

## 实验目的
找出OPT-6.7B中哪一层产生/抑制Massive Activation

## 实验方法
1. 禁用所有Attention Heads
2. 逐层恢复单层的Attention
3. 测量恢复后的激活值

## 核心结果

见`EXPERIMENT_2_SUMMARY.txt`

## 结论

分析各层Attention对MA的抑制效果。

## 文件列表
- `EXPERIMENT_2_SUMMARY.txt` - 详细报告
- 相关数据文件
