# 实验2: 层贡献分析

## 实验目的
找出LLaMA-2-13B中哪一层产生Massive Activation

## 实验方法
1. 禁用所有Attention Heads
2. 逐层恢复单层的Attention
3. 测量恢复后的激活值

## 核心结果

关键层: **Layer 3**
- Layer 3恢复后激活显著回升
- 前3层对MA贡献最大

## 结论

**Layer 3 是 LLaMA-2-13B 的关键层**

与GPT-2的Layer 2类似，都是早期层产生MA。

## 文件列表
- `layer_contribution.json` - 各层贡献数据
- `layer_contribution.png` - 可视化图表
