# 实验2: 层贡献分析

## 实验目的
找出哪一层的Attention Heads产生Massive Activation

## 实验方法
1. 禁用所有Attention Heads
2. 逐层恢复单层的Attention
3. 测量恢复后的激活值
4. 找出贡献最大的层

## 核心结果

| 层 | 恢复后激活 | 分析 |
|----|-----------|------|
| **Layer 28** | **1,125** | 唯一有效层 |
| 其他所有层 | ~40 | 无贡献 |

## 结论

**Layer 28 是唯一产生MA的层！**

- 只有Layer 28能恢复激活（1125 vs 40）
- 其他29层几乎无贡献
- 机制极度集中（与Qwen多层参与不同）

## 文件列表
- `layer_contribution.json` - 各层贡献数据
- `layer_contribution.png` - 可视化图表
