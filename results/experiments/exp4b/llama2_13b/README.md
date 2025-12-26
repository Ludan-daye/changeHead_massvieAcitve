# 实验4b: Layer 3 Attention SVD分析

## 实验目的
分析Layer 3 Attention输出投影层的SVD结构

## 实验方法
1. 提取`self_attn.o_proj`权重矩阵
2. 进行SVD分解
3. 分析主导奇异方向

## 核心结果

见`LAYER3_ATTENTION_SVD_SUMMARY.txt`

## 结论

分析Attention层的SVD结构，确认是否有主导方向。

## 文件列表
- `LAYER3_ATTENTION_SVD_SUMMARY.txt` - 详细报告
- 相关可视化图表
