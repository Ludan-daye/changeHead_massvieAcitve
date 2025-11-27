# 实验4d: Layer 3 MLP SVD分析

## 实验目的
分析Layer 3 MLP down_proj的SVD结构

## 实验方法
1. 提取`mlp.down_proj`权重矩阵
2. 进行SVD分解
3. 计算激活与主导方向的对齐

## 核心结果

见`LAYER3_MLP_SVD_SUMMARY.txt`

## 结论

LLaMA-2-13B与GPT-2机制相同：MLP down_proj的主导奇异方向产生MA。

## 文件列表
- `LAYER3_MLP_SVD_SUMMARY.txt` - 详细报告
- 相关可视化图表
