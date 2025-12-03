# RQ1: 巨量激活生成分析 - gptj_6b

## 研究问题

**MA是如何生成的？Attention heads的贡献是什么？**

## 实验设计

### Exp1: Attention贡献测试
- **Baseline**: 正常运行模型
- **All Heads Disabled**: 禁用所有attention heads
- **对比**: 激活值变化

### Exp2: 层级贡献分析
- 统计每一层的最大激活值
- 识别关键层（MA最大的层）

## 实验结果

### Exp1结论




### Exp2关键层

详见: [exp2_layer_contribution/data/](exp2_layer_contribution/data/)

## 分析

根据Exp1结果，该模型的MA生成特征为：
- Attention heads在MA生成中的作用
- 关键层的分布模式

## 相关数据

- [`exp1_attention_contribution/data/`](exp1_attention_contribution/data/)
- [`exp2_layer_contribution/data/`](exp2_layer_contribution/data/)
