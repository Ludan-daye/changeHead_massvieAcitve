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
| Top1峰值 | 3,540 | 60 | **-98%** |
| Dim 447 | 13.45 | 0.89 | -93% |
| Dim 138 | 14.69 | 0.95 | -94% |

## 结论

**✅ Attention Heads 产生 Massive Activation**

禁用后激活下降98%，证明BLOOM的MA由Attention产生，与Qwen（MLP产生）机制相反。

## 文件列表
- `baseline/results.json` - 基线测试结果
- `all_heads_disabled/results.json` - 禁用头部测试结果
- `comparison/exp1_top1_comparison.png` - 对比图
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告

---

## 补充实验: MA来源验证

### 方法
Hook关键层的Attention和MLP输出，对比最大激活值

### 结果
| 模块 | 输出Max |
|------|---------|
| Attention | 26.96875 |
| **MLP** | **92.5** |

### 结论
**MA来自MLP输出，Attention只提供触发输入。**
