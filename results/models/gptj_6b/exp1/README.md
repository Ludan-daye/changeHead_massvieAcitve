# GPT-J-6B Experiment 1: Feasibility Test

## 实验目的
验证Attention heads是否产生Massive Activation (MA)

## 方法
1. **Baseline**: 正常运行模型
2. **All Disabled**: 禁用所有16个attention heads
3. 对比激活变化

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | ~4200 | ~200 | **-96%** |
| 各层平均 | 4000+ | 150-250 | -95% |

### 各层变化
- Layer 0-25: 下降 **92-96%**
- Layer 26: 下降 **85%**
- Layer 27: 下降 **40%**

## 结论

### ⚠️ ATTENTION HEADS GENERATE MASSIVE ACTIVATIONS

GPT-J是**极强生成型**模型：
1. 禁用Attention后MA几乎完全消失（-96%）
2. 这是所有测试模型中最强的生成效应
3. Attention是MA的主要来源

## 下一步
→ Experiment 2: 识别关键层

---

## 补充实验: MA来源验证

### 方法
Hook关键层的Attention和MLP输出，对比最大激活值

### 结果
| 模块 | 输出Max |
|------|---------|
| Attention | 3.607421875 |
| **MLP** | **30.328125** |

### 结论
**MA来自MLP输出，Attention只提供触发输入。**
