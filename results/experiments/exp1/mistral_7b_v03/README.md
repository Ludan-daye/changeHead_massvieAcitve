# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证Mistral-7B-v0.3的Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型，记录各层激活值
2. **All Heads Disabled**: 将所有32个Attention Heads输出置零
3. **对比**: 分析激活值变化幅度

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 314.20 (Layer 25) | 259.00 | **-18%** |
| Dim 447 | 1.53 | 1.42 | -7% |
| Dim 138 | 1.62 | 1.06 | -35% |

### 特殊发现
| Layer | Baseline | Disabled | 变化 |
|-------|----------|----------|------|
| Layer 0 | 1.37 | 0.12 | -91% |
| Layer 1-30 | ~312 | ~254 | -18% |
| **Layer 31** | 49.66 | 101.06 | **+104%** ⬆️ |

## 结论

**⚠️ Mistral是混合机制模型**

- **大部分层(0-30)**: Attention产生MA，禁用后下降18%
- **最后一层(31)**: Attention抑制MA，禁用后反而上升104%
- 与纯生成型(BLOOM/LLaMA)和纯抑制型(Qwen/OPT)都不同

## 文件列表
- `baseline/results.json` - 基线测试数据
- `all_heads_disabled/results.json` - 禁用头部测试数据
- `comparison/exp1_top1_comparison.png` - 对比图
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告

---

## 补充实验: MA来源验证

### 方法
Hook关键层的Attention和MLP输出，对比最大激活值

### 结果
| 模块 | 输出Max |
|------|---------|
| Attention | 0.08441162109375 |
| **MLP** | **1.1689453125** |

### 结论
**MA来自MLP输出，Attention只提供触发输入。**
