# GPT-J-6B Massive Activation 实验总结

## 模型信息
- **模型**: GPT-J-6B (EleutherAI)
- **层数**: 28层
- **Attention Heads**: 16个/层
- **Hidden Size**: 4096
- **测试日期**: 2025-11-27

---

## 实验概览

| 实验 | 目的 | 状态 |
|------|------|------|
| Exp1 | 验证Attention是否产生MA | ✅ 完成 |
| Exp2 | 找出关键层 | ✅ 完成 |
| Exp4 | SVD分析 | ✅ 完成 |
| Exp4b | 对齐测试 | ✅ 完成 |

---

## 核心发现

### 1. Exp1: Attention作用测试
| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | ~4200 | ~200 | **-96%** |

**结论**: Attention几乎完全产生MA（极强生成型）

### 2. Exp2: 层贡献分析
| 关键层 | 激活值 | vs 其他层 |
|--------|--------|-----------|
| **Layer 0** | 1804.80 | **7.5x** |

**结论**: Layer 0是产生MA的绝对关键层

### 3. Exp4 & Exp4b: SVD与MA对齐
| Layer | σ₁/σ₂ | 余弦相似度 | 维度交集 |
|-------|-------|-----------|---------|
| **0** | 1.91 | **-0.69** | **8/10** |
| 27 | 1.81 | -0.06 | 5/10 |

**结论**: Layer 0 SVD与MA强对齐！

---

## 核心结论

### ✅ GPT-J是SVD对齐型生成模型

1. **极强生成型**: 禁用Attention后MA下降96%
2. **Layer 0关键**: 贡献是其他层的7.5倍
3. **SVD强对齐**: Layer 0 余弦相似度-0.69，维度交集8/10

### 与其他模型对比

| 模型 | 机制类型 | 关键层 | SVD对齐 |
|------|----------|--------|---------|
| **GPT-J** | **生成+SVD对齐** | **L0** | **0.69** |
| Qwen | 抑制+SVD对齐 | L3 | 0.99 |
| BLOOM | 生成型 | L28 | - |
| Falcon | 生成型 | L0 | 0.11 |
| Mistral | 混合型 | L0 | 0.38 |

---

## 文件结构

```
results/models/gptj_6b/
├── GPTJ_6B_SUMMARY.md
├── exp1/
│   ├── baseline/
│   ├── all_heads_disabled/
│   └── comparison/
├── exp2/
│   ├── layer_contribution.json
│   └── layer_contribution.png
└── exp4/
    └── mlp_svd.json
```
