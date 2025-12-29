# Falcon-7B Massive Activation 实验总结

## 模型信息
- **模型**: Falcon-7B
- **层数**: 32层
- **Attention Heads**: 71个/层
- **Hidden Size**: 4544
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
| Top1峰值 | 1827 | 1437 | **-21%** |

**结论**: Attention产生MA（禁用后下降21%）

### 2. Exp2: 层贡献分析
| 关键层 | 激活值 | vs 其他层 |
|--------|--------|-----------|
| **Layer 0** | 1300 | **1.10x** |

**结论**: Layer 0是产生MA的关键层

### 3. Exp4: SVD分析
| Layer | σ₁/σ₂ | u1最大维度 |
|-------|-------|-----------|
| **0** | **2.86** | Dim 2138 |
| 31 | 1.99 | Dim 3080 |

**结论**: Layer 0有最强的主导奇异值

### 4. Exp4b: SVD-MA对齐
| Layer | 余弦相似度 | 维度交集 |
|-------|-----------|---------|
| **0** | 0.11 | 5/10 |
| 31 | -0.14 | 4/10 |

**结论**: 弱整体对齐，中等维度交集

---

## 核心结论

### ✅ Falcon是生成型模型

1. **Layer 0**: 产生MA的关键层
2. **Attention机制**: 贡献MA生成（禁用后下降21%）
3. **与Mistral相似**: 都在Layer 0产生MA

### 与其他模型对比

| 模型 | 机制类型 | 关键层 | 禁用后变化 |
|------|----------|--------|-----------|
| BLOOM | 生成型 | L28 | -98% |
| LLaMA | 生成型 | L3 | -80% |
| **Falcon** | **生成型** | **L0** | **-21%** |
| Mistral | 混合型 | L0 | -18% |
| Qwen | 抑制型 | L3 | +266% |
| OPT | 抑制型 | L0-1 | +250% |

---

## 文件结构

```
results/models/falcon_7b/
├── FALCON_7B_SUMMARY.md       # 本总结
├── exp1/                       # 实验1: 头抑制测试
│   ├── README.md
│   ├── baseline/
│   ├── all_heads_disabled/
│   └── comparison/
├── exp2/                       # 实验2: 层贡献分析
│   ├── README.md
│   ├── layer_contribution.json
│   └── layer_contribution.png
├── exp4/                       # 实验4: SVD分析
│   ├── README.md
│   └── mlp_svd.json
└── exp4b/                      # 实验4b: SVD-MA对齐
    ├── README.md
    └── svd_ma_alignment.json
```

---

## 总结

Falcon-7B与Mistral-7B机制相似:
- 都在Layer 0产生MA
- Attention贡献约20%
- SVD与MA弱整体对齐，存在维度交集
