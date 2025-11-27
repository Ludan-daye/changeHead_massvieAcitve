# Mistral-7B-v0.3 Massive Activation 实验总结

## 模型信息
- **模型**: Mistral-7B-v0.3
- **层数**: 32层
- **Attention Heads**: 32个/层
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
| Top1峰值 | 314 | 259 | **-18%** |

**特殊发现**: Layer 31禁用后激活反而上升104%

### 2. Exp2: 层贡献分析
| 关键层 | 激活值 | vs 其他层 |
|--------|--------|-----------|
| **Layer 0** | 173.75 | **1.66x** |

### 3. Exp4: SVD分析
| Layer | σ₁ | σ₁/σ₂ | 主导方向 |
|-------|-----|-------|---------|
| **Layer 0** | 0.87 | **1.10** | ❌ 无 |
| Layer 31 | 2.96 | **2.19** | ✅ 有 |

### 4. Exp4b: 方向对齐测试（更新）
| Layer | 与u1对齐 | 结论 |
|-------|---------|------|
| Layer 0 | **-0.38** | ⚠️ 中等对齐 |
| Layer 31 | **0.29** | ⚠️ 中等对齐 |

**关键发现**: Layer 31 Dim 3901同时是SVD主导(u1=-0.81)和MA主导(act=11.9)

---

## 核心结论

### ⚠️ Mistral是混合机制模型

1. **Layer 0**: 产生MA的关键层（最早）
2. **Layer 1-30**: Attention产生MA（禁用后下降18%）
3. **Layer 31**: Attention抑制MA（禁用后上升104%）

### 与其他模型对比

| 模型 | 机制类型 | 关键层 | 禁用后变化 |
|------|----------|--------|-----------|
| BLOOM | 生成型 | L28 | -98% |
| LLaMA | 生成型 | L3 | -80% |
| GPT-2 | 生成型 | L2 | -60% |
| **Mistral** | **混合型** | **L0** | **-18%** |
| Qwen | 抑制型 | L3 | +266% |
| OPT | 抑制型 | L0-1 | +250% |

---

## 文件结构

```
results/models/mistral_7b_v03/
├── MISTRAL_7B_SUMMARY.md      # 本总结
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
│   ├── attention_svd.json
│   ├── attention_svd_summary.json
│   └── attention_svd.png
└── exp4b/                      # 实验4b: 对齐测试
    ├── README.md
    └── layer0_alignment.json
```

---

## 机制总结

### Mistral的MA机制特点

1. **中等SVD对齐**: Layer 0的cos=-0.38，Layer 31的cos=0.29
2. **维度级强对齐**: Layer 31 Dim 3901同时是SVD主导和MA主导
3. **混合行为**: 大部分层生成MA，末层抑制MA
4. **最早的关键层**: Layer 0是所有测试模型中最早产生MA的层

### 与其他模型对比

| 特性 | Qwen | BLOOM | Mistral |
|------|------|-------|---------|
| 关键层 | L3 | L28 | **L0** |
| σ₁/σ₂ | 14.5 | 1.18 | **1.10** |
| 对齐度 | 0.99 | 0.05 | **-0.38** |
| 机制 | 强SVD对齐 | 弱对齐 | **中等对齐** |
