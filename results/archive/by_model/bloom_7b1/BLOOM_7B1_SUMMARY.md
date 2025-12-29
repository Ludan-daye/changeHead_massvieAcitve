# BLOOM-7B1 Massive Activation 分析报告

**生成时间**: 2025-11-27

## 模型信息

| 参数 | 值 |
|------|-----|
| 模型 | BLOOM-7B1 |
| 层数 | 30 |
| Hidden Size | 4096 |
| Attention Heads | 32 |
| 位置编码 | ALiBi |

---

## 实验1: 头抑制对比 (Feasibility Test)

**目录**: `exp1/`

### 核心发现

| 指标 | Baseline | Heads Disabled | 变化 |
|------|----------|----------------|------|
| Top1激活峰值 | 3,540 (Layer 12) | 60 | **-98%** ⬇️ |
| Dim 447 | 13.45 | 0.89 | -93% |
| Dim 138 | 14.69 | 0.95 | -94% |

### 关键层变化

| Layer | Baseline | Disabled | 变化 |
|-------|----------|----------|------|
| Layer 7 | 2,730 | 4 | -99.85% |
| Layer 11-12 | ~3,540 | ~2 | -99.96% |
| Layer 28 | 383 | 40 | -89% |

### 结论

**✅ Attention Heads 产生 Massive Activation！**

禁用Attention后激活下降98%，说明：
- Attention Heads 是MA的**来源**
- 机制与LLaMA-2、GPT-2类似
- 与Qwen（MLP产生）机制相反

### 生成文件
- `baseline/results.json`
- `all_heads_disabled/results.json`
- `comparison/exp1_top1_comparison.png`
- `comparison/EXPERIMENT_1_SUMMARY.txt`

---

## 实验2: 层贡献分析

**目录**: `exp2/`

### 核心发现

| 层 | 恢复后Top1激活 | 分析 |
|----|---------------|------|
| **Layer 28** | **1,125** | **唯一产生MA的层！** |
| 其他所有层 | ~40 | 几乎无贡献 |

### 结论

**Layer 28 的 Attention Heads 是 BLOOM 唯一的 MA 来源！**

- 只有恢复Layer 28才能产生显著激活
- 其他29层的Attention对MA几乎无贡献
- 机制非常集中，与Qwen（多层参与）不同

### 生成文件
- `layer_contribution.json`
- `layer_contribution.png`

---

## 实验4: Attention SVD分析

**目录**: `exp4/`

### 核心发现

| Layer | σ₁ | σ₂ | σ₁/σ₂ | 分析 |
|-------|-----|-----|-------|------|
| Layer 0 | 10.05 | 3.84 | **2.62** | 主导方向 |
| Layer 7 | 6.82 | 3.70 | 1.85 | - |
| Layer 12 | 2.97 | 2.89 | 1.03 | 无主导 |
| **Layer 28** | 18.48 | 15.70 | **1.18** | **无主导** |
| Layer 29 | 22.84 | 10.26 | 2.23 | 主导方向 |

### 结论

**Layer 28 没有显著的SVD主导方向！**

产生MA的Layer 28反而没有主导奇异方向，说明BLOOM的机制不是简单的SVD对齐。

### 生成文件
- `attention_svd.json`
- `attention_svd.png`

---

## 实验4b: SVD对齐验证

**目录**: `exp4b/`

### 核心发现

| 向量 | 奇异值 | 与激活对齐度 | 分析 |
|------|--------|-------------|------|
| u1 | 18.48 | **0.050** | ❌ 无对齐 |
| u2 | 15.70 | **0.070** | ❌ 无对齐 |
| u3 | 11.98 | **0.433** | ⚠️ 中等对齐 |

### 结论

**❌ BLOOM的MA与SVD主导方向无对齐！**

- 与u1对齐仅0.05（Qwen是0.994）
- 反而与u3有中等对齐
- 证实BLOOM的机制**不是**SVD主导方向

### 可能的机制
1. **多头协作效应** - 32个头共同产生
2. **ALiBi位置编码** - BLOOM特有
3. **累积残差** - 前层信息在Layer 28累积

### 生成文件
- `layer28_alignment.json`
- `layer28_alignment.png`

---

## 总结机制

```
BLOOM-7B1 Massive Activation 机制:

1. 来源: Layer 28 Attention Heads (唯一)
   - 禁用后激活下降98%
   - 只有Layer 28能恢复激活

2. 非SVD机制:
   - σ₁/σ₂ = 1.18 (无主导方向)
   - 与u1对齐仅0.05

3. 可能机制:
   - 多头协作
   - ALiBi位置编码交互
   - 累积残差效应

4. 与Qwen对比:
   ┌────────────┬─────────────┬─────────────┬──────────┐
   │ 模型       │ MA来源      │ 关键层      │ 机制     │
   ├────────────┼─────────────┼─────────────┼──────────┤
   │ BLOOM-7B1  │ Attention   │ Layer 28    │ 非SVD    │
   │ Qwen2.5-7B │ MLP         │ Layer 3     │ SVD对齐  │
   └────────────┴─────────────┴─────────────┴──────────┘
```

---

## 文件结构

```
results/models/bloom_7b1/
├── BLOOM_7B1_SUMMARY.md       <- 本文件
├── exp1/                       <- 实验1: 头抑制对比
│   ├── baseline/results.json
│   ├── all_heads_disabled/results.json
│   └── comparison/
│       ├── exp1_top1_comparison.png
│       └── EXPERIMENT_1_SUMMARY.txt
├── exp2/                       <- 实验2: 层贡献分析
│   ├── layer_contribution.json
│   └── layer_contribution.png
├── exp4/                       <- 实验4: Attention SVD分析
│   ├── attention_svd.json
│   └── attention_svd.png
└── exp4b/                      <- 实验4b: SVD对齐验证
    ├── layer28_alignment.json
    └── layer28_alignment.png
```
