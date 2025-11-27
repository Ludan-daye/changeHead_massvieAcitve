# Qwen2.5-7B Massive Activation 分析报告

**生成时间**: 2025-11-27

## 模型信息

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-7B |
| 层数 | 28 |
| Hidden Size | 3584 |
| Attention Heads | 28 |

---

## 实验1: 头抑制对比 (Feasibility Test)

**目录**: `exp1/`

### 核心发现

| 指标 | Baseline | Heads Disabled | 变化 |
|------|----------|----------------|------|
| Top1激活峰值 | 11,885 (Layer 16) | 43,520 | **+266%** |
| Dim 447 | 21.59 | 70.94 | +229% |
| Dim 138 | 21.52 | 164.58 | +665% |

### 结论

**⚠️ Attention Heads 起抑制作用，而非产生作用！**

禁用Attention后massive activation**增加**而非减少，说明：
- MLP层是massive activation的**来源**
- Attention Heads 起**抑制作用**
- 机制与OPT-6.7B类似，与LLaMA-2/GPT-2相反

### 生成文件
- `baseline/results.json` - 基线结果
- `all_heads_disabled/results.json` - 禁用头结果
- `comparison/exp1_top1_comparison.png` - 对比图
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告

---

## 实验2: 层贡献分析

**目录**: `exp2/`

### 核心发现

| 层 | 恢复后Top1激活 | 分析 |
|----|---------------|------|
| **Layer 0** | 24,784 | **最强抑制层** |
| Layer 1 | 31,869 | 次强抑制 |
| Layer 3 | 39,251 | 中等 |
| Layer 2 | 42,413 | 最弱抑制 |

### 结论

**Layer 0 的 Attention Heads 是主要抑制源！**

- 恢复Layer 0后激活从43520降到24784（-43%）
- Layer 0-1 形成"早期抑制机制"

### 生成文件
- `layer_contribution.json` - 各层贡献数据
- `layer_contribution.png` - 可视化图

---

## 实验4: MLP SVD分析

**目录**: `exp4/`

### 核心发现

| Layer | σ₁/σ₂ | 分析 |
|-------|-------|------|
| **Layer 3** | **2.64** | 主导奇异方向 |
| Layer 1 | 1.69 | 次强 |
| Layer 0 | 1.23 | - |
| Layer 2 | 1.19 | - |
| Layer 26 | 1.15 | - |

### 结论

Layer 3 MLP存在主导奇异方向（σ₁/σ₂ = 2.64）

### 生成文件
- `svd_analysis.json` - SVD分析数据
- `svd_analysis.png` - 奇异值可视化

---

## 实验4b: SVD对齐验证

**目录**: `exp4b/`

### 核心发现

| 指标 | 值 |
|------|-----|
| **与u1对齐** | **0.994** |
| 与u2对齐 | 0.016 |
| 与u3对齐 | 0.049 |

### Massive Activation维度

| 维度 | 平均激活值 | u1分量 |
|------|-----------|--------|
| Dim 458 | 8855 | 0.77 |
| Dim 2570 | 5610 | 0.54 |

### 结论

**✅ 证实：Layer 3的激活与主导奇异向量u1高度对齐（cos=0.994）**

Massive activation由MLP权重矩阵的几何结构决定！

### 生成文件
- `layer3_alignment.json` - 对齐分析数据
- `layer3_alignment.png` - 对齐可视化

---

## 总结机制

```
Qwen2.5-7B Massive Activation 机制:

1. 来源: Layer 3 MLP down_proj 的主导奇异方向
   - σ₁ = 17.03 >> σ₂ = 6.45
   - 激活与u1对齐: cos = 0.994

2. 主要维度: Dim 458, Dim 2570
   - 对应u1的主导分量

3. 抑制: Layer 0-1 的 Attention Heads
   - 恢复Layer 0可降低43%激活

4. 与其他模型对比:
   ┌─────────────┬─────────────┬──────────────┐
   │ 模型        │ MA来源      │ Attention作用│
   ├─────────────┼─────────────┼──────────────┤
   │ Qwen2.5-7B  │ MLP Layer 3 │ 抑制         │
   │ OPT-6.7B    │ MLP Layer 0 │ 抑制         │
   │ LLaMA-2-13B │ Attention   │ 产生         │
   │ GPT-2       │ Attention   │ 产生         │
   └─────────────┴─────────────┴──────────────┘
```

---

## 文件结构

```
results/models/qwen2.5_7b/
├── QWEN2.5_7B_SUMMARY.md      <- 本文件
├── exp1/                       <- 实验1: 头抑制对比
│   ├── baseline/
│   │   └── results.json
│   ├── all_heads_disabled/
│   │   └── results.json
│   └── comparison/
│       ├── exp1_top1_comparison.png
│       ├── exp1_percentage_change_heatmap.png
│       ├── exp1_layerwise_breakdown.png
│       ├── exp1_critical_dimensions.png
│       └── EXPERIMENT_1_SUMMARY.txt
├── exp2/                       <- 实验2: 层贡献分析
│   ├── layer_contribution.json
│   └── layer_contribution.png
├── exp4/                       <- 实验4: MLP SVD分析
│   ├── svd_analysis.json
│   └── svd_analysis.png
└── exp4b/                      <- 实验4b: SVD对齐验证
    ├── layer3_alignment.json
    └── layer3_alignment.png
```
