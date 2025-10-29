# Experiment 5: Function Words Mapping in SVD Space

## 概述

**实验问题**: 无语义连接词在Layer 2的MLP权重矩阵W₂的**左右奇异向量空间**中如何映射？

**核心假设**:
1. **集中性假设**: 连接词的投影在较少的奇异向量上集中（低维性）
2. **左右不对称**: 连接词在左奇异空间(h₂侧)有明显对齐，但在右侧(输出侧)分散
3. **跨句子稳定**: 连接词在不同上下文中的投影高度稳定（固定表示）
4. **主方向对齐**: 连接词更强地对齐W₂的主奇异向量v₁

## 运行方式

### 基础命令
```bash
python exp5_function_words_svd_mapping.py --model gpt2 --layer_id 2 --nsamples 50 --savedir results/exp5_svd_mapping/
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | gpt2 | 模型选择: gpt2, llama, mistral |
| `--layer_id` | 2 | 分析的层索引（0-indexed，通常选2-5） |
| `--nsamples` | 50 | 文本序列数量 |
| `--savedir` | results/exp5_svd_mapping/ | 结果保存目录 |
| `--device` | cuda | 计算设备: cuda 或 cpu |

### 示例命令

```bash
# 高质量分析（推荐用于论文）
python exp5_function_words_svd_mapping.py --model gpt2 --layer_id 2 --nsamples 100 --savedir results/exp5_high_quality/

# 快速测试（低GPU内存）
python exp5_function_words_svd_mapping.py --model gpt2 --layer_id 2 --nsamples 20 --savedir results/exp5_quick_test/

# 多层分析脚本
for layer_id in 2 3 4 5; do
  python exp5_function_words_svd_mapping.py --model gpt2 --layer_id $layer_id --nsamples 50 --savedir results/exp5_layer${layer_id}/
done
```

## 四个核心分析

### Analysis 1: 方差集中性 (Concentration)

**目标**: 检验连接词的投影是否集中在少数几个奇异向量上

**输出**:
- `exp5_concentration_top5.png`: 条形图，显示前5个奇异向量解释的方差占比
- 高分(>0.5) = 低维性，说明投影高度集中
- 低分(<0.3) = 高维性，说明投影分散

**数学定义**:
```
concentration_top_k = Σ(top_k个var) / total_var
```

**预期结果**:
- ✅ 连接词（the, and, is）: >0.5（高集中）
- ❌ 内容词（dog, tree）: <0.3（分散）

---

### Analysis 2: 左右空间不对称性 (Left-Right Asymmetry)

**目标**: 检验信息在Linear1还是Linear2阶段确定

**输出**:
- `exp5_asymmetry_analysis.png`: 两个图表
  - 左图：Asymmetry Ratio (Left / Right) ——— >1表示左空间更集中
  - 右图：散点图展示左右浓度的对应关系

**数学定义**:
```
left_concentration = var(h₂ @ U) / total_var    [输入侧]
right_concentration = var(output @ Vt) / total_var  [输出侧]
asymmetry_ratio = left_concentration / right_concentration
```

**解释**:
- Ratio **>1** (红色): 连接词信息在Linear1（展开）阶段确定，Linear2不改变
- Ratio **<1** (蓝色): 连接词信息在Linear2（投影）阶段形成
- Ratio **≈1**: 两阶段平衡

**预期结果**:
```
函数词：asymmetry_ratio > 1.5  （强左侧对齐）
内容词：asymmetry_ratio < 1.0  （右侧驱动）
```

---

### Analysis 3: 跨句子稳定性 (Cross-Context Stability)

**目标**: 检验同一词在不同上下文中的表示是否稳定

**输出**:
- `exp5_stability_analysis.png`: 误差棒图，显示每个词的平均稳定性
- 高稳定(>0.8)：表示连接词有固定表示，不因上下文改变
- 低稳定(<0.5)：表示内容词表示高度依赖上下文

**数学定义**:
```
对同一词的n次出现，计算n(n-1)/2对的余弦相似度
stability = mean(cosine_similarity_pairs)
```

**预期结果**:
```
连接词（the, and）: stability > 0.85  （高度稳定）
内容词（dog, cat）: stability = 0.4-0.6  （上下文依赖）
```

---

### Analysis 4: 主奇异向量对齐 (Principal Direction Alignment)

**目标**: 检验连接词是否优先对齐W₂的最强奇异方向v₁

**输出**:
- `exp5_alignment_v1.png`: 条形图，显示每个词与v₁的余弦相似度
- 高对齐(>0.5)：词向量沿v₁方向投影强
- 低对齐(<0.1)：词向量垂直于v₁

**数学定义**:
```
v₁ = U[:, 0]  (W₂最强的左奇异向量)
alignment(word) = mean_token(cosine(h₂[token], v₁))
```

**预期结果**:
```
连接词与v₁强对齐 → 导致大激活（σ₁倍数）
内容词与v₁弱对齐 → 激活相对较小
```

---

## 输出文件说明

```
results/exp5_svd_mapping/
├── exp5_concentration_top5.png          # Analysis 1: 方差集中性
├── exp5_asymmetry_analysis.png          # Analysis 2: 左右不对称
├── exp5_stability_analysis.png          # Analysis 3: 跨句子稳定
├── exp5_alignment_v1.png                # Analysis 4: 主向量对齐
├── exp5_detailed_results.json           # 所有详细数据（JSON）
└── EXP5_SUMMARY.txt                     # 文字总结报告
```

### JSON结果结构
```json
{
  "timestamp": "2024-10-29T...",
  "concentration": {
    "the": {
      "concentration": {"top_1": 0.45, "top_3": 0.68, "top_5": 0.82},
      "top_singular_vectors": [0, 3, 7, 12, 18]
    },
    ...
  },
  "asymmetry": {
    "the": {
      "asymmetry_ratio": 1.85,
      "left_concentration_top3": 0.72,
      "right_concentration_top3": 0.39
    },
    ...
  },
  "stability": {
    "the": {
      "mean_stability": 0.87,
      "std_stability": 0.08,
      "n_occurrences": 45
    },
    ...
  },
  "alignment": {
    "the": {
      "mean_alignment_with_v1": 0.62,
      "std_alignment_with_v1": 0.15
    },
    ...
  }
}
```

## 关键指标解读

### 指标1: Concentration Score

```
score的含义：
  0.9-1.0   →  超高集中，仅依赖1-2个奇异向量
  0.7-0.9   →  高集中，依赖3-5个奇异向量
  0.5-0.7   →  中等集中，依赖5-10个奇异向量
  0.3-0.5   →  低集中，分散在多个方向
  0.0-0.3   →  极分散，全维度均匀分布
```

### 指标2: Asymmetry Ratio

```
ratio > 2.0   →  强左侧驱动（Linear1确定）
ratio > 1.5   →  中等左侧驱动
ratio ≈ 1.0   →  左右平衡
ratio < 0.7   →  强右侧驱动（Linear2确定）
```

### 指标3: Mean Stability

```
> 0.85      →  非常稳定（同义重复），连接词通常这样
0.7-0.85    →  稳定（同类上下文）
0.5-0.7     →  中等（多样上下文）
< 0.5       →  不稳定（高度上下文依赖），内容词通常这样
```

### 指标4: Alignment with v₁

```
> 0.7   →  强对齐，该词贡献大激活
0.5-0.7 →  中等对齐
0.3-0.5 →  弱对齐
< 0.3   →  基本垂直，激活较小
```

## 常见问题

**Q: 为什么要选layer_id=2？**
A: Layer 2是大激活爆炸的起点。实际应用中可以扫描所有层对比。

**Q: 样本数量多少合适？**
A:
- 快速测试: 20-30样本
- 标准分析: 50样本（推荐）
- 高质量论文: 100+样本

**Q: 什么是"连接词"？**
A: 根据语言学定义的无语义词汇：
- 冠词: the, a, an
- 介词: of, in, on, at, to, ...
- 连接词: and, or, but, if, ...
- 代词: it, they, he, ...
- 助动词: is, are, was, ...

**Q: 如何修改词表？**
A: 编辑脚本中的 `FUNCTION_WORDS` 字典，示例：
```python
FUNCTION_WORDS = {
    '自定义词1', '自定义词2', ...
}
```

## 实验建议

### 对比实验
```bash
# 比较不同层的模式
for layer in 2 3 4 5 6; do
  python exp5_function_words_svd_mapping.py --layer_id $layer --nsamples 50 --savedir results/exp5_layer$layer/
done
```

### 对比不同模型
```bash
# 比较GPT-2和LLaMA
python exp5_function_words_svd_mapping.py --model gpt2 --layer_id 2 --savedir results/exp5_gpt2/
python exp5_function_words_svd_mapping.py --model llama --layer_id 2 --savedir results/exp5_llama/
```

### 分析特定词类
编辑脚本，创建子集：
```python
# 只分析冠词
FUNCTION_WORDS = {'the', 'a', 'an'}

# 或只分析介词
FUNCTION_WORDS = {'of', 'in', 'on', 'at', 'to', 'for', ...}
```

## 预期发现

| 结果类型 | 连接词表现 | 内容词表现 |
|---------|----------|---------|
| **集中性** | 高(>0.7) | 低(<0.3) |
| **左右不对称** | ratio>1.5 | ratio<0.8 |
| **跨句稳定性** | >0.85 | <0.6 |
| **v₁对齐** | 强(>0.5) | 弱(<0.3) |

**理论意义**: 如果找到这些模式，表明：
- 连接词有**固定的语义角色**，表示稳定
- 大激活是连接词**集体对齐**的结果
- MLP**专门优化**了连接词处理（通过W₂的SVD结构）

## 相关阅读

- Exp 3: SVD几何分析（奠基工作）
- Exp 4: 函数词与v₁关系（对齐分析）
- Exp 2C: MLP内部追踪（知道爆炸发生在Linear2）
