# 🔬 Massive Activations 多模型实验报告

> 验证大规模激活(Massive Activations)现象在不同Transformer架构中的普遍性及产生机制

---

## 📊 实验概览

| 模型 | 参数量 | 架构 | 实验完成度 | 核心发现 |
|------|--------|------|------------|----------|
| GPT-2 | 124M | 标准GPT | ✅ 100% | MLP SVD对齐 (R²=0.998) |
| LLaMA-2-13B | 13B | LLaMA架构 | ✅ 100% | 与GPT-2机制一致 (R²=0.988) |
| LLaMA-2-7B-Chat | 7B | LLaMA架构 | ✅ 70% | 符合LLaMA规律 |
| OPT-6.7B | 6.7B | OPT架构 | ✅ 100% | **反向机制**: Attention抑制激活 |

---

## 🎯 核心结论

### 发现1: SVD几何对齐是大规模激活的数学本质

```
MLP.down_proj = U Σ V^T

当输入与主奇异向量v₁对齐时:
output ≈ (input · v₁) × σ₁ × u₁ = 大数值 × 固定方向

→ 产生大规模激活
```

**验证**:
- GPT-2: R² = **0.9979** 
- LLaMA-2-13B: R² = **0.9880**

### 发现2: 存在两种机制模式

| 模式 | 代表模型 | Attention作用 | MLP作用 |
|------|----------|---------------|---------|
| **模式A** | GPT-2, LLaMA | 产生/放大激活 | 核心放大器 |
| **模式B** | OPT-6.7B | **抑制**激活 | Layer0为火源 |

---

## 📁 结果目录结构

```
results/models/
├── gpt2/                          # GPT-2 (124M)
│   ├── exp1_feasibility_test/     # 注意力头禁用测试
│   ├── exp2a_mlp_feasibility_test/# MLP禁用测试
│   ├── exp2c_mlp_internal/        # MLP内部分析
│   └── exp3_svd_alignment/        # SVD对齐分析 ⭐
│
├── llama2_13b/                    # LLaMA-2-13B
│   ├── exp1_llama2_13b/           # 注意力头禁用测试
│   ├── exp2_llama2_13b/           # 单层恢复分析
│   ├── exp4b_layer3_attention/    # L3 Attention SVD
│   ├── exp4c_layer3_head_output/  # L3 Head输出分析
│   └── exp4d_layer3_mlp/          # L3 MLP SVD ⭐
│
├── llama2_7b_chat/                # LLaMA-2-7B-Chat
│   └── exp1_llama2_7b_chat/       # 注意力头禁用测试
│
├── opt_6.7b/                      # OPT-6.7B
│   ├── exp1_opt_6.7b/             # 注意力头禁用 (异常!)
│   ├── exp2_opt_6.7b/             # 单层恢复分析
│   ├── exp3_opt_fire_test/        # MLP放火测试 ⭐
│   └── exp4_opt_svd/              # MLP SVD分析
│
└── misc/                          # 其他分析
    ├── head_analysis/             # 注意力头重要性
    ├── head_pruning_massive/      # 头剪枝影响
    └── exp5_multi_model/          # 多模型对比
```

---

## 📈 各模型详细结果

### 1. GPT-2 (124M)

**实验路径**: `results/models/gpt2/`

#### 关键发现
- **激活跳变点**: Layer 2 (从<1000跳升至~2500)
- **稳定激活层**: Layer 3-11 (~3000)
- **SVD R²**: 0.9979 (几乎完美线性关系)

#### 可视化
| 图片 | 说明 |
|------|------|
| `exp1_*/exp1_top1_comparison.png` | 禁用前后Top1激活对比 |
| `exp3_*/exp3_projection_regression.png` | SVD投影回归分析 |
| `exp3_*/exp3_singular_values.png` | 奇异值分布 |

#### 数据统计
```
Layer    Top1     Median    比率
0        101      0.60      168×
2        2474     0.84      2945×  ← 跳变点
10       3019     2.61      1156×
```

---

### 2. LLaMA-2-13B

**实验路径**: `results/models/llama2_13b/`

#### 关键发现
- **激活起始层**: Layer 3
- **禁用Attention后**: 激活下降 **94-98%**
- **MLP SVD R²**: 0.9880

#### 实验序列
1. **Exp1**: 验证Attention参与大规模激活 ✅
2. **Exp4B**: Attention权重SVD对齐 ❌ (不对齐)
3. **Exp4C**: Head对function words特化 ❌ (无差异)
4. **Exp4D**: MLP SVD对齐 ✅✅✅ (R²=0.988)

#### 核心结论
> Attention是辅助机制，MLP是核心机制。两者与GPT-2完全一致。

---

### 3. OPT-6.7B (异常行为!)

**实验路径**: `results/models/opt_6.7b/`

#### 🔥 异常发现
- 禁用Attention后激活**上升250%**（与其他模型相反！）
- Attention起**抑制**作用
- MLP Layer 0是"纵火犯"

#### 机制重构
```
Layer 0 MLP  →  输出幅度1147 (火源🔥)
    ↓
Layer 1-30 Attention  →  持续抑制 (消防员🧯)
    ↓
Layer 31 MLP  →  再次爆发130.4
```

#### 抑制效率排名
| 层 | 恢复率 | 作用 |
|----|--------|------|
| Layer 3 | 19.1% | 早期抑制 |
| Layer 31 | 18.9% | 末期抑制 |
| Layer 29 | 14.2% | 最弱点 |

---

### 4. LLaMA-2-7B-Chat

**实验路径**: `results/models/llama2_7b_chat/`

#### 发现
- 与LLaMA-2-13B规律一致
- 激活起始于早期层
- 禁用Attention后激活显著下降

---

## 📝 实验日志

| 日志文件 | 内容 |
|----------|------|
| `exp1_llama2_13b_*.log` | LLaMA-2-13B Exp1运行记录 |
| `exp1_opt_6.7b.log` | OPT-6.7B Exp1运行记录 |
| `exp2_opt_6.7b.log` | OPT-6.7B 单层恢复日志 |
| `exp4_svd_alignment.log` | SVD对齐分析详细日志 |
| `exp4d_mlp_svd.log` | MLP SVD分析日志 |

---

## 🔬 实验方法

### Exp1: 可行性测试
```python
# 禁用所有Attention Head
for layer in model.layers:
    layer.self_attn.forward = lambda x: (x, None, None)

# 对比激活变化
baseline_activation vs disabled_activation
```

### Exp2: 单层恢复
```python
# 禁用所有Head后，逐层恢复单个层
for restore_layer in range(num_layers):
    enable_only(restore_layer)
    measure_recovery_rate()
```

### Exp3: MLP放火测试
```python
# 测量各层MLP输出幅度
for layer in model.layers:
    mlp_output_norm = layer.mlp(x).norm()
```

### Exp4: SVD对齐分析
```python
# 对MLP权重做SVD
U, S, Vt = torch.linalg.svd(mlp.down_proj.weight)

# 计算对齐度
alignment = cosine_similarity(input, Vt[0])

# 回归分析
R² = linear_regression(alignment, activation_magnitude)
```

---

## 📚 相关文件

### 分析报告
- `OPT_6.7B_MECHANISM_ANALYSIS.md` - OPT机制详解
- `LLAMA2_13B_FINAL_REPORT.md` - LLaMA完整报告
- `RESULTS_SUMMARY.md` - GPT-2结果总结
- `EXPERIMENT_SUMMARY.md` - 整体实验方法论

### 实验脚本
- `exp1_feasibility_test.py` - 可行性测试
- `exp2_single_layer_restoration.py` - 单层恢复
- `exp3_mlp_fire_test.py` - MLP放火测试
- `exp4_opt_mlp_svd.py` - OPT SVD分析
- `exp4d_mlp_svd_alignment.py` - LLaMA MLP SVD

---

## 🚀 待测试模型

| 模型 | 架构特性 | 预期 |
|------|----------|------|
| Qwen2.5-7B | GQA | 模式A |
| DeepSeek-V2-Lite | MoE | 待验证 |
| Mistral-7B | Sliding Window | 模式A |
| Falcon-7B | Multi-Query + ALiBi | 待验证 |

---

## 📖 参考

- **论文**: [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)
- **原始代码**: [locuslab/massive-activations](https://github.com/locuslab/massive-activations)

---

**最后更新**: 2025-11-26
