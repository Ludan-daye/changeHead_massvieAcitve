# 归因实验框架 (Attribution Experiments Framework)

## 概览

本框架包含三个创新的归因实验（Exp5/7/8），用于精确量化MLP SVD分解中各组件对Massive Activation (MA)的贡献。

### 实验体系

```
完整实验流程:
├── Exp4: SVD结构分析 (已完成)
│   └── 输出: σ₁/σ₂比值，判断是否有主导方向
├── 旧Exp3: SVD对齐观察分析 (已完成)
│   └── 输出: R²相关系数，测试h₂·v₁ → MA
├── 新Exp3: U矩阵消融因果分析 (已完成)
│   └── 输出: 移除U导致的MA变化百分比
├── Exp6: V矩阵消融因果分析 (已完成)
│   └── 输出: 移除V导致的MA变化百分比
└── 归因实验 (新增)
    ├── Exp5: U×V交互归因
    ├── Exp7: 方向叠加归因
    └── Exp8: 分解归因 (方向 vs 放大)
```

---

## 实验5: U×V交互归因 (Interaction Attribution)

### 研究问题
MA生成是U和V**独立贡献**还是需要**协同作用**？

### 数学公式
```
W₂ = U @ Σ @ Vᵀ

测试四种条件:
1. Baseline:     W = U @ Σ @ Vᵀ
2. 消融U:        W = U_random @ Σ @ Vᵀ
3. 消融V:        W = U @ Σ @ V_random
4. 同时消融:     W = U_random @ Σ @ V_random

归因计算:
U贡献 = baseline - 消融V  (保留U，移除V，看U的作用)
V贡献 = baseline - 消融U  (保留V，移除U，看V的作用)
交互 = baseline - U贡献 - V贡献 + 同时消融
```

### 预期发现
- **SVD对齐型** (Qwen, GPT-2): 交互项≈0，U和V独立贡献
- **多方向型** (BLOOM, LLaMA): 交互项显著，需要U×V协同

### 运行方法
```bash
python experiments/common/exp5_uv_interaction.py \
    --model qwen2.5_7b \
    --layer 3 \
    --nsamples 5 \
    --savedir results/experiments/exp5/qwen2.5_7b
```

### 输出文件 (exp2b格式)
```
results/experiments/exp5/{model}/
├── baseline.json          # 原始模型MA
├── ablate_u.json         # 消融U后MA
├── ablate_v.json         # 消融V后MA
├── ablate_both.json      # 同时消融MA
└── summary.json          # 归因百分比汇总
```

### Summary.json结构
```json
{
  "model": "qwen2.5_7b",
  "layer": 3,
  "attribution": {
    "baseline": 8768.0,
    "u_attribution_pct": 48.5,
    "v_attribution_pct": 46.2,
    "interaction_pct": 2.1,
    "total_explained": 96.8,
    "interpretation": "independent"  // 或 "synergistic"
  }
}
```

---

## 实验7: 方向叠加归因 (Direction Superposition)

### 研究问题
对于多方向型模型，最少需要多少个方向才能恢复90% MA？每个方向的边际贡献是多少？

### 数学公式
```
逐步添加主成分:
W_k = U[:, :k] @ diag(S[:k]) @ Vh[:k, :]

对于 k = 1, 2, 3, 5, 10, 20, 50, 100:
  recovery_rate[k] = MA(W_k) / MA(W_full)
  marginal[k] = MA(W_k) - MA(W_{k-1})

找到最小k: recovery_rate[k] >= 0.9
```

### 预期发现
- **Qwen**: k=1时即可恢复>90% → 单方向主导
- **BLOOM**: k>5才能恢复 → 多方向协作
- **关键k值**：判断机制类型的阈值

### 运行方法
```bash
python experiments/common/exp7_direction_superposition.py \
    --model bloom_7b1 \
    --layer 28 \
    --nsamples 5 \
    --k-values 1,2,3,5,10,20,50,100 \
    --savedir results/experiments/exp7/bloom_7b1
```

### 输出文件
```
results/experiments/exp7/{model}/
├── baseline.json       # 完整矩阵MA
├── keep_k1.json       # 只保留1个方向
├── keep_k2.json       # 只保留2个方向
├── keep_k3.json
├── keep_k5.json
├── keep_k10.json
├── ...
└── summary.json       # 边际贡献和关键k值
```

### Summary.json结构
```json
{
  "model": "bloom_7b1",
  "layer": 28,
  "baseline_mean": 394.2,
  "critical_k": 5,
  "marginal_contributions": {
    "1": {
      "cumulative_pct": 12.5,
      "marginal_contribution": 49.3,
      "recovery_rate": 0.125
    },
    "5": {
      "cumulative_pct": 81.2,
      "marginal_contribution": 270.7,
      "recovery_rate": 0.812
    },
    "10": {
      "cumulative_pct": 92.1,
      "marginal_contribution": 43.0,
      "recovery_rate": 0.921
    }
  }
}
```

---

## 实验8: 分解归因 (Decomposed Attribution)

### 研究问题
W₂的作用分解为"方向"(U×V)和"放大"(Σ)，哪个更关键？

### 数学分解
```
W₂ @ h₂ = (U @ Σ @ Vᵀ) @ h₂
         = U @ (Σ @ (Vᵀ @ h₂))

阶段1: α = Vᵀ @ h₂       (输入投影)
阶段2: β = Σ @ α         (奇异值放大)
阶段3: output = U @ β    (输出映射)

测试条件:
1. Baseline:        W = U @ Σ @ Vᵀ
2. 消融方向:        W = U_rand @ Σ @ V_rand  (破坏方向，保留放大)
3. 消融放大:        W = U @ Σ_uniform @ Vᵀ  (均匀化放大，保留方向)
4. 同时消融:        W = U_rand @ Σ_uniform @ V_rand
```

### 归因公式
```
方向贡献 = baseline - 消融方向
放大贡献 = baseline - 消融放大
交互 = baseline - 方向贡献 - 放大贡献 + 同时消融

约束: 方向% + 放大% + 交互% = 100%
```

### 预期发现
- 如果**方向>70%**: MA主要由U×V结构决定
- 如果**放大>70%**: MA主要由奇异值大小决定
- 如果两者都重要: 需要协同

### 运行方法
```bash
python experiments/common/exp8_decomposed_attribution.py \
    --model qwen2.5_7b \
    --layer 3 \
    --nsamples 5 \
    --savedir results/experiments/exp8/qwen2.5_7b
```

### 输出文件
```
results/experiments/exp8/{model}/
├── baseline.json            # 原始模型
├── ablate_direction.json   # 消融方向
├── ablate_magnitude.json   # 消融放大
├── ablate_both.json        # 同时消融
└── summary.json            # 归因百分比
```

### Summary.json结构
```json
{
  "model": "qwen2.5_7b",
  "layer": 3,
  "attribution": {
    "baseline": 8768.0,
    "direction_attribution_pct": 67.3,
    "magnitude_attribution_pct": 28.5,
    "interaction_pct": 4.2,
    "interpretation": {
      "direction_dominance": "high",
      "magnitude_dominance": "medium"
    }
  }
}
```

---

## 批量运行

### 使用Shell脚本（推荐）
```bash
# 运行所有模型的所有归因实验
./run_attribution_experiments.sh

# 只运行特定模型
./run_attribution_experiments.sh \
    --models gpt2,qwen2.5_7b,bloom_7b1 \
    --experiments exp5,exp7,exp8 \
    --nsamples 5

# 只运行特定实验
./run_attribution_experiments.sh \
    --experiments exp5 \
    --nsamples 10
```

### 使用Python脚本
```bash
python scripts/run_attribution_experiments.py \
    --models gpt2 gptj_6b bloom_7b1 qwen2.5_7b \
    --experiments exp5 exp7 exp8 \
    --nsamples 5 \
    --max-memory 75
```

### 输出汇总
批量运行完成后会生成：
- `results/experiments/attribution_summary.json`: JSON格式汇总
- `results/experiments/ATTRIBUTION_SUMMARY.md`: Markdown报告

---

## 数据存储格式 (遵循Exp2b标准)

所有归因实验使用统一的存储格式：

### 1. 详细结果文件
每个干预条件一个JSON文件，包含：
```json
{
  "experiment": "exp5_uv_interaction_ablate_u",
  "model": "qwen2.5_7b",
  "layer": 3,
  "date": "2025-12-23T15:00:00",
  "n_samples": 5,
  "summary": {
    "mean": 450.2,
    "std": 15.3,
    "min": 432.1,
    "max": 468.5,
    "values": [450.2, 445.3, 460.1, 442.8, 452.7]
  },
  "results": {
    "0": {"mean": 450.2, "n_samples": 1},
    "1": {"mean": 445.3, "n_samples": 1},
    ...
  }
}
```

### 2. Summary汇总文件
每个实验目录下的`summary.json`包含归因百分比：
```json
{
  "model": "qwen2.5_7b",
  "layer": 3,
  "date": "2025-12-23T15:30:00",
  "n_samples": 5,
  "attribution": {
    // 归因百分比数据
  }
}
```

---

## 关键发现的解释

### 两种MA生成机制

#### 1. SVD对齐型 (GPT-2, Qwen-2.5)
**特征:**
- σ₁/σ₂ > 2.0
- R² > 0.9 (旧Exp3)
- critical_k = 1 (Exp7)

**归因特点 (预期):**
- Exp5: U贡献≈45%, V贡献≈45%, 交互≈5% (独立)
- Exp7: k=1时恢复>90% MA
- Exp8: 方向贡献>65%

**机制公式:**
```
MA ≈ σ₁ * |h₂ · v₁| * |u₁[dim]|
```

#### 2. 多方向协作型 (BLOOM, LLaMA-2, Mistral)
**特征:**
- σ₁/σ₂ < 2.0
- R² < 0.1 (旧Exp3)
- critical_k > 5 (Exp7)

**归因特点 (预期):**
- Exp5: 交互项>20% (协同)
- Exp7: 需要k>5才能恢复90% MA
- Exp8: 方向和放大都重要

**机制公式:**
```
MA ≈ Σᵢ σᵢ * |h₂ · vᵢ| * |uᵢ[dim]|  (i=1..k)
```

---

## 使用示例

### 场景1: 分析新模型的MA机制
```bash
# Step 1: 运行Exp4查看SVD结构
python experiments/common/exp4_mlp_svd_analysis.py \
    --model new_model \
    --layers 0,1,2,3 \
    --savedir results/models/new_model/exp4

# Step 2: 如果σ₁/σ₂ > 2，运行所有归因实验
./run_attribution_experiments.sh \
    --models new_model \
    --experiments exp5,exp7,exp8

# Step 3: 查看summary.json判断机制类型
cat results/experiments/exp5/new_model/summary.json
cat results/experiments/exp7/new_model/summary.json
cat results/experiments/exp8/new_model/summary.json
```

### 场景2: 对比两个模型的归因差异
```bash
# 运行两个模型
./run_attribution_experiments.sh \
    --models gpt2,bloom_7b1 \
    --experiments exp5,exp7,exp8

# 对比结果
python scripts/compare_attribution.py \
    --models gpt2 bloom_7b1 \
    --experiments exp5 exp7 exp8
```

### 场景3: 验证假设
假设：Qwen的MA完全由单方向决定

```bash
# 运行Exp7
python experiments/common/exp7_direction_superposition.py \
    --model qwen2.5_7b \
    --layer 3 \
    --k-values 1,2,3,5 \
    --nsamples 10

# 检查critical_k
jq '.critical_k' results/experiments/exp7/qwen2.5_7b/summary.json
# 预期输出: 1

# 检查k=1的恢复率
jq '.marginal_contributions."1".recovery_rate' \
    results/experiments/exp7/qwen2.5_7b/summary.json
# 预期输出: > 0.9
```

---

## 技术细节

### 随机正交矩阵生成
```python
def create_random_orthogonal(shape, device='cpu'):
    """使用QR分解生成随机正交矩阵"""
    random_matrix = torch.randn(shape, device=device)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q
```

### 权重替换流程
1. 保存原始权重
2. SVD分解: W = U @ Σ @ Vᵀ
3. 构建干预权重 (替换U/V/Σ)
4. 设置新权重到模型
5. 运行评估
6. 恢复原始权重

### GPU显存管理
- 每个实验独立进程
- 实验间自动清理显存
- 监控显存使用，防止OOM

---

## 预期贡献

完成归因实验后，我们将获得：

1. **完整因果链**: 结构(Exp4) → 相关(旧Exp3) → 因果(新Exp3+Exp6) → 归因(Exp5/7/8)

2. **定量公式**:
   ```
   MA = α·U + β·V + γ·Σ + δ·(U×V) + ε·(U×V×Σ)
   其中 α+β+γ+δ+ε = 100%
   ```

3. **机制分类准则**:
   - critical_k = 1 → SVD对齐型
   - critical_k > 5 → 多方向型
   - 2 < critical_k < 5 → 混合型

4. **干预指导**: 明确优化方向
   - 如果方向主导 → 优化U×V结构
   - 如果放大主导 → 调整奇异值分布

---

## 引用

如果使用本框架，请引用：
```bibtex
@software{attribution_experiments,
  title={Attribution Experiments Framework for Massive Activation Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/your/repo}
}
```

---

## 常见问题 (FAQ)

### Q1: 为什么Exp5/7/8都需要？
A: 三个实验从不同角度分析：
- Exp5: 测试U和V是否需要协同
- Exp7: 找到最小方向数量
- Exp8: 分离方向和放大的贡献

### Q2: 如何选择nsamples?
A:
- 快速测试: nsamples=5 (约5分钟/模型)
- 标准分析: nsamples=10 (约10分钟/模型)
- 发表论文: nsamples=20+ (约20分钟/模型)

### Q3: 为什么用随机正交矩阵而不是零矩阵？
A: 随机正交矩阵保持了：
- 矩阵秩不变
- 不破坏数值稳定性
- 只破坏特定方向信息

### Q4: 如何处理opt_6.7b命名问题？
A: 模型配置文件使用'opt_7b'，需要在运行时指定：
```bash
./run_attribution_experiments.sh --models opt_7b
```

---

**文档版本**: v1.0
**更新日期**: 2025-12-23
**维护者**: Claude Code Assistant
