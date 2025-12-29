# MA来源验证实验

## 实验目的
验证Massive Activation是由Attention还是MLP产生

## 实验方法
在关键层分别Hook Attention和MLP的输出，对比最大激活值：

```python
# Hook Attention输出
def hook_attn(m, inp, out):
    data['attn'] = out[0].detach().cpu().numpy()

# Hook MLP输出  
def hook_mlp(m, inp, out):
    data['mlp'] = out.detach().cpu().numpy()

h1 = layer.self_attention.register_forward_hook(hook_attn)
h2 = layer.mlp.register_forward_hook(hook_mlp)

model(testseq)

attn_max = np.abs(data["attn"]).max()
mlp_max = np.abs(data["mlp"]).max()
```

## 实验结果

| 模型 | 关键层 | Attention输出Max | MLP输出Max | MA来源 | MLP/Attn比值 |
|------|--------|-----------------|------------|--------|--------------|
| GPT-J-6B | L0 | 3.61 | **30.33** | MLP | 8.4x |
| BLOOM-7B1 | L0 | 26.97 | **92.50** | MLP | 3.4x |
| Qwen-2.5-7B | L3 | 2.62 | **9160.00** | MLP | 3496x |
| Falcon-7B | L0 | 3.66 | **10.38** | MLP | 2.8x |
| Mistral-7B | L0 | 0.08 | **1.17** | MLP | 14.6x |

## 结论

**所有模型的MA都来自MLP输出，而非Attention输出。**

### 统一机制

```
Input → Attention → MLP → Output
           │          │
           │          └─ MA在这里产生（V矩阵编码）
           │
           └─ 提供触发输入（不产生MA本身）
```

### 与Exp1的关系

Exp1发现禁用Attention后MA变化的原因：
- **不是**因为MA来自Attention
- **而是**因为MLP需要Attention提供正确的输入模式才能产生MA
- 禁用Attention → MLP输入改变 → 无法触发MA

---

*实验日期: 2025-11-28*
