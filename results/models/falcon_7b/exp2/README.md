# 实验2: 层贡献分析 (Layer Contribution Analysis)

## 实验目的
找出Falcon-7B中哪一层Attention对MA贡献最大

## 实验方法
1. 禁用所有32层的Attention Heads
2. 逐层恢复单层的Attention
3. 测量恢复后的激活值，找出关键层

## 核心结果

### 关键层排名
| 排名 | Layer | 恢复后激活 |
|------|-------|-----------|
| 🔴 **1** | **Layer 0** | **1300.00** |
| 2 | Layer 26 | 1184.60 |
| 3 | Layer 23 | 1183.20 |
| 4 | Layer 27 | 1183.20 |
| 5 | Layer 5 | 1183.00 |
| ... | 其他 | ~1183 |
| 🟢 最小 | Layer 2 | 873.70 |

### 关键发现
- **Layer 0**: 恢复后激活1300，比其他层（~1183）高约10%
- Layer 2恢复后激活反而最低（874）

## 结论

**Layer 0 是 Falcon-7B 的关键层！**

- Falcon的MA在模型第一层就被Attention产生
- 与Mistral相同，关键层都在Layer 0

### 各模型关键层对比
| 模型 | 关键层 | 位置 |
|------|--------|------|
| **Falcon** | **Layer 0** | **最早** |
| **Mistral** | **Layer 0** | **最早** |
| GPT-2 | Layer 2 | 早期 |
| Qwen | Layer 3 | 早期 |
| LLaMA | Layer 3 | 早期 |
| BLOOM | Layer 28 | 后期 |

## 文件列表
- `layer_contribution.json` - 各层贡献数据（含分析）
- `layer_contribution.png` - 可视化图表
