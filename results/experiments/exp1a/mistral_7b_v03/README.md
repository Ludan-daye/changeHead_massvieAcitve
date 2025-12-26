# 实验2: 层贡献分析 (Layer Contribution Analysis)

## 实验目的
找出Mistral-7B-v0.3中哪一层Attention对MA贡献最大

## 实验方法
1. 禁用所有32层Attention Heads
2. 逐层恢复单层的Attention
3. 测量恢复后的激活值，找出关键层

## 核心结果

### 关键层排名
| 排名 | Layer | 恢复后激活 |
|------|-------|-----------|
| 🔴 **1** | **Layer 0** | **173.75** |
| 2 | Layer 30 | 107.08 |
| 3 | Layer 16 | 106.78 |
| 4 | Layer 28 | 106.58 |
| 5 | Layer 17 | 105.70 |
| ... | 其他 | ~104 |
| 🟢 最小 | Layer 1 | 100.58 |

### 关键发现
- **Layer 0**: 恢复后激活173.75，远超其他层（~104）
- **Layer 0 vs 其他**: 1.66倍差距
- 基线（全禁用）: 100.58

## 结论

**Layer 0 是 Mistral-7B 的关键层！**

- Mistral的MA在模型第一层就被Attention产生
- 这是所有测试模型中**最早**的关键层位置

### 各模型关键层对比
| 模型 | 关键层 | 位置 |
|------|--------|------|
| **Mistral** | **Layer 0** | **最早** |
| GPT-2 | Layer 2 | 早期 |
| Qwen | Layer 3 | 早期 |
| LLaMA | Layer 3 | 早期 |
| BLOOM | Layer 28 | 后期 |

## 文件列表
- `layer_contribution.json` - 各层贡献数据（含分析）
- `layer_contribution.png` - 可视化图表
