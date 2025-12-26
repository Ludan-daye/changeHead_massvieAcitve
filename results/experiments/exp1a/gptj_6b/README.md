# GPT-J-6B Experiment 2: Layer Contribution Analysis

## 实验目的
识别哪些层对MA贡献最大

## 方法
1. 禁用所有Attention heads
2. 依次恢复每层的heads
3. 测量恢复后的激活值

## 核心结果

### 🔥 最大贡献层
| 层 | 恢复后激活 | vs 全禁用 |
|----|-----------|-----------|
| **Layer 0** | **1804.80** | **7.5x** |
| Layer 24 | 253.12 | 1.1x |
| Layer 26 | 246.88 | 1.1x |

### 🧊 最小贡献层
| 层 | 恢复后激活 |
|----|-----------|
| Layer 6 | 221.18 |
| Layer 8 | 229.03 |
| Layer 15 | 229.15 |

## 结论

### Layer 0 是产生MA的关键层

1. Layer 0贡献是其他层的**7.5倍**
2. 其他层贡献相近（220-250）
3. 与Falcon、Mistral相似，都在Layer 0产生MA

## 可视化
见 `layer_contribution.png`
