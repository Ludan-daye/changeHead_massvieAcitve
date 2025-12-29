# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证LLaMA-2-13B的Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型
2. **All Heads Disabled**: 将所有Attention Heads输出置零
3. **对比**: 分析激活值变化

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 1,283 (Layer 22) | 263 | **-80%** |
| Dim 447 | 10.22 | 5.30 | -48% |
| Dim 138 | 9.11 | 5.40 | -41% |

### 关键层变化
| Layer | Baseline | Disabled | 变化 |
|-------|----------|----------|------|
| Layer 3 | 1,224 | 22 | **-98%** |
| Layer 7+ | ~1,273 | ~56 | -96% |

## 结论

**✅ Attention Heads 产生 Massive Activation**

- 禁用后激活下降80%
- Layer 3是关键层（首次爆发）
- 机制与GPT-2类似

## 文件列表
- `baseline/` - 基线测试结果
- `all_heads_disabled/` - 禁用头部测试结果
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告
