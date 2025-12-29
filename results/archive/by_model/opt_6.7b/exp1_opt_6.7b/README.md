# 实验1: 头抑制可行性测试 (Feasibility Test)

## 实验目的
验证OPT-6.7B的Attention Heads是否产生Massive Activation

## 实验方法
1. **Baseline**: 正常运行模型
2. **All Heads Disabled**: 将所有Attention Heads输出置零
3. **对比**: 分析激活值变化

## 核心结果

| 指标 | Baseline | Disabled | 变化 |
|------|----------|----------|------|
| Top1峰值 | 391 (Layer 25) | 1,370 | **+250%** ⬆️ |
| Dim 447 | 13.74 | 20.09 | +46% |
| Dim 138 | 12.96 | 15.72 | +21% |

### 关键层变化
| Layer | Baseline | Disabled | 变化 |
|-------|----------|----------|------|
| Layer 0 | 155 | 1,305 | **+744%** |
| Layer 4 | 307 | 1,370 | +347% |

## 结论

**❗ Attention Heads 抑制 Massive Activation**

禁用后激活反而上升250%，说明OPT的机制与Qwen类似：
- MLP产生MA
- Attention起抑制作用
- 与LLaMA/GPT-2机制相反

## 文件列表
- `baseline/` - 基线测试结果
- `all_heads_disabled/` - 禁用头部测试结果
- `comparison/EXPERIMENT_1_SUMMARY.txt` - 详细报告
