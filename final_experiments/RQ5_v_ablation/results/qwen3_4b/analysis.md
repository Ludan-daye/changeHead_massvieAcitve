# qwen3_4b — RQ5 v ablation 分析

**模型分类**：`FS` | **起源层 L**：`6`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 6
- **σ₁** = 6.24075984954834
- **single baseline MA** = 8051.6
- **single ablated MA** = 409.29583333333335
- **single ΔMA%** = -94.91659007733453%
- **macro σ₁** = 50097.984375
- **macro ΔMA%** = -99.67941730820023%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ -95% / macro -100%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `qwen3_4b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ -95% / macro -100%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。