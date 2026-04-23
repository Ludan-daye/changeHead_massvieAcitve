# glm4_9b — RQ5 v ablation 分析

**模型分类**：`FS` | **起源层 L**：`1`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 1
- **σ₁** = 15.062995910644531
- **single baseline MA** = 470.93333333333334
- **single ablated MA** = 252.16666666666666
- **single ΔMA%** = -46.45385050962628%
- **macro σ₁** = 39492.421875
- **macro ΔMA%** = -81.75592960979343%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ macro -82%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `glm4_9b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ macro -82%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。