# qwen3_14b — RQ5 v ablation 分析

**模型分类**：`DISP` | **起源层 L**：`6`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 6
- **σ₁** = 16.967206954956055
- **single baseline MA** = 12792.8
- **single ablated MA** = 825.3666666666667
- **single ΔMA%** = -93.54819377566548%
- **macro σ₁** = 135087.6875
- **macro ΔMA%** = -88.16367814886664%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ macro -88%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `qwen3_14b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ macro -88%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。