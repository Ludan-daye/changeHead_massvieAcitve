# qwen3_32b — RQ5 v ablation 分析

**模型分类**：`DISP` | **起源层 L**：`6`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 6
- **σ₁** = 21.45124053955078
- **single baseline MA** = 20501.333333333332
- **single ablated MA** = 430.6
- **single ΔMA%** = -97.89964880332988%
- **macro σ₁** = 71743.4375
- **macro ΔMA%** = -86.34875483039932%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ -98% / macro -86%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `qwen3_32b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ -98% / macro -86%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。