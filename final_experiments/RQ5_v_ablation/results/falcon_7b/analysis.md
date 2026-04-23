# falcon_7b — RQ5 v ablation 分析

**模型分类**：`FS` | **起源层 L**：`3`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 3
- **σ₁** = 13.275040626525879
- **single baseline MA** = 1193.6666666666667
- **single ablated MA** = 22.273177083333334
- **single ΔMA%** = -98.13405386065345%
- **macro σ₁** = 11220.3505859375
- **macro ΔMA%** = -97.42551836079308%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ -98% / macro -97%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `falcon_7b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ -98% / macro -97%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。