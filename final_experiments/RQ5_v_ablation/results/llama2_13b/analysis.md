# llama2_13b — RQ5 v ablation 分析

**模型分类**：`FS` | **起源层 L**：`0`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 0
- **σ₁** = 9.743122100830078
- **single baseline MA** = 64.759375
- **single ablated MA** = 2.827018229166667
- **single ΔMA%** = -95.6345807395969%
- **macro σ₁** = 39634.8046875
- **macro ΔMA%** = -28.57298686188006%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ -96%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `llama2_13b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ -96%

**此模型综合评分**：4/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。