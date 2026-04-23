# opt_6.7b — RQ5 v ablation 分析

**模型分类**：`ANOM` | **起源层 L**：`1`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 1
- **σ₁** = 19.472166061401367
- **single baseline MA** = 216.175
- **single ablated MA** = 178.24583333333334
- **single ΔMA%** = -17.54558421032343%
- **macro σ₁** = 
- **macro ΔMA%** = %

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **❌ -18%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `opt_6.7b_v_ablation_results.json`
- `secondary/`（子目录）

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：❌ -18%

**此模型综合评分**：3/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。