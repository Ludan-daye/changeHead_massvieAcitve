# glm4_32b — RQ5 v ablation 分析

**模型分类**：`CONC` | **起源层 L**：`0`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 0
- **σ₁** = 116.74832916259766
- **single baseline MA** = 301192.5333333333
- **single ablated MA** = 9614.933333333332
- **single ΔMA%** = -96.80771192203082%
- **macro σ₁** = 954371072.0
- **macro ΔMA%** = -17.078422484134176%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ -97%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `glm4_32b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ -97%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。