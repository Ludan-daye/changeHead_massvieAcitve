# qwen3.5_9b — RQ5 v ablation 分析

**模型分类**：`DISP` | **起源层 L**：`22→26`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 22
- **σ₁** = 3.1101417541503906
- **single baseline MA** = 175.66666666666666
- **single ablated MA** = 52.391666666666666
- **single ΔMA%** = -70.17552182163188%
- **macro σ₁** = 3082.75732421875
- **macro ΔMA%** = -0.24866785079929055%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **❌ K=20 -16% / macro -57%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `qwen3.5_9b_v_ablation_results.json`
- `recheck/`（子目录）

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：❌ K=20 -16% / macro -57%

**此模型综合评分**：2/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。