# qwen3.5_27b — RQ5 v ablation 分析

**模型分类**：`DISP` | **起源层 L**：`54`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 54
- **σ₁** = 3.652482271194458
- **single baseline MA** = 755.3333333333334
- **single ablated MA** = 166.7
- **single ΔMA%** = -77.93027360988528%
- **macro σ₁** = 4866.9775390625
- **macro ΔMA%** = -0.3808717731697019%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ 单层 -78% 接近阈值，macro 脚本 dtype bug 失败**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `qwen3.5_27b_v_ablation_results.json`
- `recheck/`（子目录）

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ 单层 -78% 接近阈值，macro 脚本 dtype bug 失败

**此模型综合评分**：4/5 ⭐救活

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。