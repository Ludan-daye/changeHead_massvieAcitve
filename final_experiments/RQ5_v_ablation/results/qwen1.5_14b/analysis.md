# qwen1.5_14b — RQ5 v ablation 分析

**模型分类**：`DISP` | **起源层 L**：`2`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 35
- **σ₁** = 5.569987773895264
- **single baseline MA** = 7657.466666666666
- **single ablated MA** = 3883.4666666666667
- **single ΔMA%** = -49.28522923159965%
- **macro σ₁** = 29468.23046875
- **macro ΔMA%** = -12.625281775619898%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **🟡 L=2 K=1 ΔMA_max=-47% (mean -76%)**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `L2_multi_v/`（子目录）
- `qwen1.5_14b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：🟡 L=2 K=1 ΔMA_max=-47% (mean -76%)

**此模型综合评分**：4/5 ⭐救活

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。