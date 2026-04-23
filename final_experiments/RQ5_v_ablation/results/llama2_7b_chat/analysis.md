# llama2_7b_chat — RQ5 v ablation 分析

**模型分类**：`—` | **起源层 L**：`1`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 1
- **σ₁** = 6.60770320892334
- **single baseline MA** = 2175.866666666667
- **single ablated MA** = 94.52291666666666
- **single ΔMA%** = -95.65584977633434%
- **macro σ₁** = 
- **macro ΔMA%** = %

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **✅ L=1 ΔMA=-96% (2026-04-22 修正错层)**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `llama2_7b_chat_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ L=1 ΔMA=-96% (2026-04-22 修正错层)

**此模型综合评分**：4/6

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。