# qwen3.5_35b_a3b — RQ5 v ablation 分析

**模型分类**：`FS MoE` | **起源层 L**：`9`

**本 RQ 在问什么**：V 矩阵消融：替换 v₁ 方向（或 macro v₁ 投影消除）后 MA 是否塌陷（因果验证）

---

## 关键指标

- **起源层 L** = 9
- **σ₁** = 0.0478985421359539
- **single baseline MA** = 4.855729166666666
- **single ablated MA** = 4.858333333333333
- **single ΔMA%** = 0.053630805534705235%
- **macro σ₁** = 3003.326416015625
- **macro ΔMA%** = 0.5229734777736399%

**判据**：单层/macro V 消融 ΔMA ≤ -80% 即 PASS — **❌ +0%**

**结论**：替换 v₁ 方向后 MA 变化测因果性。macro 变体验证多层协作模式。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `qwen3.5_35b_a3b_v_ablation_results.json`

---

## 重跑命令

```bash
# 见 RQ5_v_ablation/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：❌ +0%

**此模型综合评分**：1/5 架构特异

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。