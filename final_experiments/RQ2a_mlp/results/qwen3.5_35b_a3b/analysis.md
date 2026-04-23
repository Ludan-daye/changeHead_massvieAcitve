# qwen3.5_35b_a3b — RQ2a mlp 分析

**模型分类**：`FS MoE` | **起源层 L**：`9`

**本 RQ 在问什么**：MLP 全消融：禁用全部 MLP 后 MA 是否归零（验证 'MLP 是 MA 起源'）

---

## 关键指标

- **baseline_max_ma** = 37.78
- **disabled_max_ma** = 33.14
- **retention%** = 87.57%
- **reduction%** = 12.43%

**判据**：retention ≤ 10% 即 PASS — **❌ retain=87.6%**

**结论**：MLP 全消融后 MA 大幅下降 → 验证 H₁（MLP 是 MA 主来源）。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `all_mlp_disabled/`（子目录）
- `baseline/`（子目录）
- `comparison/`（子目录）

---

## 重跑命令

```bash
# 见 RQ2a_mlp/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：❌ retain=87.6%

**此模型综合评分**：1/5 架构特异

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。