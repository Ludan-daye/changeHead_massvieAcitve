# opt_6.7b — RQ2a mlp 分析

**模型分类**：`ANOM` | **起源层 L**：`1`

**本 RQ 在问什么**：MLP 全消融：禁用全部 MLP 后 MA 是否归零（验证 'MLP 是 MA 起源'）

---

## 关键指标

- **baseline_max_ma** = N/A
- **disabled_max_ma** = N/A
- **retention%** = N/A%
- **reduction%** = N/A%

**判据**：retention ≤ 10% 即 PASS — **⏳ hook 异常**

**结论**：MLP 全消融后 MA 大幅下降 → 验证 H₁（MLP 是 MA 主来源）。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `MISSING.txt`

---

## 重跑命令

```bash
# 见 RQ2a_mlp/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：⏳ hook 异常

**此模型综合评分**：3/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。