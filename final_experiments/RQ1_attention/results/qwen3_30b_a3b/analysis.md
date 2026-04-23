# qwen3_30b_a3b — RQ1 attention 分析

**模型分类**：`DISP MoE` | **起源层 L**：`1`

**本 RQ 在问什么**：Attention 消融：禁用全部 attention 层后 MA 是否仍存在（证伪 'attention 是 MA 起源'）

---

## 关键指标

- **baseline_top1** = 1206.6
- **disabled_top1** = 967.3
- **ΔMA%** = -19.83%
- **mode** = generative
- **peak_layer** = 36

**判据**：residual% > 0（disabled 非 0）即 PASS — **✅**

**结论**：attention 消融后 MA 未归零，证伪 H₀（attention 是 MA 起源）。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `all_heads_disabled/`（子目录）
- `baseline/`（子目录）
- `comparison/`（子目录）
- `table1_rq1.json`

---

## 重跑命令

```bash
# 见 RQ1_attention/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅

**此模型综合评分**：3/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。