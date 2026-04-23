# gptj_6b — RQ1 attention 分析

**模型分类**：`CONC` | **起源层 L**：`2`

**本 RQ 在问什么**：Attention 消融：禁用全部 attention 层后 MA 是否仍存在（证伪 'attention 是 MA 起源'）

---

## 关键指标

- **baseline_top1** = 4246.0
- **disabled_top1** = 240.03
- **ΔMA%** = -94.35%
- **mode** = generative
- **peak_layer** = 16

**判据**：residual% > 0（disabled 非 0）即 PASS — **✅ Gen ΔMA=-98.3% (residual=1.69% 最小)**

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

**此模型 × 此 RQ**：✅ Gen ΔMA=-98.3% (residual=1.69% 最小)

**此模型综合评分**：6/6 ⭐⭐⭐

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。