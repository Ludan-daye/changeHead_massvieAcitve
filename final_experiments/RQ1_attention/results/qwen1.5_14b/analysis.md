# qwen1.5_14b — RQ1 attention 分析

**模型分类**：`DISP` | **起源层 L**：`2`

**本 RQ 在问什么**：Attention 消融：禁用全部 attention 层后 MA 是否仍存在（证伪 'attention 是 MA 起源'）

---

## 关键指标

- **baseline_top1** = 7444.0
- **disabled_top1** = 6317.2
- **ΔMA%** = -15.14%
- **mode** = generative
- **peak_layer** = 37

**判据**：residual% > 0（disabled 非 0）即 PASS — **✅**

**结论**：attention 消融后 MA 未归零，证伪 H₀（attention 是 MA 起源）。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `MISSING.txt`

---

## 重跑命令

```bash
# 见 RQ1_attention/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅

**此模型综合评分**：4/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。