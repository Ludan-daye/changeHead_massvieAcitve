# yi_9b — RQ6 topk scan 分析

**模型分类**：`DISP` | **起源层 L**：`8`

**本 RQ 在问什么**：Top-K 恢复：仅保留单层 top-K 激活，MA 是否恢复

---

## 关键指标

- **critical_layer (peak)** = 0
- **sigma_ratio** = 2.0791
- **baseline** = 1.97

**判据**：CONC 期望 recovery ≥ 30% / 多层期望 recovery < 30% 一致性 — **—**

**结论**：仅保留 top-K 单层激活，测 MA 恢复率。本模型状态：—

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `yi_9b_rq6_results.json`

---

## 重跑命令

```bash
# 见 RQ6_topk_scan/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：—

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。