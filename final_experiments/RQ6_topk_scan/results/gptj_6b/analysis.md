# gptj_6b — RQ6 topk scan 分析

**模型分类**：`CONC` | **起源层 L**：`2`

**本 RQ 在问什么**：Top-K 恢复：仅保留单层 top-K 激活，MA 是否恢复

---

## 关键指标

- **critical_layer (peak)** = 3
- **sigma_ratio** = None
- **baseline** = None

**判据**：CONC 期望 recovery ≥ 30% / 多层期望 recovery < 30% 一致性 — **✅ 76%**

**结论**：仅保留 top-K 单层激活，测 MA 恢复率。本模型状态：✅ 76%

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `gptj_6b_macro_svd_full.json`

---

## 重跑命令

```bash
# 见 RQ6_topk_scan/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ 76%

**此模型综合评分**：6/6 ⭐⭐⭐

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。