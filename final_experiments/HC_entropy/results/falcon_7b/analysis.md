# falcon_7b — HC entropy 分析

**模型分类**：`FS` | **起源层 L**：`3`

**本 RQ 在问什么**：Huffman-code 熵：起源层各位置的 H(C) 熵分布（辅助）

---

## 关键指标

- **数据文件**：`exp5c_entropy_results.json` + `exp5c_raw_positions.npz`

**目的**：起源层各位置的 H(C) Huffman-code 熵分布，验证功能词的信息论锚点性质。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `exp5c_entropy_results.json`
- `exp5c_raw_positions.npz`

---

## 重跑命令

```bash
# 见 HC_entropy/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：N/A

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。