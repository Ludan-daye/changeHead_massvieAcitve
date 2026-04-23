# glm4_32b — u1 decode 分析

**模型分类**：`CONC` | **起源层 L**：`0`

**本 RQ 在问什么**：u₁ top-K 解码：将 u₁ 方向反解回词表，看哪些 token 对应最大 MA 增益（辅助）

---

## 关键指标

- **数据文件**：`glm4_32b_u1.json`（u₁ top-K token 解码）

**目的**：将 u₁[j\*] 方向反解回词表，看哪些 token 对应最大 MA 增益。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `glm4_32b_u1.json`

---

## 重跑命令

```bash
# 见 u1_decode/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：N/A

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。