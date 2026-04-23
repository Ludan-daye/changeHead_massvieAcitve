# glm4_9b — RQ2a mlp 分析

**模型分类**：`FS` | **起源层 L**：`1`

**本 RQ 在问什么**：MLP 全消融：禁用全部 MLP 后 MA 是否归零（验证 'MLP 是 MA 起源'）

---

## 关键指标

- **baseline_max_ma** = 2250.4
- **disabled_max_ma** = 107.47
- **retention%** = 4.48%
- **reduction%** = 95.52%

**判据**：retention ≤ 10% 即 PASS — **✅ retain=4.5%**

**结论**：MLP 全消融后 MA 大幅下降 → 验证 H₁（MLP 是 MA 主来源）。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `MISSING.txt`
- `all_mlp_disabled/`（子目录）
- `baseline/`（子目录）

---

## 重跑命令

```bash
# 见 RQ2a_mlp/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ retain=4.5%

**此模型综合评分**：5/5

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。