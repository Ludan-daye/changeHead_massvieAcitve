# qwen1.5_14b — RQ3 function words 分析

**模型分类**：`DISP` | **起源层 L**：`2`

**本 RQ 在问什么**：Function Token 定位：起源层 MA 极值位置是否落在功能词/结构 token

---

## 关键指标

- **exp3 字段**：{"function_alignment_mean": 0.21369943181078838, "content_alignment_mean": 0.10574784505328015, "cohens_d": 0.6896009503282139, "p_value": 4.766635384730897e-124, "significant": null, "n_layers": 1, "svd_info": {}}

**判据**：Top-1 MA token 是 function_token（含广义 FT：标点/换行/符号）即 PASS — **✅**

**结论**：Top-1 MA 位置落在结构 token（不是内容词），支持 'MA 是 MLP 在 FT 位置写的 mark' 论点。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `EXP5_SUMMARY.txt`
- `exp5_alignment_v1.png`
- `exp5_asymmetry_analysis.png`
- `exp5_concentration_top5.png`
- `exp5_detailed_results.json`
- `exp5_stability_analysis.png`
- `table1_rq3.json`

---

## 重跑命令

```bash
# 见 RQ3_function_words/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅

**此模型综合评分**：4/5 ⭐救活

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。