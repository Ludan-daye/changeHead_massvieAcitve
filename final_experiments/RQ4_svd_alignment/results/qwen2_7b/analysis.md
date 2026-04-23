# qwen2_7b — RQ4 svd alignment 分析

**模型分类**：`CONC` | **起源层 L**：`3`

**本 RQ 在问什么**：SVD 几何对齐：公式 MA = σ₁·(h₂·v₁)·u₁[j*] 的拟合精度（K=1 R² 或 macro V 消融）

---

## 关键指标

- **起源层 L** = 3
- **σ₁** = 16.458057857404615
- **σ₁/σ₂ 比** = 2.8382876736329177
- **R² (K=1 公式拟合)** = 0.9998874751382515
- **slope β** = -12.47500335254364
- **intercept b** = 1.975547293665386
- **max|proj_F|** = 456.7265844621404
- **max|MA_F|** = 5692.0

**判据**：K=1 R² ≥ 0.95 OR macro V 消融 ΔMA ≤ -80%（任一即 PASS） — **✅ K=1 R²=1.000 (err 0.6%)**

**结论**：基于公式 MA = Σᵢ σᵢ·(h₂·vᵢ)·uᵢ[j\*]，此模型在 K=1 单项/多项截断下拟合精度见上。

---

## 数据文件

位于本目录 `./` 下。主要产出：
- `EXPERIMENT_3_SUMMARY.txt`
- `exp3_alignment_comparison.png`
- `exp3_detailed_results.json`
- `exp3_projection_regression.png`
- `exp3_singular_values.png`
- `exp3_top_tokens.png`
- `exp3_trigger_rate.png`
- `table1_rq4.json`

---

## 重跑命令

```bash
# 见 RQ4_svd_alignment/code/ 或 paper_experiments/
# 详细参数参见 ../README.md
```

## 总评

**此模型 × 此 RQ**：✅ K=1 R²=1.000 (err 0.6%)

**此模型综合评分**：5/5 核心证据

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。