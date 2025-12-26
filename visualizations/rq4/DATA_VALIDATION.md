# RQ4 数据验证报告

SVD方向与激活方向对齐（Alignment）

---

## 数据来源
- 统一查找范围：`results/models/{model}/exp4*`、`results/models/{model}/RQ4_svd_alignment/*`、OPT 专用：`results/models/opt_6.7b/exp4_opt_svd/`
- 优先口径：MLP输出投影层（down_proj/fc2/c_proj 等）。Attention版结果可作为补充，但与多数模型口径不一致。

---

## 汇总表（当前可用产物）

| 模型 | 对象 | 核心文件 | 层覆盖 | 关键指标（示例） | 状态 |
|------|------|----------|--------|------------------|------|
| GPT-J | MLP | `exp4/mlp_svd.json` | L0, L27 | L0: σ1/σ2=1.908, cos_sim=-0.692 | ✅ 可用 |
| BLOOM | Attention | `exp4/attention_svd.json` | L0,7,12,26,27,28,29 | L0: σ1/σ2=2.618；L29: 2.226 | ⚠️ 可用（Attention口径） |
| Qwen | MLP | `exp4/svd_analysis.json` | 多层（含L0..3,26） | L3: σ1/σ2=2.641；含维度对齐rank | ✅ 完整 |
| Falcon | MLP | `exp4/mlp_svd.json` | L0,1,2,30,31 | L31: σ1/σ2=1.986 | ✅ 完整 |
| Mistral | Attention | `exp4/attention_svd.json` | L0,1,16,30,31 | L31: σ1/σ2=2.190 | ⚠️ 可用（Attention口径） |
| GPT-2 | MLP | `exp3_svd_alignment/*` | 多项结果 | R²强（见SUMMARY与图） | ✅ 完整（目录名不同） |
| OPT | MLP | `exp4_opt_svd/alignment_results.json` | 目标L0,3,29,31 | 文件不完整（被截断） | ❌ 待补 |

---

## 逐模型验证细节

### GPT-J (gptj_6b)
- 文件：`results/models/gptj_6b/exp4/mlp_svd.json`
- 层与指标：
  - L0: σ1=6.351, σ2=3.328, 比值=1.908；cos_sim=-0.692
  - L27: σ1=27.529, σ2=15.194, 比值=1.812；cos_sim=-0.060
- 结论：存在主导奇异方向（比值>1.8），与激活方向对齐度中等（L0）。
- 状态：✅ 可用（建议扩展更多层并统一对齐指标字段）。

### BLOOM (bloom_7b1)
- 文件：`results/models/bloom_7b1/exp4/attention_svd.json`
- 对象：Attention（dense/o-proj），非MLP。
- 代表层：
  - L0: σ1/σ2=2.618
  - L29: σ1/σ2=2.226
- 结论：Attention输出投影也存在主导奇异方向；但与多数模型“MLP口径”不一致。
- 状态：⚠️ 可用；建议补充MLP版本（down_proj/dense_4h_to_h）。

### Qwen (qwen2.5_7b)
- 文件：`results/models/qwen2.5_7b/exp4/svd_analysis.json`
- 代表层：
  - L3: σ1/σ2=2.6406；维度447对齐值=0.000686 (rank=2057)；维度138对齐值=7.25e-05 (rank=3411)
  - L1: σ1/σ2=1.6856；维度447对齐值=0.0183 (rank=452)
- 结论：提供了完整层级与维度对齐统计，可直接用于RQ4分析可视化。
- 状态：✅ 完整。

### Falcon (falcon_7b)
- 文件：`results/models/falcon_7b/exp4/mlp_svd.json`
- 代表层：
  - L0: σ1/σ2=2.8647
  - L31: σ1/σ2=1.9857
- 结论：多层存在主导奇异方向。
- 状态：✅ 完整。

### Mistral (mistral_7b_v03)
- 文件：`results/models/mistral_7b_v03/exp4/attention_svd.json`
- 对象：Attention（o_proj），非MLP。
- 代表层：
  - L31: σ1/σ2=2.190
  - L1: σ1/σ2=1.416
- 结论：Attention侧也有主导方向；建议补充MLP以统一口径。
- 状态：⚠️ 可用（Attention口径）。

### GPT-2 (gpt2)
- 文件：`results/models/gpt2/exp3_svd_alignment/*`
- 产物：EXPERIMENT_3_SUMMARY.txt、对齐图、回归（R²）等完整材料。
- 结论：与GPT-2论文相符，down_proj强对齐（高R²）。
- 状态：✅ 完整（目录名差异可接受）。

### OPT (opt_6.7b)
- 文件：`results/models/opt_6.7b/exp4_opt_svd/alignment_results.json`
- 问题：文件被截断（仅见`{"0": {"fc2_u1_alignment":`前缀）。
- 结论：当前不可用，需重跑导出。
- 状态：❌ 待补。

---

## 结论与后续建议
- 统一口径：建议以“MLP输出投影（down_proj/fc2/c_proj）”为主进行对齐分析；Attention版可作为补充材料。
- 数据缺口：
  1) OPT 的 RQ4 产物不完整 → 需重跑导出（建议层：0,3,29,31；nsamples≈20）。
  2) BLOOM/Mistral 当前为 Attention 口径 → 建议新增 MLP 版（与其他模型一致）。
  3) GPT-J 仅两层 → 可扩层以增强可比性。

---

生成时间：2025-12-11
