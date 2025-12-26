# RQ4 数据状态报告

SVD Alignment（奇异值方向与激活方向对齐）

---

## 数据来源与范围

各模型目录中与RQ4相关的数据文件梳理如下（命名不完全统一，已对应至RQ4语义）：

| 模型 | 分析对象 | 文件路径 | 层覆盖 | 关键指标（示例） | 数据状态 | 需要补充 |
|------|----------|---------|--------|------------------|----------|----------|
| GPT-J | MLP（down/输出投影） | `results/models/gptj_6b/exp4/mlp_svd.json` | L0, L27 | σ1/σ2: L0=1.908, L27=1.812；cos_sim(L0)=-0.692 | ✅ 可用（核心层覆盖） | 可扩层，补充统一对齐指标 |
| BLOOM | Attention（o/dense） | `results/models/bloom_7b1/exp4/attention_svd.json` | L0,7,12,26,27,28,29 | σ1/σ2: L0=2.618, L29=2.226 | ⚠️ 可用（注意是Attention，对齐口径不一） | 建议增加MLP版（与其他模型统一） |
| Qwen | MLP（down/输出投影） | `results/models/qwen2.5_7b/exp4/svd_analysis.json` | 多层（含L0..3,26等） | L3: σ1/σ2=2.641；对齐(447/138) | ✅ 完整（含层级与对齐统计） | 否 |
| Falcon | MLP（down/输出投影） | `results/models/falcon_7b/exp4/mlp_svd.json` | L0,1,2,30,31 | L31: σ1/σ2=1.986 | ✅ 完整（多层） | 否 |
| Mistral | Attention（o_proj） | `results/models/mistral_7b_v03/exp4/attention_svd.json` | L0,1,16,30,31 | L31: σ1/σ2=2.190 | ⚠️ 可用（Attention口径） | 建议增加MLP版 |
| GPT-2 | MLP（c_proj） | `results/models/gpt2/exp3_svd_alignment/…` | 多项结果 | R²强（见EXPERIMENT_3_SUMMARY） | ✅ 完整（不同目录名） | 否 |
| OPT | MLP（fc2） | `results/models/opt_6.7b/exp4_opt_svd/alignment_results.json` | 目标L0,3,29,31 | 文件不完整（被截断） | ❌ 不完整 | ✅ 需重跑 |

说明：
- RQ4的“统一口径”推荐分析MLP输出投影（down_proj/fc2等），BLOOM、Mistral当前为Attention投影，对齐结论可用但与其他模型口径不一致。
- GPT-2的RQ4结果在exp3目录，包含更完整的对齐、回归与可视化，等价可用。

---

## 结论与建议

- ✅ 已达成：Qwen、Falcon、GPT-J（核心层）、GPT-2（不同目录）
- ⚠️ 可用但口径需统一：BLOOM、Mistral（当前为Attention，建议补充MLP版）
- ❌ 待补：OPT（alignment_results.json不完整，需重跑导出）

### 建议动作（按优先级）
1) P0：重跑 OPT 的 RQ4（建议层：0,3,29,31；nsamples≈20），补齐 `alignment_results.json` 与 summary。
2) P1：为 BLOOM 与 Mistral 增加“MLP输出投影”版 SVD对齐（与其他模型统一口径）。
3) P2：GPT-J 可扩展更多层（当前仅L0/L27），补充对齐可比性。

---

## 已知数据要点（示例）
- Qwen-L3：σ1/σ2≈2.64，且提供维度447/138的对齐统计（rank/value）。
- Falcon-L31：σ1/σ2≈1.99，存在明显主导奇异方向。
- BLOOM-L0：Attention dense σ1/σ2≈2.62；L29≈2.23（Attention视角）。
- GPT-J-L0：σ1/σ2≈1.91，并有cos_sim≈-0.692（实现口径与其他略异）。

---

生成时间：2025-12-11
