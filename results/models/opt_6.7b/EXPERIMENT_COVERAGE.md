# OPT-6.7B 实验覆盖情况（按RQ汇总）

> 目的：梳理OPT-6.7B的所有实验目录（即使命名不标准），对应到RQ1–RQ5，并标注数据可用性与后续动作。

---

## 一览表

| RQ | 研究问题 | 实验目录/文件 | 关键证据 | 结论/用途 | 状态 | 建议动作 |
|----|----------|----------------|----------|-----------|------|----------|
| RQ1 | Attention对MA的作用 | results/models/opt_6.7b/exp1_opt_6.7b/ | README.md, comparison/*.png | 禁用Attention后MA显著上升（+250%，L0 +744%）→ Attention抑制，MLP产出 | ✅ 完整 | 可直接用于图2（Attention Role） |
| RQ2 | MA来源：MLP vs Attention输出 | （暂无标准目录） | — | 尚无直接对比Attention输出vs MLP输出的数值表 | ⚠️ 缺失 | 可补充生成 verification.json（同其他模型RQ2） |
| RQ3 | 触发位置：功能词/标点 | （暂无） | — | 未做Token-level触发类型统计 | ⚠️ 缺失 | 可按通用脚本补充，产出 MA_POSITION_TOKEN_ANALYSIS_OPT.json |
| RQ4 | SVD对齐/主导方向 | results/models/opt_6.7b/exp4_opt_svd/ | alignment_results.json（不完整） | 文件不完整，无法得出对齐结论 | ⚠️ 待补 | 重新导出alignment_results.json；或用统一脚本重跑 |
| RQ5 | V矩阵依赖（V-Ablation） | results/models/opt_6.7b/exp6/ | v_ablation_simple.json, v_ablation_results.json, v_ablation_results.png | random-orthogonal替换V → -31.8%；remove_top_k/keep_top_k一致指向弱依赖 | ✅ 可用 | 建议增样本nsamples=10复核；保持L0层 | 

---

## 目录与文件详情

- exp1_opt_6.7b（对应RQ1）
  - baseline/results.json, all_heads_disabled/results.json
  - comparison/ 多张图（layerwise、top1等）
  - 结论：Attention抑制（禁用后MA上升），与Qwen类似；支持“MLP产生MA、Attention提供/调制触发”

- exp2_opt_6.7b（辅助RQ1）
  - EXPERIMENT_2_SUMMARY.txt（逐层恢复Attention，仍呈抑制特性，单层恢复不足以>50%恢复）
  - 用途：补充RQ1机制阐释（多层协同抑制）

- exp3_opt_fire_test（工程/探索性）
  - mlp_fire_stats.json（各层Top1/L2概览，非Token级）
  - 用途：可视化MLP各层激活强度分布，非论文核心证据

- exp4_opt_svd（对应RQ4，未完成）
  - alignment_results.json（内容不完整）
  - 用途：需补齐以对齐其他模型的SVD对齐分析

- exp6（对应RQ5，核心）
  - v_ablation_simple.json（Baseline:148.05 → Ablated:100.98，-31.8%）
  - v_ablation_results.json（含Top-k移除/保留；remove_top_k@5≈-31% 与simple一致）
  - 结论：OPT对V的依赖较弱；但与同为L0的GPT-J/Falcon不同，建议二次复核

---

## 命名映射（非标准 → 标准化建议）

| 非标准目录 | 建议映射为 | 说明 |
|------------|------------|------|
| exp1_opt_6.7b | exp1 | 与其他模型exp1语义一致（Attention作用） |
| exp2_opt_6.7b | exp2 | 层级贡献/恢复测试（Attention相关） |
| exp3_opt_fire_test | exp3_exploration | 探索性数据（可选纳入） |
| exp4_opt_svd | exp4 | SVD对齐（需补齐） |
| exp6 | exp6 | V消融（已对齐） |

如需统一调用脚本，可创建符号链接到标准名目录（不改变原数据）。

---

## 异常与注意事项

- v_ablation_simple.json 内字段 model 为 "opt_7b"（与目录名 opt_6.7b 不一致）
  - 影响：仅命名不一致，数值正确；建议统一元数据（model_id: opt_6.7b）
- RQ5弱依赖（-31.8%）与 remove_top_k 结果一致（≈-31%），说明现有实验方法内部自洽
  - 仍建议：将 nsamples=5 → 10 复核一次，排除采样方差影响

---

## 结论与下一步

- RQ1：✅ 充分；可直接用于论文图2
- RQ5：✅ 可用；但建议增加样本复核
- RQ2/3/4：⚠️ 尚未按“标准产物”输出（或文件不完整）

### 建议执行（按优先级）
1) P0：重跑 OPT RQ5（nsamples=10），校验-31.8%是否稳健（保留L0层设定）
2) P1：补齐 RQ4 对齐文件（alignment_results.json）
3) P2：按统一脚本生成 RQ2 验证（MLP vs Attention输出）
4) P3：如需，补充 RQ3 Token触发统计

> 是否需要我：
> - 统一创建标准化目录软链接？
> - 直接补齐 RQ4 和 RQ2 的统计脚本并运行？
