# RQ4 — SVD 几何对齐（执行计划）

## 做什么

在**起源层**做 SVD(`W_down`)，测：
- `σ₁/σ₂`（谱比 η）— 放大器"功率"的集中度
- `top_alignments`（MA 维度 vs `u₁` 主维度）— 输出集中位置

对应 MA 公式 `MA ≈ σ₁ × (h₂·v₁) × u₁[max]` 的 **σ₁ 和 u₁** 两项。

## 为什么必须 25 个全做 + 单层

已做的 18 个都在 peak±2 的 5 层上做，`avg_sigma_ratio` 普遍 **1.1 – 2.0**，**没有一个 ≥ 3**——和论文 GPT-J 起源层 η=5.74 严重不符。

**单层**（num_layers=1）的决定：详见 `docs/EXPERIMENT_PLAN.md` 附录 C。多层聚合分析归 RQ6。

## 使用的脚本

| 脚本 | 关键参数 | 输出 |
|---|---|---|
| `paper_experiments/RQ4_svd_alignment/exp3_svd_alignment_analysis.py` | `--model --layer_id L_ORIGIN --nsamples 30 --savedir` | `{model}_svd_alignment.json` 含 `sigma_ratio, top_alignments, sigma1, explained_var_sigma1` |

## 模型清单（25 个全做）

```bash
# 23 个已知起源层
bash run_rq345_origin_layer.sh "" rq4

# 依赖前置的 2 个延后
```

### 每模型层号表（与 RQ3 相同）

| 模型 | L_origin | 预期 σ₁/σ₂ |
|---|:-:|:-:|
| gptj_6b | 2 | **≥ 4.5**（Pilot 关键，论文 5.74）|
| bloom_7b1 | 3 | ≥ 3 |
| falcon_7b | 3 | ≥ 3 |
| llama3.1_8b | 1 | ≥ 3 |
| qwen2.5_0.5b | 0 | ≥ 3 |
| qwen2.5_7b | 3 | ≥ 3 |
| qwen2_7b | 3 | ≥ 3 |
| qwen3_0.6b | 2 | ≥ 3 |
| qwen3_1.7b | 2 | ≥ 3 |
| yi_9b | 1 | ≥ 3 |
| 中间 9 个 | 见 RQ3 PLAN | 2 – 3 |
| 模式 B 4 个（gpt2/opt/qwen3_32b/qwen3.5_27b）| 见 RQ3 PLAN | < 2（弱结果对；真谱比在 RQ6 macro）|
| MoE 2 个 | 见 RQ3 PLAN | < 1.5（MoE 异常）|

## 验收标准

| 模式 | `sigma_ratio` | `top_alignments[0].rank` (4096 维里) |
|---|:-:|:-:|
| A (10 个) | **≥ 3**（GPT-J ≥ 4.5）| **< 50**（MA 维度应是前几名）|
| 中间 (9 个) | 2 - 3 | < 200 |
| B (4 个) + MoE (2 个) | < 2 | 无严格要求（单层无用）|

**rank 字段解释**：对功能词 h₂ 最对齐的 hidden 维度，在 `u₁` 幅度排序中排第几。起源层应该排很前（rank < 50），说明"功能词对准的方向"和"输出大值的方向"一致——才能乘起来放大。peak 层这个 rank 普遍 > 1000。

## 执行成本

- 每模型 ~6-12 min（SVD 比 RQ3 多做一次矩阵分解）
- 23 个 ≈ **3h**（双卡 ~1.5h）
- 后续 llama2_13b + glm4_32b ≈ +45 min

## 输出

`results/wikitext_run/RQ4_origin/{model}/*.json`（保留旧 `RQ4/` 错层结果作对照）。
