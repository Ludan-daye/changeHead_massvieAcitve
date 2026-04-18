# RQ5 — V 矩阵消融（终局因果验证，执行计划）

## 位置

**最后做**（依赖 RQ6 的 macro v₁ 给模式 B 用）。执行顺序：RQ1 → RQ2 → RQ3 → RQ4 → RQ6 → **RQ5**。

## 做什么

分两个版本：

### 5a — 单层 V 替换（所有 25 个模型）

在起源层对 `W_down = U Σ V^T`，把 V 替换为随机正交矩阵 V_rand，保留 U 和 Σ。测 MA 变化。

预期：
- 模式 A → ΔMA ≤ -85%（v₁ 方向是因果关键）
- 模式 B → ΔMA > -30%（单层 V 对 MA 贡献小，**弱结果是正确结果**）

### 5b — Macro v₁ 投影消除（仅模式 B 4 个）

对多个起源层 `L ∈ origin_layers`，同时做 `W_down_L_ablated = (I − v v^T) @ W_down_L`，其中 v 是 macro v₁（从 RQ6 的 Δh_macro SVD 得到）。测 MA 变化。

预期：模式 B → ΔMA ≤ -80%（macro v₁ 才是真的因果方向）。

## 使用的脚本

| 版本 | 脚本 | 关键参数 |
|:-:|---|---|
| 5a 单层 | `paper_experiments/RQ5_v_matrix_ablation/exp5_v_ablation.py` | `--model --layer_id L_ORIGIN --nsamples 30 --savedir` |
| 5b macro | `paper_experiments/RQ5_v_matrix_ablation/exp5_macro_v_ablation.py`（本轮新增）| `--model --origin_layers "L0,L1,...,Lk" --nsamples 30 --savedir` |

两个都通过 `run_rq345_origin_layer.sh` 封装调用。

## 模型清单

### 5a 单层 RQ5（23 个 + 依赖 2 个）

```bash
# 23 个已知起源层
bash run_rq345_origin_layer.sh "" rq5
```

| 模式 | 数量 | 模型 | 预期 ΔMA |
|:-:|:-:|---|:-:|
| A | 10 | bloom_7b1, falcon_7b, gptj_6b, llama3.1_8b, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, qwen3_1.7b, yi_9b | **≤ -85%**（Pilot gptj_6b 应 ≤ -95%）|
| 中间 | 9 | mistral_7b_v03, qwen1.5_14b, glm4_9b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3.5_9b, qwen3_30b_a3b, qwen3.5_27b | -60% ~ -85%（**偏 A 强，偏 B 弱，对应 RQ2c 分类**）|
| B | 4 | gpt2, opt_6.7b, qwen3_32b, qwen3.5_35b_a3b | > -30%（**弱是对的**，证明需要 macro 版本）|

### 5b macro RQ5（4 个模式 B，可能加 1-2 个真 B 候选）

```bash
bash run_rq345_origin_layer.sh "gpt2 opt_6.7b qwen3_32b qwen3.5_27b" rq5_macro
```

| 模型 | origin_layers | 预期 ΔMA |
|---|:-:|:-:|
| gpt2 | `"0,1,2,3,4,5"` | ≤ -80%（MA_CONCLUSIONS.md 4.7 已验证）|
| opt_6.7b | `"0,1,2,3"` | ≤ -80% |
| qwen3_32b | `"40,41,42,43,44,45"` | ≤ -75% |
| qwen3.5_27b | `"50,51,52,53,54,55"` | ≤ -80%（macro η=4.59 很强）|

**可选扩展**（若 RQ6 确认为真 B）：
- glm4_9b: `"15,16,17,18,19"` （macro η=4.33）
- qwen3.5_35b_a3b: `"37,38,39,40,41"`（MoE，预期中等）

## 验收标准

| 测试 | 指标 | 阈值 |
|---|---|:-:|
| 5a 模式 A | `delta_ma.top1_mean_pct` | **≤ -85%** |
| 5a 模式 B | `delta_ma.top1_mean_pct` | > -30%（弱）|
| 5b 模式 B macro | `delta_ma.top1_mean_pct` | **≤ -80%** |
| 5b 对比 5a 同模型 | macro 比 single 的 ΔMA 大 50% 以上 | — |

## 执行成本

| 子任务 | 时间 |
|:-:|:-:|
| 5a × 23 | ~3h（每个 ~8 min） |
| 5b × 4-6 | ~1h |
| **合计** | **~4h**（双卡 ~2.5h） |

## 输出

```
results/wikitext_run/
├── RQ5_origin/{model}/{model}_v_ablation_results.json       # 5a
└── RQ5_macro/{model}/{model}_macro_v_ablation_results.json  # 5b
```

## 特别注意

- **MoE 的 5a 单层结果预期会弱**，这是 MoE 异常的对照数据，不是失败
- **5b macro 需要 RQ6 macro-SVD 先跑完**（确认 `origin_layers` 范围合理）
- **信息分轨实证**（MA 消 -85% 但 PPL 不涨）是可选扩展，需要在脚本里加 `eval_ppl` 调用；本轮先不做
