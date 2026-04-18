# RQ2 — MLP 是 MA 源头 + 起源层定位（执行计划）

## 做什么

三个子实验：

| 子实验 | 做什么 | 目的 |
|:-:|---|---|
| **RQ2a** | 整层禁用所有 MLP | 证 MLP 是 MA 写入者 |
| **RQ2b** | 逐层禁用单个 MLP | 定位起源层 `critical_layer` |
| **RQ2c** | 贪心累积消融（按贡献排序依次累加）| 严格判定模式 A / B / 调节型 |

RQ2c 与 RQ6.4 progressive 是同一个实验，**一次跑、结果双用**。

## 使用的脚本

| 子实验 | 脚本 | 关键参数 | 输出 |
|:-:|---|---|---|
| RQ2a | `paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py` | `--model --nsamples 30 --dataset wikitext --savedir` | `baseline/results.json`, `all_mlp_disabled/results.json` |
| RQ2b | `changeHead_massvieAcitve/experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py`（**在老仓库**）| `--model --nsamples 20 --seqlen 4096 --savedir` | `baseline.json`, `layer_N_disabled.json` × N |
| RQ2c | `paper_experiments/RQ6_single_layer_activation/exp6_progressive_ablation.py` | `--model --nsamples 30 --threshold_pct 10.0 --max_steps 20 --savedir` | `{model}_rq6_greedy.json` |

## 模型清单

### RQ2a（20 个：19 未跑 + 1 异常需 fp32 修）

```bash
# 模式 B 4 个（最高优先，论文必需）
for m in gpt2 opt_6.7b qwen3_32b qwen3.5_35b_a3b; do
    python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
        --model "$m" --nsamples 30 --dataset wikitext \
        --savedir "results/wikitext_run/RQ2a/$m"
done

# 其余 15 个
for m in llama3.1_8b qwen1.5_14b qwen2.5_0.5b qwen2_7b \
         qwen3_0.6b qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b \
         qwen3.5_9b qwen3.5_27b yi_9b glm4_9b qwen3_30b_a3b llama2_13b; do
    python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
        --model "$m" --nsamples 30 --dataset wikitext \
        --savedir "results/wikitext_run/RQ2a/$m"
done

# glm4_32b（fp32 自动生效）
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
    --model glm4_32b --nsamples 30 --dataset wikitext \
    --savedir results/wikitext_run/RQ2a/glm4_32b
```

**已有数据（跳过）**：5 个 — gptj_6b, bloom_7b1, falcon_7b, mistral_7b_v03, qwen2.5_7b

### RQ2b（2 个：必须跑 + 异常修）

```bash
cd ../changeHead_massvieAcitve
python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py \
    --model llama2_13b --nsamples 20 --seqlen 4096 \
    --savedir "$PWD/../paper_experiments/results/wikitext_run/RQ2b/llama2_13b"

python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py \
    --model glm4_32b --nsamples 20 --seqlen 4096 \
    --savedir "$PWD/../paper_experiments/results/wikitext_run/RQ2b/glm4_32b"
cd -
```

**已有数据（跳过）**：23 个 — 所有其他模型的 RQ2b 数据已在 `ALL_EXPERIMENTS_SUMMARY.json` 里。

### RQ2c（23 个，全部新跑；数据同时填 RQ6.4 progressive）

```bash
for m in bloom_7b1 falcon_7b gptj_6b llama3.1_8b qwen2.5_0.5b qwen2.5_7b \
         qwen2_7b qwen3_0.6b qwen3_1.7b yi_9b mistral_7b_v03 qwen1.5_14b \
         qwen3_4b qwen3_8b qwen3_14b qwen3.5_9b glm4_9b qwen3_30b_a3b \
         gpt2 opt_6.7b qwen3_32b qwen3.5_27b qwen3.5_35b_a3b; do
    python RQ6_single_layer_activation/exp6_progressive_ablation.py \
        --model "$m" --nsamples 30 --seqlen 1024 \
        --threshold_pct 10.0 --max_steps 20 \
        --savedir "results/wikitext_run/RQ2c/$m"
done
```

**待做**：llama2_13b（等 RQ2b）、glm4_32b（等 RQ1 fp32）。

## 验收标准

| 子实验 | 指标 | 阈值 |
|:-:|---|:-:|
| RQ2a 模式 A | `delta_top1_pct`（禁全部 MLP）| **≤ -95%** |
| RQ2a 模式 B | `delta_top1_pct` | **≤ -90%**（聚合 MLP 几乎全部写入 MA） |
| RQ2b | 每模型 `critical_layer` 与 `max_reduction_pct` 与旧 JSON 对照一致 | — |
| RQ2c 累积曲线 | top-1 ≥ -85% → 模式 A<br>top-2 ≥ -95% 但 top-1 < 85% → 双层主导<br>top-5 才 ≤ -80% → 模式 B<br>top-K 非单调 → 有调节层 | 按此定性分 A/B/双/调 |

## 执行成本

| 子实验 | 时间 |
|:-:|:-:|
| RQ2a × 20 | ~1h（每个 ~3min） |
| RQ2b × 2 | ~40 min |
| RQ2c × 23 | ~2h（每个 ~5min） |
| **合计** | **~4h**（双卡并行 ~2h） |

## 输出目录

```
results/wikitext_run/
├── RQ2a/{model}/          # 禁全部 MLP
├── RQ2b/{model}/          # 逐层（仅 llama2_13b + glm4_32b 新增）
└── RQ2c/{model}/          # 累积消融（23 个）
```
