# RQ6 — 多层聚合分析（Macro-SVD，执行计划）

## 位置

放在 RQ5 之前，给 RQ5 的模式 B 版本提供 macro v₁。执行顺序：RQ1 → RQ2 → RQ3 → RQ4 → **RQ6** → RQ5。

## 做什么

对**多个** MLP 层的净贡献 `Δh_L` 累加，得 `Δh_macro`，对它做 SVD 得 macro σ₁, macro v₁。解决"模式 B 多层协作"的 MA 起源问题。

子实验（3 个，本轮全跑）：

| 子实验 | 做什么 | 目的 |
|:-:|---|---|
| **6.1 macro-SVD** | 对 Δh_macro 做 SVD | 得 macro σ₁/σ₂，macro v₁ |
| **6.2 / 6.3 top-K** | 从 Δh_macro 减去/保留 top-K 奇异方向测 MA | 定量"几个主方向维持 MA" |
| **6.4 progressive**（= RQ2c）| 贪心累积消融 | 判 A / B / 混合模式 |

## 使用的脚本

| 子实验 | 脚本 | 关键参数 |
|:-:|---|---|
| 6.1 | `exp6_macro_svd_full.py` | `--model --origin_layers "L0,L1,..." --nsamples 20 --seqlen 512 --savedir` |
| 6.2 / 6.3 | `exp6_single_layer_activation.py` | `--model --layers_to_scan all --nsamples 30 --seqlen 1024 --savedir` |
| 6.4 | `exp6_progressive_ablation.py` | `--model --nsamples 30 --threshold_pct 10 --max_steps 20 --savedir` |

注：`exp6_macro_svd_full.py` 用的参数名是 `--origin_layers`（逗号分隔），**不是** `--aggregate_layers`。

## 模型清单（高优先 13 + 低优先 10 = 23 个）

### 🔥 高优先 13：模式 B 4 + 中间 9（2.5h，本轮核心产出）

用来把 9 个中间形态**最终分类**为 A 或 B。

| 模型 | origin_layers | 模式 | 预期 macro σ₁/σ₂ |
|---|:-:|:-:|:-:|
| **gpt2** | `"0,1,2,3,4"` | B | **≥ 3**（历史 3.48）|
| **opt_6.7b** | `"0,1,2,3"` | B | ≥ 2 |
| **qwen3_32b** | `"40,41,42,43,44,45"` | B | ≥ 2 |
| **qwen3.5_27b** | `"50,51,52,53,54,55"` | B | **≥ 4**（已测 4.59）|
| **qwen3.5_35b_a3b** (MoE) | `"37,38,39,40,41"` | B(MoE) | 1.0-1.5（异常对照） |
| mistral_7b_v03 | `"0,1,2,3"` | 中间 | 待测 |
| qwen1.5_14b | `"0,1,2,3,4"` | 中间 | 已 1.60（补 top-K） |
| qwen3_4b | `"3,4,5,6,7"` | 中间 | 已 1.72（补 top-K） |
| qwen3_8b | `"4,5,6,7,8"` | 中间 | 已 2.02（补 top-K） |
| qwen3_14b | `"4,5,6,7,8"` | 中间 | 已 2.00（补 top-K） |
| qwen3.5_9b | `"24,25,26,27,28"` | 中间 | 已 2.58（补 top-K） |
| **glm4_9b** | `"15,16,17,18,19"` | 中间→**真 B 候选** | 已 **4.33**（补 top-K）|
| qwen3_30b_a3b (MoE) | `"0,1,2,3,4"` | 中间(MoE) | 已 1.06（MoE 异常）|

### ⭐ 低优先 10：模式 A 补齐（1.5h，对照证据）

模式 A 单层已主导，macro 不会显著提升 σ₁/σ₂，正好作 A 的对照证据。

| 模型 | origin_layers | 预期 macro σ₁/σ₂ |
|---|:-:|:-:|
| bloom_7b1 | `"0,1,2,3,4,5"` | 1.5 - 3 |
| falcon_7b | `"0,1,2,3,4,5"` | 1.5 - 3 |
| gptj_6b | `"0,1,2,3,4,5"` | **≥ 4**（历史 5.74）|
| llama3.1_8b | `"0,1,2,3"` | 1.5 - 3 |
| qwen2.5_0.5b | `"0,1,2"` | 1.5 - 3 |
| qwen2.5_7b | `"0,1,2,3,4"` | 1.5 - 3 |
| qwen2_7b | `"0,1,2,3,4"` | 1.5 - 3 |
| qwen3_0.6b | `"0,1,2,3"` | 1.5 - 3 |
| qwen3_1.7b | `"0,1,2,3"` | 1.5 - 3 |
| yi_9b | `"0,1,2"` | 1.5 - 3 |

### ⏸ 依赖前置 2 个

- llama2_13b（等 RQ2b 跑完确定起源层）
- glm4_32b（等 RQ1/RQ2a fp32 修复）

## 命令

### 高优先 13 — 示例

```bash
# === 模式 B 4 个 + MoE 1 个 ===
for m in gpt2 opt_6.7b qwen3_32b qwen3.5_27b qwen3.5_35b_a3b; do
    case $m in
        gpt2)              OL="0,1,2,3,4" ;;
        opt_6.7b)          OL="0,1,2,3" ;;
        qwen3_32b)         OL="40,41,42,43,44,45" ;;
        qwen3.5_27b)       OL="50,51,52,53,54,55" ;;
        qwen3.5_35b_a3b)   OL="37,38,39,40,41" ;;
    esac
    python RQ6_single_layer_activation/exp6_macro_svd_full.py \
        --model "$m" --origin_layers "$OL" --nsamples 20 --seqlen 512 \
        --savedir "results/wikitext_run/RQ6/$m"
    python RQ6_single_layer_activation/exp6_single_layer_activation.py \
        --model "$m" --layers_to_scan all --nsamples 30 --seqlen 1024 \
        --savedir "results/wikitext_run/RQ6/$m"
done

# === 中间 8 个 ===
declare -A MID_OL=(
    [mistral_7b_v03]="0,1,2,3"
    [qwen1.5_14b]="0,1,2,3,4"
    [qwen3_4b]="3,4,5,6,7"
    [qwen3_8b]="4,5,6,7,8"
    [qwen3_14b]="4,5,6,7,8"
    [qwen3.5_9b]="24,25,26,27,28"
    [glm4_9b]="15,16,17,18,19"
    [qwen3_30b_a3b]="0,1,2,3,4"
)
for m in "${!MID_OL[@]}"; do
    OL="${MID_OL[$m]}"
    python RQ6_single_layer_activation/exp6_macro_svd_full.py \
        --model "$m" --origin_layers "$OL" --nsamples 20 --seqlen 512 \
        --savedir "results/wikitext_run/RQ6/$m"
    python RQ6_single_layer_activation/exp6_single_layer_activation.py \
        --model "$m" --layers_to_scan all --nsamples 30 --seqlen 1024 \
        --savedir "results/wikitext_run/RQ6/$m"
done
```

### 低优先 10 — 示例

```bash
declare -A A_OL=(
    [bloom_7b1]="0,1,2,3,4,5"  [falcon_7b]="0,1,2,3,4,5"
    [gptj_6b]="0,1,2,3,4,5"    [llama3.1_8b]="0,1,2,3"
    [qwen2.5_0.5b]="0,1,2"     [qwen2.5_7b]="0,1,2,3,4"
    [qwen2_7b]="0,1,2,3,4"     [qwen3_0.6b]="0,1,2,3"
    [qwen3_1.7b]="0,1,2,3"     [yi_9b]="0,1,2"
)
for m in "${!A_OL[@]}"; do
    OL="${A_OL[$m]}"
    python RQ6_single_layer_activation/exp6_macro_svd_full.py \
        --model "$m" --origin_layers "$OL" --nsamples 20 --seqlen 512 \
        --savedir "results/wikitext_run/RQ6/$m"
    python RQ6_single_layer_activation/exp6_single_layer_activation.py \
        --model "$m" --layers_to_scan all --nsamples 30 --seqlen 1024 \
        --savedir "results/wikitext_run/RQ6/$m"
done
```

## 验收标准（模式精细判定表）

| macro σ₁/σ₂ | remove top-1 ΔMA | 判定 |
|:-:|:-:|---|
| ≥ 3 | ≤ -80% | **真模式 B（多层协作）** |
| 2 – 3 | -50% ~ -80% | 中间倾 B |
| 1 – 2 | > -50% | 中间倾 A |
| ≈ 1 + MoE | 任意 | **MoE 异常（expert mark 不共享，Tier C）** |

**核心产出**：25 模型 × A / B / 中间-A / 中间-B / MoE 异常 **五分类表**——论文 Table 1 的核心数据。

## 执行成本

| 分组 | 时间（串行）| 时间（双卡）|
|:-:|:-:|:-:|
| 高优先 13 × 2 脚本 | 2.5h | 1.5h |
| 低优先 10 × 2 脚本 | 1.5h | 1h |
| **合计** | **~4h** | **~2.5h** |

## 输出

```
results/wikitext_run/RQ6/{model}/
├── {model}_macro_svd_full.json   # 6.1
└── {model}_rq6_results.json      # 6.2 / 6.3

results/wikitext_run/RQ2c/{model}/
└── {model}_rq6_greedy.json       # 6.4 = RQ2c，数据已在 RQ2 计划里跑
```
