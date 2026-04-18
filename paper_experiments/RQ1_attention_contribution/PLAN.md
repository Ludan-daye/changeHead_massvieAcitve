# RQ1 — Attention 消融测试（执行计划）

## 做什么

整层禁用所有 attention，测 MA 大小。否证"MA 来自 attention"，并按 ΔMA 符号分 generative / suppressive。

## 使用的脚本

| 脚本 | 参数 | 输出 |
|---|---|---|
| `exp1_feasibility_test.py` | `--model M --nsamples 30 --dataset wikitext --savedir DIR` | `baseline/results.json`, `all_heads_disabled/results.json`, `table1_rq1.json` |

`table1_rq1.json` 字段：`key_layer = peak_layer`（注意：这就是之前 bug 的源头，RQ3/4/5 **不要**读这个字段；要读 `exp2.critical_layer`）。

## 模型清单（25 个）

### 本轮需要重做：2 个（数据异常）

| 模型 | 原因 | 命令 |
|---|---|---|
| **qwen2_7b** | ΔTop1 = +Infinity（baseline 接近 0 致除零）| `python RQ1_attention_contribution/exp1_feasibility_test.py --model qwen2_7b --nsamples 60 --dataset wikitext --savedir results/wikitext_run/RQ1/qwen2_7b_fixed` |
| **glm4_32b** | baseline = Infinity（fp16 溢出）| `python RQ1_attention_contribution/exp1_feasibility_test.py --model glm4_32b --nsamples 30 --dataset wikitext --savedir results/wikitext_run/RQ1/glm4_32b_fp32`（lib/load_model.py 已自动为 glm4 启用 fp32） |

### 其他 23 个：**保留原结果、不动**

| 类别 | 数量 | 模型 |
|:-:|:-:|---|
| Generative (ΔMA < 0) | 17 | bloom_7b1, falcon_7b, gpt2, gptj_6b, glm4_9b, llama2_13b, llama3.1_8b, mistral_7b_v03, qwen1.5_14b, qwen2.5_0.5b, qwen3_0.6b, qwen3_1.7b, qwen3_4b, qwen3_8b, qwen3_30b_a3b, qwen3.5_9b, qwen3.5_27b |
| Suppressive (ΔMA > 0) | 7 | opt_6.7b (+250%), qwen2.5_7b (+266%), yi_9b (+27%), qwen3_14b (+85%), qwen3_32b (+59%), qwen3.5_35b_a3b (+5%), qwen2_7b (Inf → 待重测) |
| 待重做 | 2 | qwen2_7b, glm4_32b |

## 验收标准

| 模型 | 指标 | 阈值 |
|---|---|:-:|
| qwen2_7b | `delta_top1_pct` | 有限数字（非 Inf / NaN）|
| glm4_32b | `baseline/results.json`.top1 | 有限数字；peak_layer 非 0 |

## 执行成本

- qwen2_7b 重测：~15 min
- glm4_32b 重测：~1h（fp32 大模型）

**合计：~1.25h**
