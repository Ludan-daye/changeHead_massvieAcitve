# 起源层自动判定结果（SUMMARY）

> 由 `determine_origin_layer.py` 从 `ALL_EXPERIMENTS_SUMMARY_v2.json` 的 `exp2c` 自动产出

## 所有模型的起源层

| 模型 | 单层起源 (L_ORIGIN) | Macro 起源层集合 | 类别 | 消融步数 | MA 总降 % |
|---|:-:|---|:-:|:-:|:-:|
| `bloom_7b1` | **3** | — | ? | None | — |
| `deepseek_v2_lite` | **—** | — | ? | None | — |
| `falcon_7b` | **3** | — | ? | None | — |
| `glm4_32b` | **0** | [0] | CONCENTRATED | 1 | 95.9% |
| `glm4_9b` | **1** | [0,1] | FEW-SOURCE | 2 | 90.7% |
| `gpt2` | **3** | — | ? | None | — |
| `gptj_6b` | **2** | — | ? | None | — |
| `llama2_13b` | **—** | — | ? | None | — |
| `llama2_7b_chat` | **—** | — | ? | None | — |
| `llama3.1_8b` | **1** | [0,1] | FEW-SOURCE | 2 | 90.3% |
| `mistral_7b_v03` | **1** | — | ? | None | — |
| `opt_6.7b` | **1** | — | ? | None | — |
| `qwen1.5_14b` | **35** | [3,4,26,33,34,35,36] | DISPERSED | 15 | 39.7% |
| `qwen2.5_0.5b` | **0** | [0] | CONCENTRATED | 1 | 97.1% |
| `qwen2.5_0.5b_optimized` | **—** | — | ? | None | — |
| `qwen2.5_7b` | **3** | [3] | CONCENTRATED | 1 | 89.5% |
| `qwen2.5_7b_old_nan` | **—** | — | ? | None | — |
| `qwen2_7b` | **3** | [3] | CONCENTRATED | 1 | 92.5% |
| `qwen3.5_27b` | **54** | [34,48,50,52,54,58] | DISPERSED | 12 | 55.6% |
| `qwen3.5_35b_a3b` | **9** | [7,9,38] | FEW-SOURCE | 3 | 15.9% |
| `qwen3.5_9b` | **22** | [6,10,18,19,22,23,25] | DISPERSED | 15 | 52.2% |
| `qwen3_0.6b` | **2** | [2] | CONCENTRATED | 1 | 92.8% |
| `qwen3_1.7b` | **2** | [0,1,2] | FEW-SOURCE | 3 | 89.8% |
| `qwen3_14b` | **6** | [3,6,7,10,12,19,22] | DISPERSED | 15 | 71.6% |
| `qwen3_30b_a3b` | **1** | [1,3,10,11,26] | DISPERSED | 10 | 75.4% |
| `qwen3_32b` | **6** | [5,6,40,41,42] | DISPERSED | 10 | 80.2% |
| `qwen3_4b` | **6** | [6,15] | FEW-SOURCE | 2 | 89.1% |
| `qwen3_8b` | **6** | [5,6,15,24,25,28,29] | DISPERSED | 15 | 57.7% |
| `yi_9b` | **8** | [8,24] | DISPERSED | 5 | 89.6% |

## 分类说明

| 类别 | 判定 | 含义 |
|:-:|:-:|---|
| CONCENTRATED | 1 步消 ≥ 80% | 单层主导（模式 A），起源层是 L_ORIGIN |
| FEW-SOURCE | 2-5 步消 ≥ 80% | 少数层主导，macro 集合是完整起源 |
| DISPERSED | > 5 步 | 多层分散（模式 B），单层实验必弱，真故事在 macro |

## 和 v1 L_ORIGIN 的对比

详见 `compare_v1_vs_v2.txt`。关键差异：

| 模型 | v1 | v2 | 类别 |
|---|:-:|:-:|:-:|
| `glm4_9b` | 17 | **1** | FEW-SOURCE |
| `qwen1.5_14b` | 2 | **35** | DISPERSED |
| `qwen3.5_35b_a3b` | 39 | **9** | FEW-SOURCE |
| `qwen3.5_9b` | 26 | **22** | DISPERSED |
| `qwen3_32b` | 43 | **6** | DISPERSED |
| `yi_9b` | 1 | **8** | DISPERSED |

## 使用这些数据的方式

```bash
# 方式一：直接 source 到 shell
source output/L_ORIGIN.sh
echo "${L_ORIGIN[gptj_6b]}"  # → 2

# 方式二：JSON 程序化读取
python3 -c "import json; d=json.load(open('output/L_ORIGIN.json')); print(d['gptj_6b'])"

# 方式三：粘贴 bash 数组到 run_rq345_origin_layer.sh
cat output/L_ORIGIN.sh  # 复制 declare -A 部分
```
