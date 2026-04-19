# Origin Layer Finder — 起源层自动判定

> **用 1 条命令产出 25+ 模型对应的起源层**，供 RQ3 / RQ4 / RQ5 / RQ6 实验使用。

## 目录

```
origin_layer/
├── README.md                        本文件
├── determine_origin_layer.py        主代码（从 exp2c 推导起源层）
├── run.sh                           一键运行脚本
└── output/                          ← 产出结果（每次重跑都更新）
    ├── SUMMARY.md                   人类可读汇总表
    ├── L_ORIGIN.json                单层起源：{model: L}
    ├── L_ORIGIN.sh                  单层起源 bash 关联数组（可 source）
    ├── ORIGIN_LAYERS_MACRO.json     macro 起源集合：{model: [L1,L2,...]}
    ├── ORIGIN_LAYERS_MACRO.sh       macro 起源 bash 关联数组
    └── compare_v1_vs_v2.txt         和旧 L_ORIGIN 的 diff
```

## 为什么要有这个工具

所有 RQ3 / RQ4 / RQ5 实验的准确性都取决于 `--layer_id` 传对。
老脚本 `run_all_rq.sh` 读的是 `peak_layer`（MA 观测最大层，错），
正确的应该是 `exp2c.l_origin_from_step1`（贪心消融第一步砍掉的层）。

**修一次层号，约 30 处指标异常自动变预期值**（见 `docs/V2_ROOT_CAUSE.md`）。

## 一条命令跑完

```bash
cd paper_experiments/origin_layer
bash run.sh
# 或者：python3 determine_origin_layer.py --dump-all
```

产出的 `output/` 里就是**所有模型的起源层**，任何实验脚本可以直接消费。

## 判定算法

```
输入: results/ALL_EXPERIMENTS_SUMMARY_v2.json 的 exp2c 字段
    exp2c 是"贪心累积消融"的结果，按层对 MA 的贡献排序
    关键字段：
      l_origin_from_step1     第 1 步砍掉的层（最大单层贡献）
      final_disabled_set      完整被砍掉的层序列
      category                CONCENTRATED / FEW-SOURCE / DISPERSED

输出:
    单层起源层 (RQ3/4/5-single 用):  L_ORIGIN[model] = l_origin_from_step1
    Macro 起源集合 (RQ5b/RQ6 用):    ORIGIN_LAYERS_MACRO[model] = final_disabled_set
                                    (DISPERSED 模型取前 50%，避免包含弱贡献层)
```

类别解读：

| exp2c.category | 消融步数 | 含义 | 单层实验预期 |
|:-:|:-:|---|:-:|
| CONCENTRATED | 1 步 ≥ 80% | 单层主导，典型模式 A | 强 |
| FEW-SOURCE | 2-5 步 | 少数层主导 | 中-强 |
| DISPERSED | > 5 步 | 多层分散，典型模式 B | **必弱**（需用 macro）|

## 所有模型的起源层（最新产出）

这张表**由代码自动生成**，具体在 [`output/SUMMARY.md`](output/SUMMARY.md)：

| 模型 | 单层起源 L_ORIGIN | Macro 集合 | 类别 |
|---|:-:|---|:-:|
| gptj_6b | **2** | — | (v1 回退，无 exp2c) |
| bloom_7b1 | 3 | — | (v1 回退) |
| falcon_7b | 3 | — | (v1 回退) |
| gpt2 | 3 | — | (v1 回退) |
| mistral_7b_v03 | 1 | — | (v1 回退) |
| opt_6.7b | 1 | — | (v1 回退) |
| llama3.1_8b | **1** | [0,1] | FEW-SOURCE |
| qwen2.5_0.5b | **0** | [0] | CONCENTRATED |
| qwen2.5_7b | **3** | [3] | CONCENTRATED |
| qwen2_7b | **3** | [3] | CONCENTRATED |
| qwen3_0.6b | **2** | [2] | CONCENTRATED |
| qwen3_1.7b | **2** | [0,1,2] | FEW-SOURCE |
| qwen3_4b | **6** | [6,15] | FEW-SOURCE |
| qwen3_8b | 6 | [5,6,15,24,25,28,29] | DISPERSED |
| qwen3_14b | 6 | [3,6,7,10,12,19,22] | DISPERSED |
| qwen3.5_9b | 22 | [6,10,18,19,22,23,25] | DISPERSED |
| qwen3.5_27b | 54 | [34,48,50,52,54,58] | DISPERSED |
| qwen3_32b | 6 | [5,6,40,41,42] | DISPERSED |
| glm4_9b | **1** | [0,1] | FEW-SOURCE |
| glm4_32b | **0** | [0] | CONCENTRATED |
| yi_9b | 8 | [8,24] | DISPERSED |
| qwen1.5_14b | 35 | [3,4,26,33,34,35,36] | DISPERSED |
| qwen3_30b_a3b (MoE) | 1 | [1,3,10,11,26] | DISPERSED |
| qwen3.5_35b_a3b (MoE) | **9** | [7,9,38] | FEW-SOURCE |

加粗表示**和 v1 旧层号显著不同**（见 `output/compare_v1_vs_v2.txt`）。

## 下游消费方式

### 方式 1：shell 脚本里 source

```bash
source paper_experiments/origin_layer/output/L_ORIGIN.sh
echo "${L_ORIGIN[gptj_6b]}"    # → 2

source paper_experiments/origin_layer/output/ORIGIN_LAYERS_MACRO.sh
echo "${ORIGIN_LAYERS_MACRO[qwen3_32b]}"    # → "5,6,40,41,42"
```

### 方式 2：Python 里 JSON 读取

```python
import json
L = json.load(open("paper_experiments/origin_layer/output/L_ORIGIN.json"))
print(L["gptj_6b"])     # 2

M = json.load(open("paper_experiments/origin_layer/output/ORIGIN_LAYERS_MACRO.json"))
print(M["qwen3_32b"])   # [5, 6, 40, 41, 42]
```

### 方式 3：复制到实验脚本

```bash
# 把 L_ORIGIN 数组粘贴到 run_rq345_origin_layer.sh 替换原数组
cat paper_experiments/origin_layer/output/L_ORIGIN.sh
```

## 重新跑

当 `ALL_EXPERIMENTS_SUMMARY_v2.json` 更新后（比如新加了模型的 exp2c 数据），重跑即可：

```bash
cd paper_experiments/origin_layer && bash run.sh
```

## 进阶命令

```bash
# 只看新旧 L_ORIGIN 对比
python3 determine_origin_layer.py --compare

# JSON 输出到 stdout（供 pipe）
python3 determine_origin_layer.py --json

# 自定义输出目录
python3 determine_origin_layer.py --dump-all /tmp/my_output
```

## 关联文档

- [`docs/V2_ROOT_CAUSE.md`](../docs/V2_ROOT_CAUSE.md) — 为什么要做这个工具
- [`docs/EXECUTION_PLAN.md`](../docs/EXECUTION_PLAN.md) — 主实验手册
- [`run_rq345_origin_layer.sh`](../run_rq345_origin_layer.sh) — 消费 L_ORIGIN 的主批处理脚本
