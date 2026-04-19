# Origin Layer Finder — 起源层自动判定

> 用 1 条命令从 `ALL_EXPERIMENTS_SUMMARY_v2.json` 推导**每个模型的起源层**，供 RQ3 / RQ4 / RQ5 / RQ6 实验使用。
>
> 当前产出覆盖 **24 个模型**（JSON 里 5 个模型缺 exp2/exp2c 数据，跳过）。

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

RQ3 / RQ4 / RQ5 单层实验的结果**强依赖**于 `--layer_id` 传的层是否真的是 MA 起源层。

**历史上三种层号选择**：

| 方案 | 数据源 | 问题 |
|---|---|---|
| ❌ `exp1.peak_layer` | v1 的老脚本 `run_all_rq.sh` | MA **观测最大**层，非写入层 |
| ⚠️ `exp2.critical_layer` | v1 JSON 里的 per-layer 单层消融最大层 | 对 CONCENTRATED 模型可用；对 DISPERSED 严重失真 |
| ✅ `exp2c.l_origin_from_step1` | 贪心累积消融第 1 步砍掉的层 | **当前最优**，本工具的默认选择 |

例：glm4_9b 在方案 ⚠️ 下给出 L17，方案 ✅ 给出 L1——两个层差 16。用 L17 跑 RQ3 得到 Cohen's d=+0.24，用 L1 做 macro RQ5b 得到 ΔMA=−82%。差距由层选择决定。

详见 [`../docs/V2_ROOT_CAUSE.md`](../docs/V2_ROOT_CAUSE.md)。

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

**类别说明（`exp2c.category` 仅按 `steps_to_kill` 划分，不看消融 %）**：

| exp2c.category | steps_to_kill | 典型含义 |
|:-:|:-:|---|
| CONCENTRATED | = 1 | 单层主导，典型模式 A |
| FEW-SOURCE | 2 – 5 | 少数层共同主导 |
| DISPERSED | > 5 | 多层分散（典型模式 B，单层实验必弱）|

> 反例提醒：`total_drop_pct` **不参与分类** — 例如 qwen3.5_35b_a3b 总降仅 15.9% 仍标 FEW-SOURCE，
> 因为它在 3 步内耗尽消融预算（但每步贡献不够大）。

## 所有模型起源层

**直接看产出文件**（代码每次运行自动更新）：

- **人类可读表**：[`output/SUMMARY.md`](output/SUMMARY.md)（推荐，含覆盖统计 + 完整表 + 和 v1 对比）
- **程序化读取**：[`output/L_ORIGIN.json`](output/L_ORIGIN.json)（单层）/ [`output/ORIGIN_LAYERS_MACRO.json`](output/ORIGIN_LAYERS_MACRO.json)（macro）
- **Shell 脚本 source**：[`output/L_ORIGIN.sh`](output/L_ORIGIN.sh)
- **diff 报告**：[`output/compare_v1_vs_v2.txt`](output/compare_v1_vs_v2.txt)

> ⚠️ 本 README **不在此处列全表**——避免和自动产出漂移失同步。以 `output/` 下实际文件为准。

## 下游消费方式

### 方式 1：shell 脚本里 source

```bash
source paper_experiments/origin_layer/output/L_ORIGIN.sh
echo "${L_ORIGIN[gptj_6b]}"    # → 2

source paper_experiments/origin_layer/output/ORIGIN_LAYERS_MACRO.sh
echo "${ORIGIN_LAYERS_MACRO[qwen3_32b]}"    # → "5,6,40,41,42"
```

> ⚠️ **需要 bash 4+**。macOS 默认 bash 3.2 不支持关联数组（`declare -A`）。
> 如需在 macOS 上测，用 Homebrew 安装的 bash：`/opt/homebrew/bin/bash`。
> Linux 服务器默认 bash 4+，开箱即用。

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
