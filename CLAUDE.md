# CLAUDE.md

> 本文件面向 Claude Code 和任何接手开发者。记录当前所有实验进度、代码工具、执行流程、关键 gotcha。
> **最后一次更新：2026-04-20**（新增 §16 本轮分析进度：RQ1 H₀ 证伪完成 + 26 模型基数 + RQ2 数据盘点）

---

## 1. 项目概览

研究 **Massive Activations (MAs)** — LLM 和 ViT 中的极端激活值（300–3000× 中位数）。

**核心论点**（本轮已用 25+ 模型数据支撑）：

> MA 是 MLP 在功能词位置写入的"语法重音 mark"，经 attention 广播并由模型调节为稳态。

完整机制详见根目录 [`README.md`](README.md) §I-III 的**五步机制链**。

---

## 2. 仓库结构（已整理）

```
ma/
├── README.md                        ← 根 README：机制讲解 + Quick Start
├── CLAUDE.md                        ← 本文件
├── .gitmodules                      ← 登记 changeHead_massvieAcitve 为 submodule
├── .gitignore
│
├── paper_experiments/               ← ★ 新开发主线（本轮工作焦点）
│   ├── README.md
│   ├── setup.sh                     ← 环境 + 依赖安装
│   ├── requirements.txt
│   ├── run_rq345_origin_layer.sh    ← ★ 主批处理：跑 RQ3/4/5 在起源层
│   ├── main_llm.py, main_vit.py     统一入口
│   │
│   ├── origin_layer/                ← ★ 起源层自动判定工具（2026-04-19 新增）
│   │   ├── README.md
│   │   ├── determine_origin_layer.py
│   │   ├── run.sh
│   │   └── output/                  产出：24 模型起源层（单层 + 多层）
│   │       ├── SUMMARY.md
│   │       ├── L_ORIGIN.json / .sh
│   │       ├── ORIGIN_LAYERS_MACRO.json / .sh
│   │       └── compare_v1_vs_v2.txt
│   │
│   ├── docs/                        ← 理论 + 进度追踪
│   │   ├── MA_FRAMEWORK.md          三部分框架（生成/传递/调节）
│   │   ├── MA_WHY.md                理论解释 15 章
│   │   ├── MA_CONCLUSIONS_AND_ARGUMENTS.md
│   │   ├── EXPERIMENT_PLAN.md       逐 RQ 讨论记录 + 机制附录
│   │   ├── EXECUTION_PLAN.md        ★ 主执行手册（批次顺序 + 命令 + 验收）
│   │   ├── PROGRESS_MATRIX.md       25 模型 × 6 RQ 状态矩阵
│   │   ├── TODO_EXPERIMENTS.md      待办清单
│   │   ├── V2_ROOT_CAUSE.md         ★ 根因诊断（错层是主因）
│   │   ├── FINDINGS_TRACKING.md
│   │   ├── CONFLICTS.md
│   │   └── CONCLUSIONS.md
│   │
│   ├── RQ1_attention_contribution/  每个 RQ 一个文件夹
│   │   ├── PLAN.md                  ★ 本 RQ 的执行计划
│   │   ├── README.md
│   │   └── exp1_feasibility_test.py
│   ├── RQ2_mlp_source/
│   │   ├── PLAN.md
│   │   ├── exp2a_mlp_feasibility_test.py    （禁全部 MLP）
│   │   └── exp2c_mlp_internal_analysis.py   （up vs down 分析）
│   │   注: RQ2b (逐层消融) 脚本在老仓库 changeHead_massvieAcitve/
│   ├── RQ3_function_words/
│   │   ├── PLAN.md
│   │   └── exp5_function_words_svd_mapping.py
│   ├── RQ4_svd_alignment/
│   │   ├── PLAN.md
│   │   └── exp3_svd_alignment_analysis.py
│   ├── RQ5_v_matrix_ablation/
│   │   ├── PLAN.md
│   │   ├── exp5_v_ablation.py                单层 V 替换随机正交
│   │   ├── exp5_macro_v_ablation.py          ★ 多层 macro v₁ 投影消除（本轮新增）
│   │   ├── exp5_mock_validation.py
│   │   └── exp5_validation_report.py
│   ├── RQ6_single_layer_activation/
│   │   ├── PLAN.md
│   │   ├── exp6_macro_svd_full.py            macro-SVD 完整版
│   │   ├── exp6_macro_svd_gpt2.py
│   │   ├── exp6_single_layer_activation.py   top-K 删/留
│   │   ├── exp6_progressive_ablation.py      ★ RQ2c ≡ RQ6.4 贪心累积消融
│   │   ├── exp6_exhaustive_gpt2.py
│   │   ├── exp6_exhaustive_parallel.py
│   │   └── exp6_fast_exhaustive.py
│   ├── RQ7_training_dynamics/       ← Pythia MA 演化（本轮排除，不跑）
│   │   └── exp7_pythia_ma_evolution.py
│   │
│   ├── lib/                         共享库
│   │   ├── model_dict.py            ★ 已加 hf_fallback 和 resolve_model_id()
│   │   ├── load_model.py            ★ 已加 glm4 fp32 分支
│   │   ├── load_data.py
│   │   ├── hook.py
│   │   ├── eval_utils.py            ★ eval_ppl 已参数化（支持 device/seqlen）
│   │   └── plot_utils_*.py
│   ├── monkey_patch/                activation 捕获 hook
│   └── results/
│       ├── wikitext_run/RQ1..RQ7/   实验输出
│       ├── ALL_EXPERIMENTS_SUMMARY_v2.json  ★ 29 模型 v2 结果
│       └── ...
│
├── changeHead_massvieAcitve/        ← 老代码库（submodule）
│   └── experiments/exp2_mlp_layers/
│       └── exp2b_mlp_layer_ablation.py   ★ 逐层 MLP 消融（只有这里有）
│
├── paper/                           论文
│   ├── Function Words as Geometric Anchors.pdf
│   ├── acl_source/                  ACL LaTeX
│   └── notes_zh/                    中文笔记、rebuttal、审稿
│
├── figures/                         实验输出图
├── archives/                        zip 归档
├── scripts/                         顶层可视化脚本
└── results_per_doc/                 per-doc 独立采样实验结果
```

---

## 3. 实验体系（6 个 RQ，RQ7 排除）

每个 RQ 的详细 PLAN 在 `paper_experiments/RQ{N}/PLAN.md`。

### 3.1 RQ1 — Attention 消融（基本就绪）

- **做什么**：整层禁用所有 attention，测 MA 变化
- **判 Gen/Sup**：ΔMA < 0 → generative（17 模型）；ΔMA > 0 → suppressive（7 模型）
- **脚本**：`RQ1_attention_contribution/exp1_feasibility_test.py`
- **本轮状态**：23/25 完成。2 个数据异常待修：
  - `qwen2_7b` ΔTop1 = +∞（baseline 接近 0 除零）→ 改 `--nsamples 60`
  - `glm4_32b` baseline = Infinity（fp16 溢出）→ lib 已加 fp32 自动分支

### 3.2 RQ2 — MLP 来源（三子实验）

- **RQ2a 全禁 MLP**：正面证明 MLP 是来源 → 目前全 null 待补 19 + glm4_32b
- **RQ2b 逐层消融**：找起源层 → 23/25 已跑；llama2_13b + glm4_32b 待补
  - ⚠️ 脚本在**老仓库** `changeHead_massvieAcitve/experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py`
- **RQ2c 贪心累积消融**（新增）：严格区分 A/B/调节型 → 18/25 已跑
  - 脚本 = `exp6_progressive_ablation.py`（合并复用）
  - 产出 `category`（CONCENTRATED / FEW-SOURCE / DISPERSED）+ `l_origin_from_step1` + `final_disabled_set`

### 3.3 RQ3 — 功能词 SVD 映射

- **做什么**：起源层的 h₂ 在 W_down 的 v₁ 上投影，比较功能词 vs 内容词的 Cohen's d
- **脚本**：`RQ3_function_words/exp5_function_words_svd_mapping.py`
- **当前状态**：v2 里 18 模型有 exp3 数据，但**全部在 peak 层**跑，9 个 Cohen's d 为负（错层证据）
- **本轮要做**：25 模型全部在**起源层**重跑（由 `origin_layer/output/L_ORIGIN.sh` 提供层号）

### 3.4 RQ4 — SVD 几何对齐

- **做什么**：起源层对 W_down 做 SVD，测 σ₁/σ₂（谱比）+ top_alignments
- **脚本**：`RQ4_svd_alignment/exp3_svd_alignment_analysis.py`
- **当前状态**：18 模型有数据，全部 σ₁/σ₂ < 3（模式 A 应 ≥ 3，典型错层证据）
- **本轮要做**：25 模型全部起源层**单层**（非 ± 2 的 5 层）重跑

### 3.5 RQ5 — V 矩阵消融（终局因果验证）

- **5a 单层**：`exp5_v_ablation.py` — 替 V 为随机正交矩阵
- **5b 多层 macro**（新增）：`exp5_macro_v_ablation.py` — 投影掉 macro v₁ 方向
- **当前状态**：v2 里 11 模型有 exp5b，**8 个 strong**（ΔMA ≤ -80%），3 个弱（可能 DISPERSED 的 origin_layers 选择问题）
- **执行顺序**：**最后做**，因为依赖 RQ6 的 macro v₁

### 3.6 RQ6 — Macro-SVD（多层聚合，本轮重点）

- **6.1 macro-SVD**：`exp6_macro_svd_full.py` — 跨多层 Δh 累加做 SVD
- **6.2/6.3 top-K**：`exp6_single_layer_activation.py` — 删/保留 top-K 测 MA
- **6.4 progressive**（= RQ2c）：`exp6_progressive_ablation.py`
- **当前状态**：18/25 已有 macro σ₁/σ₂ 数据
- **本轮要做**：高优 13（4 模式 B + 9 中间形态）+ 低优 10（模式 A 对照）

---

## 4. 本轮新增的核心工具（2026-04-18/19）

### 4.1 `paper_experiments/origin_layer/` — 起源层自动判定

**问题**：RQ3/4/5 `--layer_id` 传错会让全部指标失真。手动维护层号容易漂移。

**方案**：一键从 `ALL_EXPERIMENTS_SUMMARY_v2.json` 的 `exp2c` 字段自动推导。

**使用**：
```bash
cd paper_experiments/origin_layer
bash run.sh
# 产出 output/：
#   L_ORIGIN.json      单层起源（24 模型）
#   L_ORIGIN.sh        单层 bash 关联数组（可 source）
#   ORIGIN_LAYERS_MACRO.json  多层起源集合（24 模型）
#   ORIGIN_LAYERS_MACRO.sh    多层 bash 数组
#   SUMMARY.md         人类可读汇总
#   compare_v1_vs_v2.txt   新旧层号对比
```

**判定规则**：
```
单层 L_ORIGIN:
  优先级 1: exp2c.l_origin_from_step1
  优先级 2: exp2.critical_layer (v1 fallback)
  优先级 3: None → 不列入输出

多层 ORIGIN_LAYERS_MACRO:
  优先级 1: exp2c.final_disabled_set
     DISPERSED: 取前 50%（按 greedy 顺序）
     否则: 全部
  优先级 2: 启发式窗口（有 exp2.critical_layer L 时）
     L ≤ 5: [0..5]
     L > 5: [L-2..L+2]
  优先级 3: None
```

**当前覆盖**：
- 单层：24 模型（18 精确 exp2c + 6 v1-fallback）
- 多层：24 模型（18 精确 + 6 启发式窗口）
- 跳过 5 模型（完全没数据）：deepseek_v2_lite, llama2_13b, llama2_7b_chat, qwen2.5_0.5b_optimized, qwen2.5_7b_old_nan

### 4.2 `paper_experiments/run_rq345_origin_layer.sh` — 主批处理脚本

**已修复**：现在自动 source `origin_layer/output/L_ORIGIN.sh` 和 `ORIGIN_LAYERS_MACRO.sh`，不再硬编码层号。

**使用**：
```bash
bash run_rq345_origin_layer.sh "gptj_6b" all           # Pilot
bash run_rq345_origin_layer.sh "" rq3                  # 全模型 RQ3
bash run_rq345_origin_layer.sh "" all                  # 全模型 RQ3+4+5
bash run_rq345_origin_layer.sh "gpt2 opt_6.7b" rq5_macro  # macro v₁
```

### 4.3 `paper_experiments/RQ5_v_matrix_ablation/exp5_macro_v_ablation.py` — 新脚本

多层 macro v₁ 投影消除。解决单层 v_ablation 对模式 B 模型无效的问题。

算法：
1. 捕获 `Δh_macro = h_after_last - h_before_first` 跨 origin_layers
2. SVD(Δh_macro) → macro v₁（hidden space 方向）
3. 对 origin_layers 中每层：`W_down_ablated = (I - v v^T) @ W_down`
4. 测 MA 变化

预期模式 B 的 ΔMA ≤ -80%。

---

## 5. 25 模型本轮进度快照

详见 [`paper_experiments/origin_layer/output/SUMMARY.md`](paper_experiments/origin_layer/output/SUMMARY.md)。

**模式分布**（基于 exp2c.category）：
- CONCENTRATED（单层主导）：6 模型（glm4_32b, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b）
- FEW-SOURCE（2-5 层主导）：6 模型（glm4_9b, llama3.1_8b, qwen3_1.7b, qwen3_4b, qwen3.5_35b_a3b, yi_9b）
- DISPERSED（>5 层分散）：7 模型（qwen1.5_14b, qwen3.5_27b, qwen3.5_9b, qwen3_14b, qwen3_30b_a3b, qwen3_32b, qwen3_8b）
- 待分类（无 exp2c）：6 模型（bloom_7b1, falcon_7b, gpt2, gptj_6b, mistral_7b_v03, opt_6.7b）
- 数据缺失：5 模型（见 4.1 列表）

**关键层号更新**（v1 → v2）：

| 模型 | v1 critical_layer | v2 exp2c step1 | 备注 |
|---|:-:|:-:|---|
| qwen3_32b | 43 | **6** | DISPERSED，真 origin 早 |
| qwen3.5_35b_a3b | 39 | **9** | FEW-SOURCE |
| glm4_9b | 17 | **1** | FEW-SOURCE，真 B 候选 |
| yi_9b | 1 | 8 | DISPERSED |
| qwen3.5_9b | 26 | 22 | DISPERSED |

---

## 6. 环境与部署

### 6.1 本地开发者（有模型权重）

```bash
cd paper_experiments
bash setup.sh    # 创建 conda env + 装依赖
# 如果本地有 /model/ 路径的模型，保持 lib/model_dict.py 不变
```

### 6.2 外部部署（无本地权重）

```bash
git clone --recurse-submodules https://github.com/Ludan-daye/changeHead_massvieAcitve.git ma
cd ma/paper_experiments
bash setup.sh

# lib/model_dict.py 的每条都有 hf_fallback → 自动 HF 下载
# resolve_model_id() 若本地路径不存在会自动回退到 HF
# 20 个公开模型可直接跑；5 个（qwen3.5 系列 + glm4_32b）无公开 HF，需自备权重
```

### 6.3 依赖注意

- `requirements.txt` 已加 `scikit-learn`（RQ6 依赖）和 `scipy`（RQ3 依赖）
- `transformers==4.36.0`、`torch>=2.0.0`、`timm==0.9.12`
- `determine_origin_layer.py` 只用标准库
- bash 4+ 才支持关联数组（macOS 默认 bash 3.2 不行，用 `/opt/homebrew/bin/bash`）

---

## 7. 典型执行流程

```bash
# 1. 起源层判定（每次 ALL_EXPERIMENTS_SUMMARY_v2.json 更新后重跑）
cd paper_experiments/origin_layer && bash run.sh

# 2. Pilot 验证（GPT-J 单模型，~30 min）
cd .. && bash run_rq345_origin_layer.sh "gptj_6b" all
# 验收：RQ5 ΔMA ≤ -85%

# 3. 批量跑 RQ3 / RQ4（24 模型，~5h 双卡）
bash run_rq345_origin_layer.sh "" rq3
bash run_rq345_origin_layer.sh "" rq4

# 4. 跑 RQ6 macro-SVD（需分批，见 docs/EXECUTION_PLAN.md §批次 3）

# 5. RQ5 最后跑（依赖 RQ6）
bash run_rq345_origin_layer.sh "" rq5
bash run_rq345_origin_layer.sh "" rq5_macro

# 6. 补洞（llama2_13b、glm4_32b、qwen2_7b 等）
# 见 docs/EXECUTION_PLAN.md §批次 7
```

---

## 8. 关键数学（论文核心）

### 8.1 MA 生成公式

```
MA ≈ σ₁ · |h₂ᵀ · v₁| · max_j|(u₁)_j| + bias
     └─┘   └──────┘   └──────────┘
     RQ4    RQ3         RQ4
     谱集中  功能词对准    输出堆到主维度
```

三件事**同时满足**才有 MA：
1. W_down 的谱集中（σ₁/σ₂ ≥ 3）
2. 功能词的 h₂ 在 v₁ 方向对齐
3. u₁ 集中在少数 hidden 维度

### 8.2 两种生成模式

- **模式 A 单层主导**（CONCENTRATED / FEW-SOURCE）：1-2 层 MLP 完成全部 MA 写入
- **模式 B 多层协作**（DISPERSED）：5+ 层 MLP 同向接力；需用 macro-SVD 分析

### 8.3 两个正交维度（见 README §II 步骤 4）

```
生成模式（A/B）         × 调节模式（Generative/Suppressive）
└── RQ2b + RQ2c + RQ6        └── RQ1
```

4 种组合，25 模型分别落位。

---

## 9. 本轮重要文档索引

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 根 README — 5 步机制链 + Quick Start |
| [paper_experiments/docs/EXECUTION_PLAN.md](paper_experiments/docs/EXECUTION_PLAN.md) | ⭐ 主执行手册（批次 + 命令 + 验收）|
| [paper_experiments/docs/EXPERIMENT_PLAN.md](paper_experiments/docs/EXPERIMENT_PLAN.md) | 每个 RQ 的讨论记录 + 机制附录 |
| [paper_experiments/docs/V2_ROOT_CAUSE.md](paper_experiments/docs/V2_ROOT_CAUSE.md) | ⭐ 根因诊断：错层是主因 |
| [paper_experiments/docs/PROGRESS_MATRIX.md](paper_experiments/docs/PROGRESS_MATRIX.md) | 25 模型 × 6 RQ 状态矩阵 |
| [paper_experiments/docs/TODO_EXPERIMENTS.md](paper_experiments/docs/TODO_EXPERIMENTS.md) | 每模型待办 |
| [paper_experiments/origin_layer/README.md](paper_experiments/origin_layer/README.md) | 起源层判定工具 |
| [paper_experiments/origin_layer/output/SUMMARY.md](paper_experiments/origin_layer/output/SUMMARY.md) | ⭐ 24 模型起源层汇总 |
| [paper_experiments/RQ1-6/PLAN.md](paper_experiments/RQ1_attention_contribution/PLAN.md) | 每个 RQ 的独立执行计划 |

---

## 10. 支持的模型（v2 JSON 里的 29 个 + 少数 ViT）

### LLM（本轮重点，26 = 25 核心 + 1 辅助）

**25 核心**：gpt2, gptj_6b, opt_6.7b, bloom_7b1, falcon_7b, mistral_7b_v03, llama2_13b, llama3.1_8b, mistral_7b_v03, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, qwen3_1.7b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_30b_a3b (MoE), qwen3_32b, qwen3.5_9b, qwen3.5_27b, qwen3.5_35b_a3b (MoE), yi_9b, qwen1.5_14b, glm4_9b, glm4_32b

**1 辅助**（有 exp1 但缺 exp2 链）：llama2_7b_chat

**3 占位/无数据**（本轮不分析）：deepseek_v2_lite, qwen2.5_0.5b_optimized, qwen2.5_7b_old_nan

### ViT
MAE（base/large/huge）、CLIP、DINOv2、DINOv2-reg

---

## 11. 关键 Gotcha（容易踩的坑）

| 问题 | 解决 |
|---|---|
| RQ3/4/5 层号给错（peak 而非 origin） | **必须用 `origin_layer/output/L_ORIGIN.sh` 的层** |
| RQ2b 脚本在哪？ | **老仓库** `changeHead_massvieAcitve/experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py` |
| exp5 macro mode | 用 **`exp5_macro_v_ablation.py`**，不是 exp5_v_ablation |
| RQ2c = RQ6.4 progressive | **同一个脚本**，结果文件可双用 |
| glm4 fp16 溢出 | `lib/load_model.py` 已自动切 fp32（检测 "glm4"）|
| qwen2_7b RQ1 = +∞ | nsamples 30→60 重跑 |
| macOS bash 3.2 不支持 declare -A | 用 `/opt/homebrew/bin/bash` |
| `run_rq345_origin_layer.sh` 硬编码过层号（已修） | 现在自动 source `origin_layer/output/`，不用再改脚本 |

---

## 12. 下一步待办（参考 EXECUTION_PLAN 批次）

| 批次 | 内容 | 预估时间 |
|:-:|---|:-:|
| 批次 0 | 数据异常修 (qwen2_7b RQ1、glm4_32b fp32、llama2_13b RQ2b) | 1h |
| 批次 1 | Pilot GPT-J 全套 RQ3/4/5 | 30 min |
| 批次 2 | RQ2a 补 20 + RQ2c 跑 23 | 2h |
| 批次 3 | RQ6 高优 13（中间 + 模式 B） | 2.5h |
| 批次 4 | RQ3 + RQ4 全 24 起源层重跑 | 5h（双卡 2.5h）|
| 批次 5 | RQ6 低优 10（模式 A 对照） | 1.5h |
| 批次 6 | RQ5 全部（含 4-6 macro） | 4h |
| 批次 7 | 补洞（llama2_13b 全链、glm4_32b 全链）| 2h |
| **总计** | | **~18.5h 单卡 / ~12h 双卡** |

---

## 13. Git 分支

- `main` — 主开发分支（含全部最新工作，已从 local_ma 合并而来，无匿名限制）
- `local_ma` — 历史开发分支，已 merge 到 main
- 其他旧分支 ACL 匿名化提交**保留在 history 里**但被 local_ma 覆盖

---

## 14. 语言

文档**中英混合**：
- 代码注释：多为英文
- 论文：英文
- `paper/notes_zh/`：中文笔记、rebuttal
- `docs/*.md` 和 `origin_layer/*.md`：中文

---

## 15. 联系 / 协作

所有讨论记录在 `paper_experiments/docs/EXPERIMENT_PLAN.md`。
所有待办记录在 `paper_experiments/docs/TODO_EXPERIMENTS.md`。
根因诊断在 `paper_experiments/docs/V2_ROOT_CAUSE.md`。

对 Claude Code 的提示：修改代码时优先考虑用 `origin_layer/` 产出的层号，不要硬编码；所有层依赖的脚本通过 source 消费，而不是复制粘贴。

---

## 16. 本轮分析进度（2026-04-20）

### 16.1 模型基数统一：26

本轮起把模型数统一定为 **26**：25 核心（见 §10）+ 1 辅助（`llama2_7b_chat`）。

- JSON `ALL_EXPERIMENTS_SUMMARY_v2.json` 里共 29 个 key
- 去掉 2 个**完全无数据占位**（`qwen2.5_0.5b_optimized`、`qwen2.5_7b_old_nan`）+ 1 个**重复条目**（`deepseek_v2_lite` 无核心实验数据）
- 余下 26 个真实模型作为本轮分析基数

### 16.2 分析工作流程约定

用户要求的工作流（**不创建新文档，整合进已有文档**）：

1. **讨论优先**：先口头分析、对齐判读口径
2. **等用户确认**后再写入文档
3. **首选目标文档**：`paper_experiments/docs/EXPERIMENT_PLAN.md`（逐 RQ 讨论记录的**唯一归宿**）
4. 不要新建 `RQ{N}_ANALYSIS.md` 之类的并行文件——会造成文档碎片化

### 16.3 RQ1 分析已完成（2026-04-20）

**重新定位**：RQ1 的目的不是给模型分 Generative/Suppressive，而是**证伪 H₀"attention 是 MA 起源"**。

**两条结论记录**：

1. **主结论**（H₀ 证伪）：**atten_h 不是 MA 的起源**
   - 26 个模型中有数据的 25 个，关 attention 后 MA 都有**残留**
   - 最小残留 `residual% = disabled_top1 / baseline_top1 × 100` = **1.69%**（gptj_6b）
   - **没有任何一个模型归零**——MA 必须还有另一个来源（→ RQ2 指向 MLP）
2. **副结论（影响方向）**：
   - **Generative (ΔMA<0)**：17 个——attention 是下游**放大器/广播器**
   - **Suppressive (ΔMA>0)**：8 个——attention 是下游**抑制器/稳态器**
   - 数据缺失：1 个（`qwen2_7b` baseline≈0 除零→+Inf，需 `--nsamples 60` 重跑）

**4 个观察**（已写入 `EXPERIMENT_PLAN.md` RQ1 节）：

1. **同家族翻转**：qwen2.5_7b Sup vs qwen2.5_0.5b Gen；glm4_32b Sup vs glm4_9b Gen——size 与训练策略同时变化
2. **baseline 相关性**：baseline 越大越容易 suppressive（大模型 attention 倾向收束）
3. **Suppressive 集群在中国开源家族**：Qwen 4/13、Yi 1/1、GLM 1/2；西方族（GPT/BLOOM/Falcon/Mistral/Llama-base）全为 Gen
4. **MoE 弱响应**：qwen3.5_35b_a3b ΔMA=+5%，整层 attention 消融对 MoE 路由影响小，需 PE-level 重跑

**指标定义**：
- `residual% = disabled_top1 / baseline_top1 × 100`（主证伪指标，表征"MA 还剩多少"）
- `ΔMA% = (disabled − baseline) / baseline × 100`（方向指标，决定 Gen/Sup）

**文档入口**：`paper_experiments/docs/EXPERIMENT_PLAN.md` §RQ1（行号 21–114）。

### 16.4 RQ2 分析已完成（2026-04-20）

**详见** → [`paper_experiments/docs/EXPERIMENT_PLAN.md` §RQ2](paper_experiments/docs/EXPERIMENT_PLAN.md#rq2--mlp-来源与起源层定位)。

**一句话摘要**：
- **主结论**：MLP 是 MA 主要来源（H₁ 验证）——24/26 已测，20/24 retain ≤ 10%，bloom_7b1 归零（100% reduce）
- **4 个残留异常**（retain > 15%）：qwen3.5_35b_a3b 81%（MoE 脚本工件）、gpt2 39%（小模型老架构）、qwen3.5_9b 32%、qwen3.5_27b 20%
- **家族级新发现**：qwen3.5 家族 3/3 全体 retain > 15%，与 qwen3（1-8%）形成强反差，提示 qwen3.5 有非 MLP 源
- **模式分布**（RQ2c）：CONCENTRATED 8 / FEW-SOURCE 8 / DISPERSED 8 / ANOMALY 1(opt_6.7b)
- **缺口**：仅 3 模型零散缺口（llama2_13b RQ2a+2b、opt_6.7b RQ2a、llama2_7b_chat RQ2b+2c），合计 ~45 min
- **RQ2b critical_layer ≠ RQ2c L_origin**：这是 V2 错层的根因，下游 RQ3/4/5 必须以 RQ2c.L_origin 为准

### 16.5 RQ3 + RQ4 暂停（2026-04-20）——**论点从"功能词"重定位到"结构 token"**

**gpt2 Top-K 验证的震撼发现**：在起源层 L3（σ₁/σ₂=3.05 强谱），按 |h₂·v₁| 排序的 Top-10：
- 整体 token 里 40% 是功能词
- **Top-10 只有 1/10 是功能词**（是 `' .'` 句号）
- Top-1 是 `'\n\n'`（换行符），MA=165.88，比第 2 名高 10×
- Top-10 主要构成：换行 / 标点 / @ 符号 / 日文字符 / 罕见内容词

**结论**：MA 不在"语法功能词"位置，而在**结构 token**——包括换行、标点、特殊符号、部分功能词。和学界 attention sink 研究（Xiao et al.）一致。

**影响**：
1. **RQ3 和 RQ4 都要重做**——不只是修 bug，**论点本身要重定位**
2. **RQ3 旧 bug 仍需修**：脚本只存功能词 + MoE 不支持 + get_mlp_submodules 白名单（3 个 bug 合并修）
3. **RQ4 分析指标换**：从"Cohen's d 平均差异"换成"Top-K 里结构 token 占比"
4. **论文主论点修正**：**"MA = MLP 在结构 token 位置写的 mark"**（不是单纯"功能词 mark"）

**重做成本**（见 `EXPERIMENT_PLAN.md §全局补跑清单 RQ4 节`）：
- 任务 1 Top-K 验证扩到 23 模型 ~15 min（不需重跑模型）
- 任务 2 定义"结构 token"词表 ~10 min
- 任务 3 修 RQ3 脚本 ~15 min
- 任务 4 RQ3+RQ4 全 26 模型重跑 ~2h
- 任务 5 RQ4 分析重写 ~30 min
- **合计 ~3.5h**

### 16.6 MoE（多专家机制）单独归类（Tier C）——2026-04-20 定案

本轮 26 模型里有 2 个 MoE：
- **qwen3_30b_a3b**（`Qwen3MoeSparseMoeBlock`，30B 总，3B 激活）
- **qwen3.5_35b_a3b**（`Qwen3_5MoeSparseMoeBlock`，35B 总，3B 激活）

**决定**：MoE **不纳入主结论统计**——24 个 dense 模型作为主样本。MoE 单独在论文附录讨论。

**原因**：MoE 的 MA 生成走**专家级（per-expert）机制**，和 dense MLP 的"整层单 v₁"机制**本质不同**：
- 每个 token 只激活 K=2~8 个专家（总共 64~256 个）
- 每个专家有独立的 W_up/W_down
- 功能词可能走特定专家，内容词走别的专家
- 整层平均 v₁ **会稀释**真正的 mark 方向

**脚本系统性不兼容（4 类 bug）**：
- B1：直接访问 `.up_proj / .down_proj` 在 `SparseMoeBlock` 上失败（RQ3、RQ6）
- B2：`torch.zeros_like(tuple)` 行为未定义（RQ2a）
- B3：`get_mlp_submodules()` 无 MoE 分支（RQ3）
- B4：`get_mlp_down_proj()` 无 MoE 分支（RQ6）

详见 `EXPERIMENT_PLAN.md §MoE（多专家机制）模型——单独类别`。

**论文叙事定稿**：
> "Our main analyses cover 24 dense MLP models. Two Mixture-of-Experts models exhibit distinct MA generation mechanisms incompatible with single-V-direction theory; preliminary per-expert analysis shows a subset of experts may specialize in MA writing. Full MoE treatment is deferred to future work."

### 16.7 全局重跑计划（2026-04-21）

主结论定稿前需要的所有重跑，按依赖链排序。详见 `EXPERIMENT_PLAN.md §全局重跑计划汇总`。

| 阶段 | 内容 | 成本 |
|:-:|---|:-:|
| 0 | **修 7 个脚本 bug**（B1-B7，阻塞所有阶段）| ~2h |
| 1 | RQ1/RQ2 数据缺口（opt_6.7b, qwen2_7b, llama2_13b, llama2_7b_chat）| ~1h |
| 2 | RQ3/RQ4 结构 token 重做（全 26 模型）| ~3h |
| 3 | RQ6 exp6 全重跑（修 baseline 错层）| ~2.5h |
| 4 | RQ5 补数据（缺 single/macro 的 8 模型）| ~2h |
| 5 | **MoE 专项（Tier C，优先级低）** | ~3h |
| **合计** | | **~13.5h** |

**7 个脚本 bug 汇总**：
- B1：RQ3 `add_token` 只存功能词，丢内容词
- B2：MoE `SparseMoeBlock` 无 `.up_proj/.down_proj`（RQ3/6）
- B3：`get_mlp_submodules()` 缺 glm4/qwen1.5/qwen3.5/yi 白名单
- B4：`get_mlp_down_proj()` 缺 glm4/MoE 分支（RQ6）
- B5：`get_critical_layer()` 默认 L0 不读 L_origin（RQ6）
- B6：RQ6 baseline 只在 critical_layer 测（非真 MA）
- B7：RQ2a `MLPDisableHook` 未处理 tuple（MoE 静默失败）

**主结论定稿标准**（24 dense 模型）：
- ≥ 20 个支持 H₁-H₅（主论点："MA = MLP 在结构 token 位置的 v₁ mark"）
- qwen3.5 dense（Tier D）+ opt_6.7b（Tier E）单独讨论
- MoE（Tier C）附录讨论

### 16.8 剩余 RQ（RQ5/6）待分析

按顺序：RQ6（macro-SVD，RQ3/4 重做后合并分析）→ RQ5（V 消融）。当前数据均为 v2 错层版本，待起源层重跑（见 §12 批次）。

---

