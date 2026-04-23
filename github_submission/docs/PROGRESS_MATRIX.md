# 实验进度矩阵（function word + MA 框架）

> 数据来源：`ALL_EXPERIMENTS_SUMMARY.json` + 本轮 Stage 2 新数据
> 作用域：**排除 RQ7 训练动力学**，覆盖 RQ1-RQ6 + 新实验 H(C) + u₁ 稀疏度
> 初版：2026-04-17；**最近一次更新：2026-04-22 AM**（论点 E 定稿 + RQ3/RQ4 Stage 2 + H(C) 证伪 + 4 新 bug）

---

## 2026-04-22 更新概要

### 本轮主要产出
- **论点 E（双重稀疏假说）定稿** → `EXPERIMENT_PLAN.md` 对应节
- **RQ3 起源层 20/26** 完成（B1 + B13/B14/B15/B16 修复后）
- **RQ4 起源层 20/26** 完成（Agent D + Agent C）
- **H(C) entropy 实测 20/26**（证伪论点 C）
- **u₁ 稀疏度抽取 18/26**（支持论点 E hidden 侧）
- **挖矿木马 清理** on primary 117.50.223.194（/root/.sys-cache 已删，无复活）

### 4 个新 bug（今晚发现）
- **B13**：OPT 用 `self_attn_layer_norm`/`final_layer_norm`，llama 补丁找 `input_layernorm` 失败 → 路由到 forward-hook
- **B14**：gptj/bloom/falcon 的 `modify_gpt2.py` 前向改写撞 `ln_1` / `_get_embed_positions` → 同 B13 forward-hook
- **B15**：Falcon 本地权重自带 tx 4.x remote code（缺 `get_head_mask`）→ `load_model.py` 强制 `trust_remote_code=False`
- **B16**：`llama2_13b` monkey_patch `self.self_attn()` 解包 3-tuple（tx 5.x 返回 2-tuple）→ 路由 forward-hook

### 论点演化

| 版本 | 论点 | 结果 |
|:-:|---|:-:|
| A | 功能词 mark | ✗ |
| B | 结构 token mark | ✗ 只 glm4 |
| C | 低熵 token mark | ✗ 9/14 REFUTE |
| **E（定稿）** | **双重稀疏：「少数 token」×「少数 hidden 维度」** | **★ 20 模型支持** |

### 数据文件索引

```
paper_experiments/fixes/
├── results_stage2/
│   ├── RQ3_primary/          primary 14（B1-fix 数据）
│   ├── RQ4_stage2/           primary 14（Agent D）
│   └── HC_entropy/
│       ├── systemd_<20>/     14 primary + 6 secondary，NPZ+JSON
│       └── full_tokens.json  secondary 6 u₁
├── results_stage2_recovery/
│   ├── RQ3/                  secondary 6（Agent C）
│   └── RQ4/                  secondary 6（Agent C）
├── systemd_full_tokens.json  primary 14 u₁
├── systemd_topK_tokens.json  primary 14 Top-200 分类
├── analyze_HC.py             跨模型分析脚本
└── analyze_HC_results.md     H(C) 证伪报告
```

---

---

## 图例

| 符号 | 含义 |
|:-:|---|
| `✓` | 完成且层位正确、数据有效 |
| `◐` | 完成但**层位错误**（用了 peak_layer 而非 critical_layer） |
| `·` | 未跑 |
| `⚠` | 数据异常（NaN / Inf / 0 / 缺字段） |

---

## 一、26 模型 × 6 实验矩阵

| # | 模型 | RQ1<br>attn | RQ2<br>mlp | RQ3<br>func.words | RQ4<br>svd | RQ5<br>v-abl | RQ6<br>macro | 起源层 | 峰值层 | 生成模式 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | bloom_7b1 | ✓ | ✓ (-95%) | · | · | · | · | L3 | L12 | A 单层 |
| 2 | falcon_7b | ✓ | ✓ (-87%) | · | · | · | · | L3 | L23 | A 单层 |
| 3 | gpt2 | ✓ | ✓ (-6.7%) | · | · | · | · | L3 | L16 | B 多层 |
| 4 | gptj_6b | ✓ | ✓ (-90%) | · | · | · | · | L2 | L16 | A 单层 |
| 5 | llama2_13b | ✓ | ⚠ 缺 | · | · | · | · | 未知 | L22 | 未知 |
| 6 | llama3.1_8b | ✓ | ✓ (-87%) | ◐ | ◐ | ◐ L30 | ✓ | **L1** | L17 | A 单层 |
| 7 | mistral_7b_v03 | ✓ | ✓ (-41%) | · | · | · | · | L1 | L25 | 中间 |
| 8 | opt_6.7b | ✓ +250% 抑制 | ✓ (-12%) | · | · | · | · | L1 | L25 | B 多层 |
| 9 | qwen1.5_14b | ✓ | ✓ (-69%) | ◐ | ◐ | ◐ L38 | ✓ | **L2** | L37 | 中间 |
| 10 | qwen2.5_0.5b | ✓ | ✓ (-90%) | ◐ | ◐ | ◐ L22 | ✓ | **L0** | L15 | A 单层 |
| 11 | qwen2.5_7b | ✓ +266% 抑制 | ✓ (-85%) | · | ◐ | · | · | L3 | L16 | A 单层 |
| 12 | qwen2_7b | ✓ +∞ 抑制 | ✓ (-89%) | ◐ | ◐ | ◐ L26 | ✓ | **L3** | L16 | A 单层 |
| 13 | qwen3_0.6b | ✓ | ✓ (-94%) | ◐ | ◐ | ◐ L26 | ✓ | **L2** | L25 | A 单层 |
| 14 | qwen3_1.7b | ✓ | ✓ (-84%) | ◐ | ◐ | ◐ L26 | ✓ | **L2** | L25 | A 单层 |
| 15 | qwen3_4b | ✓ | ✓ (-34%) | ◐ | ◐ | ◐ L34 | ✓ | L5 | L17 | 中间 |
| 16 | qwen3_8b | ✓ | ✓ (-39%) | ◐ | ◐ | ◐ L34 | ✓ | L6 | L33 | 中间 |
| 17 | qwen3_14b | ✓ +85% 抑制 | ✓ (-67%) | ◐ | ◐ | ◐ L38 | ✓ | L6 | L33 | 中间 |
| 18 | qwen3_30b_a3b (MoE) | ✓ | ✓ (-58%) | ◐ | ◐ | · | ✓ | L2 | L36 | 中间 |
| 19 | qwen3_32b | ✓ +59% 抑制 | ✓ (-15%) | ◐ | ◐ | ◐ L62 | ✓ | L43 | L53 | B 多层 |
| 20 | qwen3.5_9b | ✓ | ✓ (-50%) | ◐ | ◐(n=1) | ◐ L30 | ✓ | L26 | L31 | 中间 |
| 21 | qwen3.5_27b | ✓ | ✓ (-30%) | ◐ | ◐ | ◐ L62 | ✓ | L54 | L58 | B 多层 |
| 22 | qwen3.5_35b_a3b (MoE) | ✓ | ✓ (-6.7%) | ◐ | ◐ | · | ✓ | L39 | L39 | B 多层 |
| 23 | yi_9b | ✓ +27% 抑制 | ✓ (-88%) | ◐ | ◐ | ◐ L46 | ✓ | **L1** | L47 | A 单层 |
| 24 | glm4_9b | ✓ | ✓ (-33%) | ◐ | ◐ | ◐ L38 | ✓ | L17 | L1 | 中间 |
| 25 | glm4_32b | ⚠ Inf | ⚠ 全 NaN | · | ⚠ NaN | ⚠ NaN | ⚠ NaN | 不可定 | L0 | 未知 |

**2026-04-17 初版汇总**：

| RQ | 完成 | 层位正确 | 需修复 |
|---|:-:|:-:|:-:|
| RQ1 (attn 消融) | 23 / 25 | 23 | 0 |
| RQ2 (MLP 消融) | 24 / 25 | 24 | 0 |
| RQ3 (function words) | 16 / 25 | 0 | 16（全错层） |
| RQ4 (SVD 对齐) | 18 / 25 | 0 | 18（全错层） |
| RQ5 (V 消融) | 14 / 25 | 0 | 14（全错层 / 模式 B 需换 RQ6） |
| RQ6 (macro-SVD) | 17 / 25 | 17 | 0 |

---

## 一-B、2026-04-22 更新后 26 模型 × 实验矩阵

本次 Stage 2 完成后的**最新状态**。图例：✓ 已跑 | · 未做 | 错 = 旧错层数据待 rerun | ⚠ u₁ 抽取失败。

```
                    RQ3  RQ4  H(C) u₁    | RQ1  RQ2a  RQ5   RQ6
                    ---  ---  ---  ---   | ---  ----  ---   ---
1  gpt2              ·    ·    ·    ·    |  ✓    ✓    错    错
2  gptj_6b           ✓    ✓    ✓    ⚠    |  ✓    ✓    错    错
3  opt_6.7b          ✓    ✓    ✓    ⚠    |  ✓    ✓    错    错
4  bloom_7b1         ✓    ✓    ✓    ✓    |  ✓    ✓    错    错
5  falcon_7b         ✓    ✓    ✓    ⚠    |  ✓    ✓    错    错
6  mistral_7b_v03    ✓    ✓    ✓    ✓    |  ✓    ✓    错    错
7  llama2_13b        ✓    ✓    ✓    ✓    |  ✓    ·    错    错
8  llama3.1_8b       ✓    ✓    ✓    ✓    |  ✓    ✓    错    错
9  qwen2.5_0.5b      ·    ·    ·    ·    |  ✓    ✓    错    错
10 qwen2.5_7b        ✓    ✓    ✓    ✓    |  ✓    ✓    错    错
11 qwen2_7b          ✓    ✓    ✓    ✓    |  ⚠Inf ✓    错    错
12 qwen3_0.6b        ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
13 qwen3_1.7b        ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
14 qwen3_4b          ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
15 qwen3_8b          ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
16 qwen3_14b         ✓    ✓    ✓    ⚠    |  ✓    ✓    错    ✓
17 qwen3_30b_a3b MoE ·    ·    ·    ·    |  ✓    ✓    ·     ✓
18 qwen3_32b         ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
19 qwen3.5_9b        ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
20 qwen3.5_27b       ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
21 qwen3.5_35b MoE   ·    ·    ·    ·    |  ✓    ✓    ·     ✓
22 yi_9b             ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
23 qwen1.5_14b       ✓    ✓    ✓    ✓    |  ✓    ✓    错    ✓
24 glm4_9b           ✓    ✓    ✓    ⚠    |  ✓    ✓    错    ✓
25 glm4_32b          ·    ·    ·    ·    |  ⚠Inf ·    ·     ·
26 llama2_7b_chat    ·    ·    ·    ·    |  ·    ·    ·     ·
─────────────────────────────────────────────────
覆盖              20/26 20/26 20/26 15/26 | 23/26 24/26 14-错 17-错
```

**2026-04-22 汇总**：

| 实验 | 覆盖 | 状态 |
|---|:-:|:-:|
| **RQ1** (attn 消融) | 23/26 | ✅ 定稿，缺 2 Inf + 1 aux |
| **RQ2a** (全禁 MLP) | 24/26 | ✅ 定稿 |
| **RQ3** Stage 2 (B1+B13-16 fix 起源层) | **20/26** | ✅ **论点 E token 侧** |
| **RQ4** Stage 2 (起源层) | **20/26** | ✅ **论点 E hidden 侧** |
| **RQ5** (V 消融起源层) | 0/26 新数据 | 🟡 14/26 旧错层数据；代码修完待 rerun |
| **RQ6** (macro-SVD) | 0/26 新数据 | 🟡 17/26 baseline 错层；B5/B6 修完待 rerun |
| **H(C) entropy** 🆕 | 20/26 | ✅ **证伪论点 C** |
| **u₁ 稀疏度** 🆕 | 18/26（3 抽取失败 qwen3_14b/falcon/gptj/opt/glm4）| ✅ **论点 E 定量** |

**共同缺口（6 模型）**：gpt2 / qwen2.5_0.5b / glm4_32b / qwen3_30b_a3b / qwen3.5_35b_a3b / llama2_7b_chat

---

## 二、核心 Bug：RQ3/4/5 层位错误

**根因**：`paper_experiments/run_all_rq.sh:21-25` 和 `run_rq345_peak_layer.sh:13-20` 把 `KEY_LAYER` 读成 `table1_rq1.json["key_layer"]`（peak_layer），传给了 RQ3/4/5 的 `--layer_id`。正确做法是读 RQ2 结果 `exp2.critical_layer`（起源层）。

**两种错位类型**：

| 类型 | 触发条件 | 错在哪 | 修复 |
|---|---|---|---|
| **A. 层号错位** | 单层 ΔMA ≥ 60%（模式 A / 部分中间） | 起源层 ≠ 峰值层，脚本用了峰值层 | 改 `--layer_id = exp2.critical_layer` 重跑 RQ3/4/5 |
| **B. 模式错位** | 单层 ΔMA < 60%（模式 B / MoE） | 不该用单层 RQ5，应用 macro-SVD | 用 `RQ6_single_layer_activation/exp6_macro_svd_*` |

**直接证据**（`MA_CONCLUSIONS_AND_ARGUMENTS.md:183`）：
> GPT-J L2 消融 MA → **-99.1%**，L16 消融 → **-0.01%**。

---

## 三、数据异常清单

| 模型 | 问题 | 处理 |
|---|---|---|
| glm4_32b | exp1 Infinity、exp2 L10+ 全 NaN、exp4/5/6 全 NaN | 疑似 dtype / overflow；需查 loader + fp16→fp32 cast |
| llama2_13b | 缺 exp2 | 必须补 RQ2 才能确定起源层 |
| qwen2_7b exp1 | ΔTop1 = +∞（baseline 接近 0） | RQ1 结果不可信，重测 baseline |
| qwen3.5_9b exp4 | num_layers = 1（其他都 5） | 数据截断，重跑 |
| qwen3_30b_a3b / qwen3.5_35b_a3b (MoE) | 缺 exp5 | 需 expert-level v-ablation 框架（单独设计） |

---

## 四、重跑方案（按优先级）

### Tier A — 起源层重跑 RQ3 / RQ4 / RQ5（核心 bug 修复）

修改脚本：把 `KEY_LAYER` 改成读取 `exp2.critical_layer`（新建 `run_rq345_origin_layer.sh`）。结果存 `results/wikitext_run/RQ3-5_origin/` 避免覆盖原对照数据。

| 子批 | 对象 | 起源层范围 | 预期 RQ5 ΔMA | 成本 |
|---|---|---|---|---|
| **A1** 模式 A（10 个）| bloom_7b1, falcon_7b, gptj_6b, llama3.1_8b, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, qwen3_1.7b, yi_9b | L0-L3 | **-85% ~ -99%** | ≈ 3h |
| **A2** 中间形态（7 个） | mistral_7b_v03, qwen1.5_14b, glm4_9b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3.5_9b | L1-L26 | **-30% ~ -70%** | ≈ 3h |
| **A3** 模式 B（5 个）| gpt2, opt_6.7b, qwen3_32b, qwen3.5_27b, qwen3.5_35b_a3b | 单层做意义弱 | < -30%，佐证"非单层主导" | ≈ 2h |

### Tier B — 数据补洞

| 任务 | 成本 |
|---|---|
| llama2_13b 补 RQ2 | ~15 min |
| qwen2_7b exp1 baseline 重测 | ~10 min |
| qwen3.5_9b RQ4 重跑（完整 5 层） | ~20 min |
| glm4_32b 全流程 fp32 重测 | ~1h |

### Tier C — 暂缓（需新框架）

- MoE 模型 qwen3_30b_a3b, qwen3.5_35b_a3b 的 expert-level v-ablation
- RQ7 训练动力学（本轮排除）

---

## 五、建议执行顺序（2026-04-17 原版）

1. 先修脚本（`run_rq345_origin_layer.sh`），对 **GPT-J** 单模型做 pilot 验证：起源层 RQ5 应出 -90%+
2. pilot 通过后批量跑 Tier A1（10 个模式 A 模型）
3. Tier B 数据补洞（可与 A 并行）
4. Tier A2 → Tier A3
5. 结果合并到新 ALL_EXPERIMENTS_SUMMARY_v2.json，更新本表格

---

## 六、2026-04-22 **当前剩余**执行清单

今晚 Stage 2 + H(C) + u₁ + Agent C/D/E 完成后，剩余工作：

### 🔴 Tier A（已完成）
- ~~RQ3 起源层重跑~~ ✅ 20/26（缺 6 模型列在下面）
- ~~RQ4 起源层重跑~~ ✅ 20/26
- **RQ5 起源层重跑** 🟡 待做 — 代码修完，14-20 runs 约 ~1.5h

### 🟡 Tier A' — 新增待办
| 任务 | 成本 |
|---|:-:|
| **RQ6 exp6 全 26 rerun**（B4/B5/B6 修完后）| ~2.5h |
| **RQ5 起源层全 rerun**（20-26 dense） | ~1.5h |
| **u₁ 抽取脚本补 gptj_6b / opt_6.7b**（decoder layer detection 修） | ~30 min |

### 🟢 Tier B — 26 模型缺口补跑（4 个非 MoE）
| 模型 | 实验 | 在哪跑 | 成本 |
|---|---|---|:-:|
| **gpt2** | RQ3 + RQ4 + H(C) + u₁ | 主/副 | ~10 min |
| **qwen2.5_0.5b** | RQ3 + RQ4 + H(C) + u₁ | 主/副 | ~10 min |
| **qwen2_7b RQ1** 补 nsamples=60 | RQ1 | 主/副 | ~10 min |
| **glm4_32b** fp32 | 全部 | 需先解决权重 (65 GB) | 延办 |

### ⏸ Tier C — 暂缓
- **qwen3_30b_a3b / qwen3.5_35b_a3b (MoE)** — B2 未修，需 per-expert 分析框架
- **llama2_7b_chat** — 辅助模型，非核心

---

## 七、论点 E 定量定稿（2026-04-22）

详细表格与判据见 `EXPERIMENT_PLAN.md` 的 "论点 E 定稿" 节。

**核心数字**：
- u₁ hidden eff_dim/hidden ≤ 1%：**12/18 模型（67%）**
- u₁ hidden eff_dim/hidden ≤ 0.2%（极端稀疏）：**5/18**
- 最极端：**bloom_7b1 u₁ eff_dim = 1.8/4096 = 0.04%**
- Top-1 token 占 Top-K ≥ 50%：**5/20**（qwen3_8b ` the`×95%、qwen3_4b、glm4_9b、qwen1.5_14b、qwen3_32b）
- σ₁/σ₂ ≥ 2（强谱）：2/20（qwen2_7b 2.84、qwen2.5_7b 2.64）
