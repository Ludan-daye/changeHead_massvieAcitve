# 25 模型待补充实验清单

> 目标：建立完整 function word + MA 框架体系（**排除 RQ7 训练动力学**）
> 基于 `ALL_EXPERIMENTS_SUMMARY.json` 做差异分析
> 生成日期：2026-04-17

---

## 图例（TODO 动作）

| 符号 | 含义 |
|:-:|---|
| `✓` | 已完成（数据可用、层位正确），**不做** |
| `🔧 起` | 层位错（做在了 peak），**在起源层重跑** |
| `➕ 新` | 未做，**新跑** |
| `⚠ 修` | 数据异常（NaN/Inf/截断），**修复 loader 或重测** |
| `🔄 B` | 属模式 B/MoE，单层做无意义，**走 macro-SVD 代替（RQ6）** |
| `⏸ 暂` | MoE 需专项框架，**本轮暂缓** |

**层位规则**：
- RQ1 / RQ2：不挑层（都做全部层扫描）
- RQ3 / RQ4 / RQ5：**必须在 `exp2.critical_layer`（起源层）**，NOT `exp1.peak_layer`（观测层）
- RQ6：不挑层（macro-SVD 跨多层）

---

## 一、25 模型待办矩阵

> 列 = 各 RQ 的 TODO；表最后一列是优先级（基于模式 A 优先、完整度优先）

| # | 模型 | 总层 | **L_origin** | L_peak | 模式 | RQ1 | RQ2 | RQ3 | RQ4 | RQ5 | RQ6 | 优先级 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **gptj_6b** | 28 | **L2** | L16 | A 单层 | ✓ | ✓ | ➕ 新 L2 | ➕ 新 L2 | ➕ 新 L2 | ➕ 新 | ★ Pilot |
| 2 | bloom_7b1 | 30 | **L3** | L12 | A 单层 | ✓ | ✓ | ➕ 新 L3 | ➕ 新 L3 | ➕ 新 L3 | ➕ 新 | ★★★ |
| 3 | falcon_7b | 32 | **L3** | L23 | A 单层 | ✓ | ✓ | ➕ 新 L3 | ➕ 新 L3 | ➕ 新 L3 | ➕ 新 | ★★★ |
| 4 | llama3.1_8b | 32 | **L1** | L17 | A 单层 | ✓ | ✓ | 🔧 起 L1 | 🔧 起 L1 | 🔧 起 L1 | ✓ | ★★★ |
| 5 | qwen2.5_0.5b | 24 | **L0** | L15 | A 单层 | ✓ | ✓ | 🔧 起 L0 | 🔧 起 L0 | 🔧 起 L0 | ✓ | ★★★ |
| 6 | qwen2.5_7b | 28 | **L3** | L16 | 中间(85%) | ✓ +266% | ✓ | ➕ 新 L3 | 🔧 起 L3 | ➕ 新 L3 | ➕ 新 | ★★★ |
| 7 | qwen2_7b | 28 | **L3** | L16 | A 单层 | ⚠ 修 Inf | ✓ | 🔧 起 L3 | 🔧 起 L3 | 🔧 起 L3 | ✓ | ★★★ |
| 8 | qwen3_0.6b | 28 | **L2** | L25 | A 单层 | ✓ | ✓ | 🔧 起 L2 | 🔧 起 L2 | 🔧 起 L2 | ✓ | ★★★ |
| 9 | qwen3_1.7b | 28 | **L2** | L25 | 中间(84%) | ✓ | ✓ | 🔧 起 L2 | 🔧 起 L2 | 🔧 起 L2 | ✓ | ★★★ |
| 10 | yi_9b | 48 | **L1** | L47 | A 单层 | ✓ +27% | ✓ | 🔧 起 L1 | 🔧 起 L1 | 🔧 起 L1 | ✓ | ★★★ |
| 11 | mistral_7b_v03 | 32 | **L1** | L25 | 中间(41%) | ✓ | ✓ | ➕ 新 L1 | ➕ 新 L1 | ➕ 新 L1 | ➕ 新 | ★★ |
| 12 | qwen1.5_14b | 40 | **L2** | L37 | 中间(69%) | ✓ | ✓ | 🔧 起 L2 | 🔧 起 L2 | 🔧 起 L2 | ✓ | ★★ |
| 13 | qwen3_4b | 36 | **L5** | L17 | 中间(34%) | ✓ | ✓ | 🔧 起 L5 | 🔧 起 L5 | 🔧 起 L5 | ✓ | ★★ |
| 14 | qwen3_8b | 36 | **L6** | L33 | 中间(39%) | ✓ | ✓ | 🔧 起 L6 | 🔧 起 L6 | 🔧 起 L6 | ✓ | ★★ |
| 15 | qwen3_14b | 40 | **L6** | L33 | 中间(67%) | ✓ +85% | ✓ | 🔧 起 L6 | 🔧 起 L6 | 🔧 起 L6 | ✓ | ★★ |
| 16 | qwen3.5_9b | 32 | **L26** | L31 | 中间(50%) | ✓ | ✓ | 🔧 起 L26 | ⚠ 修(n=1→5) L26 | 🔧 起 L26 | ✓ | ★★ |
| 17 | glm4_9b | 40 | **L17** | L1 | 中间(33%) | ✓ | ✓ | 🔧 起 L17 | 🔧 起 L17 | 🔧 起 L17 | ✓ | ★★ |
| 18 | llama2_13b | ? | **未知** | L22 | ? | ✓ | ⚠ 修 缺 | ➕ 新 (依赖 RQ2) | ➕ 新 (依赖 RQ2) | ➕ 新 (依赖 RQ2) | ➕ 新 | ★★ 先补 RQ2 |
| 19 | gpt2 | 12 | L3 | L16 | **B 多层** | ✓ | ✓ | ➕ 新 L3 | ➕ 新 L3 | 🔄 B 跑作对照 | ➕ 新 | ★ |
| 20 | opt_6.7b | 32 | L1 | L25 | **B 多层** | ✓ +250% | ✓ | ➕ 新 L1 | ➕ 新 L1 | 🔄 B 跑作对照 | ➕ 新 | ★ |
| 21 | qwen3_32b | 64 | L43 | L53 | **B 多层** | ✓ +59% | ✓ | 🔧 起 L43 | 🔧 起 L43 | 🔄 B 跑作对照 | ✓ | ★ |
| 22 | qwen3.5_27b | 64 | L54 | L58 | **B 多层** | ✓ | ✓ | 🔧 起 L54 | 🔧 起 L54 | 🔄 B 跑作对照 | ✓ | ★ |
| 23 | qwen3_30b_a3b (MoE) | 48 | L2 | L36 | 中间(58%) | ✓ | ✓ | 🔧 起 L2 | 🔧 起 L2 | ⏸ 暂 MoE | ✓ | ★ |
| 24 | qwen3.5_35b_a3b (MoE) | 40 | L39 | L39 | **B 多层** | ✓ | ✓ | 🔧 起 L39 | 🔧 起 L39 | ⏸ 暂 MoE | ✓ | ★ |
| 25 | glm4_32b | 61 | **不可定** | L0 | ? | ⚠ 修 Inf | ⚠ 修 全 NaN | ⚠ 修 依赖 | ⚠ 修 | ⚠ 修 | ⚠ 修 | 🔧 先 fp32 修复 |

---

## 二、任务总量汇总

### 按 RQ 统计（实际要跑的实验次数）

| RQ | 说明 | 已 OK | 重跑 (🔧) | 新跑 (➕) | 修复 (⚠) | MoE 暂缓 (⏸) | **本轮要跑** |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| RQ1 | attention 消融 | 23 | 0 | 0 | 2 | 0 | **2** |
| RQ2 | MLP 全层扫描 | 23 | 0 | 1 | 1 | 0 | **2** |
| RQ3 | function words / SVD | 0 | 14 | 10 | 1 | 0 | **25** |
| RQ4 | SVD 对齐 | 0 | 15 | 9 | 1 | 0 | **25** |
| RQ5 | V 矩阵消融 | 0 | 14 | 6 | 1 | 2 | **21 + 4** (4 是 B 模式对照) |
| RQ6 | macro-SVD | 16 | 0 | 8 | 1 | 0 | **9** |
| **合计** | | **62** | **43** | **34** | **7** | **2** | **≈ 84 次实验** |

### 按模型的"待办量"排序（从少到多，方便安排）

| 等级 | 待办数 | 模型 |
|:-:|:-:|---|
| A | 4 个（全新跑） | bloom_7b1, falcon_7b, gptj_6b |
| B | 3 个（全重跑起源） | yi_9b, qwen3_0.6b, qwen3_1.7b, qwen2_7b(+RQ1修)、llama3.1_8b, qwen2.5_0.5b, qwen1.5_14b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3.5_9b(+RQ4修), glm4_9b, qwen3_32b, qwen3.5_27b, qwen3_30b_a3b, qwen3.5_35b_a3b |
| C | 4-5 个（混合） | mistral_7b_v03, qwen2.5_7b, gpt2, opt_6.7b, llama2_13b |
| D | 全部 | glm4_32b（先修 loader） |

---

## 三、执行批次（建议顺序）

### 批次 0 — 前置修复（30 min）

| 任务 | 操作 | 预期 |
|---|---|---|
| 改脚本：`run_rq345_origin_layer.sh` | 从 RQ2 读 `exp2.critical_layer` 作为 `--layer_id` | 替换 peak 读取逻辑 |
| 修 qwen2_7b RQ1 数值 | baseline_top1 重测（改 `--nsamples 50`） | 避免除零 |
| 修 glm4_32b dtype | 强制 fp32 推理 | 得到有效 baseline |

### 批次 1 — Pilot：gptj_6b 全套（~30 min）

跑 gptj_6b 的 RQ3/4/5/6（L2 起源层）
**验收标准**：RQ5 的 ΔMA 应在 **-85% ~ -99%**（对比之前 L16 峰值层的 -0.01%）
通过后再放开批量。

### 批次 2 — Tier A：10 个模式 A 模型（~3h）

bloom_7b1, falcon_7b, llama3.1_8b, qwen2.5_0.5b, qwen2.5_7b, qwen2_7b, qwen3_0.6b, qwen3_1.7b, yi_9b
（含 RQ3/4/5 起源层 + 缺的 RQ6）

### 批次 3 — Tier 中间形态（7 个，~3h）

mistral_7b_v03, qwen1.5_14b, glm4_9b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3.5_9b

### 批次 4 — Tier B 模式 B 对照（4 个，~2h）

gpt2, opt_6.7b, qwen3_32b, qwen3.5_27b（RQ5 预期 <30% 佐证非单层，RQ6 macro-SVD 才是重点）

### 批次 5 — 数据补洞

llama2_13b（先补 RQ2，再跑 RQ3/4/5/6），glm4_32b（fp32 全流程）

### 批次 6 — 暂缓（不做）

MoE 的 qwen3_30b_a3b / qwen3.5_35b_a3b 的 RQ5（待设计 expert-level 框架）

---

## 四、脚本修改点

**`paper_experiments/run_all_rq.sh` 第 21-25 行**：
```bash
# 原（错）:
KEY_LAYER=$(python3 -c "
import json
d = json.load(open('results/wikitext_run/RQ1/${model}/table1_rq1.json'))
print(d['key_layer'])
")

# 改（正确，起源层）:
KEY_LAYER=$(python3 -c "
import json
d = json.load(open('results/wikitext_run/RQ2a/${model}/table1_rq2.json'))
print(d['critical_layer'])
")
```

（RQ2 脚本需要输出 `critical_layer` 到 `table1_rq2.json` 如果还没有）

---

## 五、交付物

全部跑完后输出：
- `results/wikitext_run/RQ3-5_origin/*/` — 起源层版结果
- 更新后的 `ALL_EXPERIMENTS_SUMMARY_v2.json`
- 更新本 `TODO_EXPERIMENTS.md`：所有 🔧/➕ 变 ✓
- `PROGRESS_MATRIX.md` 的全 ✓ 版本

---

## 六、待你确认

1. 优先级★/★★/★★★划分是否合理？
2. MoE 两个 ⏸ 暂缓是否 OK？如果本轮要做，需要额外设计框架。
3. glm4_32b 是否本轮修？还是先放弃？
4. 批次 1 (gptj_6b pilot) 通过的验收数字 **-85% ~ -99%** 是否接受？

---

## 七、2026-04-22 Stage 2 RQ3+RQ4 执行后剩余项

> 说明：今晚在 secondary（8.138.30.52:6007）修了 B13/B14/B15 三个 bug 并跑完 6 个模型，primary 之前跑完 15 dense。以下是 stage 2 尚未完成项。

### 7.1 ⚠ 延办：1 runs（GPU 竞争导致 OOM，非代码问题）

| 模型 | 运行 | 状态 | 原因 | 建议 |
|---|---|---|---|---|
| `llama2_13b` | RQ3 | ❌ OOM × 2 | secondary 其他用户持续占 51GB/80GB；试过 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `seqlen=512` 都失败 | **primary 空闲时补跑 ~10 min**；数据文件：`fixes/logs/pipeline_20260421_231037/llama2_13b__RQ3*.log` |

### 7.2 ➕ 结果搬运：secondary → primary

| 源 | 目的 | 内容 | 命令 |
|---|---|---|---|
| `vicuna@8.138.30.52:/home/vicuna/ma/paper_experiments/fixes/results/` | `/Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_secondary/` | 11 runs（opt/gptj/bloom/falcon/mistral × RQ3+RQ4，llama2_13b × RQ4）| `rsync -av -e "ssh -p 6007" vicuna@8.138.30.52:~/ma/paper_experiments/fixes/results/ ./fixes/results_secondary/` |

### 7.3 ⚠ Tier D：3 个架构不兼容模型（stage 2 primary 跳过）

| 模型 | 失败原因 | 处理方案 |
|---|---|---|
| `qwen3.5_9b` | B10：Qwen3_5DecoderLayer 全用 `linear_attn`（GatedDeltaNet），不是标准 self_attn | 架构根本不同 → **Tier D 附录讨论，不纳入主结论** |
| `qwen3.5_27b` | B10 同上 | 同上 |
| `glm4_9b` | B11：ChatGLM shim `GlmMLP.gate_up_proj` 缺失 — 实际问题：GLMBlock 用 `self_attention`（下划线）+ fused gate_up | secondary **已用 forward-hook 绕过**（hook 不重写 forward）。primary 同款修复已 push — 需要 primary 重跑。**可补 ~12 min** |

### 7.4 ✅ 已完成并验证的 3 个 bug fix（本轮成果）

| ID | 现象 | 修复 | 验证 |
|---|---|---|---|
| **B13** | OPT 用 `self_attn_layer_norm`/`final_layer_norm`，llama 补丁找 `input_layernorm` 失败 | `enable_custom_block` 把 `opt_*` 路由到 forward-hook（不重写 forward）| ✓ opt_6.7b RQ3+RQ4 |
| **B14** | gptj/bloom/falcon 的 `modify_gpt2.py` 前向改写撞 `ln_1` / `_get_embed_positions` / `get_head_mask` | 同 B13 思路，`enable_custom_block` 把 gptj/bloom/falcon 也路由到 forward-hook | ✓ 三个模型 RQ3+RQ4 |
| **B15** | Falcon 本地权重自带 `modeling_falcon.py`（transformers 4.x，缺 `get_head_mask`）| `lib/load_model.py` falcon 分支强制 `trust_remote_code=False`（用 tx 5.x 原生 `FalconForCausalLM`）| ✓ falcon_7b RQ3+RQ4 |

### 7.5 Stage 2 RQ3+RQ4 整体进度快照

| 服务器 | 模型数 | RQ3 ok | RQ4 ok | 备注 |
|---|:-:|:-:|:-:|---|
| Primary | 15 | 11 | 11 | + 1 skip + 3 Tier D（qwen3.5×2、glm4_9b）|
| Secondary | 6 | 5 | 6 | 新修 bug 打通 opt/gptj/bloom/falcon/mistral，llama2_13b RQ3 延办 |
| **合计** | **21/26** | **16** | **17** | **剩 llama2_13b RQ3 + 3 Tier D = 4 runs**；MoE（qwen3_30b_a3b / qwen3.5_35b_a3b）不纳入 |
