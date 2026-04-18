# 实验执行手册（v2 — 已对齐仓库真实脚本接口）

> 本手册和 `EXPERIMENT_PLAN.md` 中的讨论结论**逐条对齐**，所有命令**经过代码核实**。
> 2026-04-17 更新：修复 v1 里脚本参数错（aggregate_layers / topk_list / macro_mode 等）、补齐 glm4_32b / llama2_13b 依赖路径、新增 `exp5_macro_v_ablation.py`。

---

## 0. 前置清单

### 0.1 环境

```bash
conda activate torch               # 或你的 PyTorch 环境
export HF_ENDPOINT=https://hf-mirror.com
cd paper_experiments
```

### 0.2 必需的已完成修改（本轮已做）

| 改动 | 文件 | 作用 |
|---|---|---|
| ✓ 补齐 8 个模型条目 | `lib/model_dict.py` | 让 bloom_7b1, opt_6.7b, mistral_7b_v03, qwen2.5_0.5b, qwen3.5_{9b,27b,35b_a3b}, glm4_32b 都能被加载 |
| ✓ glm4 走 fp32 分支 | `lib/load_model.py` | 解决 glm4_32b fp16 溢出 Infinity；同时支持 `--force_fp32` 全局开关 |
| ✓ eval_ppl 参数化 | `lib/eval_utils.py` | 去掉 seqlen=4096/device=cuda:0 硬编码 |
| ✓ 新脚本 `exp5_macro_v_ablation.py` | `RQ5_v_matrix_ablation/` | 模式 B 的多层 macro v₁ 投影消除 |
| ✓ 新脚本 `run_rq345_origin_layer.sh` | `paper_experiments/` | 从起源层读 layer_id（替代老 peak layer 脚本）|

### 0.3 输入依据

- `ALL_EXPERIMENTS_SUMMARY.json` 的 `exp2.critical_layer` → 起源层真值来源
- 旧脚本 `run_rq345_peak_layer.sh` 读的是 `key_layer = peak_layer` ← **不再用**
- RQ2b（per-layer MLP 消融）脚本位置：**老仓库** `changeHead_massvieAcitve/experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py`（`paper_experiments/RQ2_mlp_source/` 里只有 `exp2a` 禁全部和 `exp2c` 内部分析）

### 0.4 输出目录约定

```
results/wikitext_run/
├── RQ1/{model}/              # 既有保留
├── RQ2a/{model}/             # 既有保留（禁全部 MLP）
├── RQ2b/{model}/             # 逐层消融（新跑或补跑）
├── RQ2c/{model}/             # 贪心累积消融（≡ RQ6.4 progressive，合并输出）
├── RQ3_origin/{model}/       # 起源层新跑
├── RQ4_origin/{model}/       # 起源层新跑
├── RQ5_origin/{model}/       # 起源层单层 V 消融
├── RQ5_macro/{model}/        # 模式 B 多层 macro v₁ 消融
├── RQ6/{model}/              # macro-SVD + top-K
```

**旧目录** `results/wikitext_run/RQ3/, RQ4/, RQ5/`（错层版）**保留**作对照。

---

## 1. 25 模型起源层查找表

| # | 模型 (--model) | **L_origin** | L_peak | RQ2b max ΔMA | 初步模式 | 优先级 | 备注 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|---|
| 1 | `bloom_7b1` | **3** | 12 | -95% | A | ★★★ | |
| 2 | `falcon_7b` | **3** | 23 | -87% | A | ★★★ | |
| 3 | `gptj_6b` | **2** | 16 | -90% | A | ★ Pilot | **首批验收** |
| 4 | `llama3.1_8b` | **1** | 17 | -87% | A | ★★★ | |
| 5 | `qwen2.5_0.5b` | **0** | 15 | -90% | A | ★★★ | |
| 6 | `qwen2.5_7b` | **3** | 16 | -85% | A(近中间) | ★★★ | +266% suppressive |
| 7 | `qwen2_7b` | **3** | 16 | -89% | A | ★★★ | **RQ1 需先修 (nsamples→60)** |
| 8 | `qwen3_0.6b` | **2** | 25 | -94% | A | ★★★ | |
| 9 | `qwen3_1.7b` | **2** | 25 | -84% | A(近中间) | ★★★ | |
| 10 | `yi_9b` | **1** | 47 | -88% | A | ★★★ | +27% suppressive |
| 11 | `mistral_7b_v03` | **1** | 25 | -41% | 中间 | ★★ | 注意是 v0.3 不是 v0.1 |
| 12 | `qwen1.5_14b` | **2** | 37 | -69% | 中间 | ★★ | |
| 13 | `qwen3_4b` | **5** | 17 | -34% | 中间 | ★★ | |
| 14 | `qwen3_8b` | **6** | 33 | -39% | 中间 | ★★ | |
| 15 | `qwen3_14b` | **6** | 33 | -67% | 中间 | ★★ | +85% suppressive |
| 16 | `qwen3.5_9b` | **26** | 31 | -50% | 中间 | ★★ | |
| 17 | `glm4_9b` | **17** | 1 | -33% | 中间→**真 B 候选** | ★★ | macro η=4.33 |
| 18 | `qwen3_30b_a3b` (MoE) | **2** | 36 | -58% | 中间→MoE 异常 | ★ | macro η=1.06 |
| 19 | `gpt2` | **3** | 16 | -6.7% | B | ★ | macro 历史 3.48 |
| 20 | `opt_6.7b` | **1** | 25 | -12% | B | ★ | +250% suppressive |
| 21 | `qwen3_32b` | **43** | 53 | -15% | B | ★ | +59% suppressive |
| 22 | `qwen3.5_27b` | **54** | 58 | -30% | B→强聚合 B | ★ | macro η=4.59 |
| 23 | `qwen3.5_35b_a3b` (MoE) | **39** | 39 | -6.7% | B | ★ | L_origin = L_peak |
| 24 | `llama2_13b` | **待定** | 22 | — | — | ⏸ | **先跑 RQ2b** |
| 25 | `glm4_32b` | **待定** | 0 | — | — | ⏸ | **fp32 重跑 RQ1/2a/2b** |

Bash 关联数组已经写进 `run_rq345_origin_layer.sh`，不需手动维护。

---

## 2. 真实脚本接口（核实过）

### 2.1 RQ1：整层禁用 attention

`paper_experiments/RQ1_attention_contribution/exp1_feasibility_test.py`

```bash
python RQ1_attention_contribution/exp1_feasibility_test.py \
    --model {MODEL} \
    --nsamples 30 \
    --dataset wikitext \
    --savedir results/wikitext_run/RQ1/{MODEL} \
    --access_token "your_token_or_placeholder"
```

输出：`baseline/results.json`, `all_heads_disabled/results.json`, `table1_rq1.json`（含 `key_layer = peak_layer` 字段——**这就是之前的 bug 根源**）。

### 2.2 RQ2a：全部禁用 MLP

`paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py`

```bash
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
    --model {MODEL} \
    --nsamples 30 \
    --dataset wikitext \
    --savedir results/wikitext_run/RQ2a/{MODEL} \
    --access_token "..."
```

注：**该脚本不支持 `--mode per_layer`**（我 v1 手册错了）。只做"全禁"。

### 2.3 RQ2b：逐层 MLP 消融（**走老仓库**）

```bash
cd ../changeHead_massvieAcitve   # 切到老仓库
python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py \
    --model {MODEL} \
    --nsamples 20 \
    --seqlen 4096 \
    --savedir {ABS_PATH}/paper_experiments/results/wikitext_run/RQ2b/{MODEL} \
    --access_token "..." --revision main
cd -
```

输出：`baseline.json` + `layer_0_disabled.json` ... `layer_N_disabled.json`。

### 2.4 RQ2c ≡ RQ6.4 progressive：贪心累积消融

`paper_experiments/RQ6_single_layer_activation/exp6_progressive_ablation.py`

```bash
python RQ6_single_layer_activation/exp6_progressive_ablation.py \
    --model {MODEL} \
    --nsamples 30 \
    --seqlen 1024 \
    --threshold_pct 10.0 \
    --max_steps 20 \
    --savedir results/wikitext_run/RQ2c/{MODEL}     # 或 RQ6_progressive/{MODEL}
```

输出：`{MODEL}_rq6_greedy.json`——同时填 RQ2c 和 RQ6.4。

注意：**没有** `--top_k` 参数（我 v1 错了）。是 `--threshold_pct`（默认 10%，超过该值即停）+ `--max_steps`（最多消融多少层）。`--nsamples` 默认 10，**必须传 30 以对齐其他实验**。

### 2.5 RQ3：功能词 SVD 映射

`paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py`

```bash
python RQ3_function_words/exp5_function_words_svd_mapping.py \
    --model {MODEL} \
    --layer_id {L_ORIGIN} \
    --nsamples 30 \
    --savedir results/wikitext_run/RQ3_origin/{MODEL}
```

脚本默认 `nsamples=50`，我们手册传 30 以对齐。

### 2.6 RQ4：SVD 几何对齐

`paper_experiments/RQ4_svd_alignment/exp3_svd_alignment_analysis.py`

```bash
python RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
    --model {MODEL} \
    --layer_id {L_ORIGIN} \
    --nsamples 30 \
    --savedir results/wikitext_run/RQ4_origin/{MODEL}
```

单层分析，**按 `EXPERIMENT_PLAN.md` 附录 C 的定案**：只跑起源层 1 层。

### 2.7 RQ5（单层）：V 矩阵替换为随机正交

`paper_experiments/RQ5_v_matrix_ablation/exp5_v_ablation.py`

```bash
python RQ5_v_matrix_ablation/exp5_v_ablation.py \
    --model {MODEL} \
    --layer_id {L_ORIGIN} \
    --nsamples 30 \
    --savedir results/wikitext_run/RQ5_origin/{MODEL}
```

输出：`{MODEL}_v_ablation_results.json`。
注意 v1 手册里写的 `--macro_mode` / `--macro_v_path` **不存在**，见 §2.9 用新脚本。

### 2.8 RQ6 macro-SVD（完整）

`paper_experiments/RQ6_single_layer_activation/exp6_macro_svd_full.py`

```bash
python RQ6_single_layer_activation/exp6_macro_svd_full.py \
    --model {MODEL} \
    --origin_layers "0,1,2,3,4" \
    --nsamples 20 \
    --seqlen 512 \
    --savedir results/wikitext_run/RQ6/{MODEL}
```

**关键**：参数是 `--origin_layers`（逗号分隔），不是 `--aggregate_layers`。输出：`{MODEL}_macro_svd_full.json`。

对**深层起源**模型（qwen3_32b L43、qwen3.5_27b L54），`--origin_layers` 应覆盖起源层附近：例如 `qwen3_32b` 用 `"40,41,42,43,44,45"`；`qwen3.5_27b` 用 `"50,51,52,53,54,55"`。

### 2.9 RQ6 single-layer activation（仅保留单层的 MA）

`paper_experiments/RQ6_single_layer_activation/exp6_single_layer_activation.py`

```bash
python RQ6_single_layer_activation/exp6_single_layer_activation.py \
    --model {MODEL} \
    --layers_to_scan all \
    --nsamples 30 \
    --seqlen 1024 \
    --savedir results/wikitext_run/RQ6/{MODEL}
```

输出：`{MODEL}_rq6_results.json`（内部循环 top-K 列表，**不是外部参数**）。

### 2.10 RQ5 macro：多层 macro v₁ 投影消除（新脚本）

`paper_experiments/RQ5_v_matrix_ablation/exp5_macro_v_ablation.py`（本轮新增）

```bash
python RQ5_v_matrix_ablation/exp5_macro_v_ablation.py \
    --model {MODEL} \
    --origin_layers "0,1,2,3,4,5" \
    --nsamples 30 \
    --savedir results/wikitext_run/RQ5_macro/{MODEL}
```

对模式 B 的 4 个模型（gpt2, opt_6.7b, qwen3_32b, qwen3.5_27b），origin_layers 从 0 到 `L_origin+2` 覆盖全部早期/聚合段：
- gpt2: `"0,1,2,3,4,5"`
- opt_6.7b: `"0,1,2,3"`
- qwen3_32b: `"40,41,42,43,44,45"`
- qwen3.5_27b: `"50,51,52,53,54,55,56"`

输出：`{MODEL}_macro_v_ablation_results.json`。

---

## 3. 批次执行

### 批次 0 — 前置修复（**本轮已做一半，剩余 3 项运行时修**，~1h）

已做（代码已改）：
- ✅ `lib/model_dict.py` 补齐
- ✅ `lib/load_model.py` glm4 走 fp32
- ✅ `lib/eval_utils.py` eval_ppl 参数化
- ✅ 新脚本 `exp5_macro_v_ablation.py`
- ✅ 新脚本 `run_rq345_origin_layer.sh`

运行时要做：
```bash
# 0.1 修 qwen2_7b RQ1（nsamples 30→60）
python RQ1_attention_contribution/exp1_feasibility_test.py \
    --model qwen2_7b --nsamples 60 --dataset wikitext \
    --savedir results/wikitext_run/RQ1/qwen2_7b_fixed

# 0.2 修 glm4_32b RQ1（fp32）
python RQ1_attention_contribution/exp1_feasibility_test.py \
    --model glm4_32b --nsamples 30 --dataset wikitext \
    --savedir results/wikitext_run/RQ1/glm4_32b_fp32
# （load_model.py 会自动检测 "glm4" 走 fp32，不需要额外 flag）

# 0.3 补 llama2_13b 的 RQ2b（从老仓库跑）
cd ../changeHead_massvieAcitve
python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py \
    --model llama2_13b --nsamples 20 --seqlen 4096 \
    --savedir "$PWD/../paper_experiments/results/wikitext_run/RQ2b/llama2_13b"
cd ../paper_experiments

# 得到 critical_layer 后，填进 run_rq345_origin_layer.sh 的 L_ORIGIN 表
```

**验收**：
- qwen2_7b 的 `table1_rq1.json` 里 `delta_top1_pct` 是数字
- glm4_32b 的 `baseline/results.json` 里 Top1 是有限数字
- llama2_13b 的 RQ2b 目录里能找到 `layer_N_disabled.json`，且有一层 `ΔMA` 显著（> 50%）

### 批次 1 — Pilot：GPT-J 全套（~30min）

GPT-J 是模式 A、起源层 L2，是最清洁的验证场景。Pilot 做 RQ3/4/5（单层）；**不需要 RQ6 macro**（因为单层已主导）。

```bash
bash run_rq345_origin_layer.sh "gptj_6b" all
```

**验收**（逐项核对 `results/wikitext_run/RQ*_origin/gptj_6b/*.json`）：

| 实验 | JSON 字段 | 阈值 | 论文参考值 |
|---|---|:-:|:-:|
| RQ3 | `cohens_d` | **> 0.5** | 论文实测功能词强聚集 |
| RQ4 | `sigma_ratio` (σ₁/σ₂) | **≥ 4.5** | GPT-J 起源层 η=**5.74** |
| RQ4 | `top_alignments[0].rank` | **< 50**（在 4096 维里） | 论文: MA 维度在 rank < 10 |
| RQ5 | `delta_ma.top1_mean_pct` | **≤ -85%** | 论文: **-99.1%** at L2 |

**达成后**：进入批次 2。
**未达成**：检查 `lib.get_mlp_down_proj` 对 GPT-J 的 `fc_out` 提取是否正确（见 exp6_macro_svd_full.py:56-58 参考）。

### 批次 2 — RQ2 补齐（~2h，可并行）

#### 2.1 RQ2a 补 19 个 + glm4_32b

**模式 B 的 4 个（最高优先）**：
```bash
for m in gpt2 opt_6.7b qwen3_32b qwen3.5_35b_a3b; do
    python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
        --model "$m" --nsamples 30 --dataset wikitext \
        --savedir "results/wikitext_run/RQ2a/$m"
done
```

**其余 15 个**（包含 glm4_32b，fp32 自动生效）：
```bash
for m in llama3.1_8b qwen1.5_14b qwen2.5_0.5b qwen2_7b \
         qwen3_0.6b qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b \
         qwen3.5_9b qwen3.5_27b yi_9b glm4_9b \
         qwen3_30b_a3b glm4_32b llama2_13b; do
    python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
        --model "$m" --nsamples 30 --dataset wikitext \
        --savedir "results/wikitext_run/RQ2a/$m"
done
```

**验收**：`delta_top1_pct ≤ -95%`（模式 A）/ `≤ -90%`（模式 B/中间，聚合起来 MLP 应近乎全部写入）。

#### 2.2 RQ2c ≡ RQ6.4 progressive（23 个，~1.5h）

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

**验收与模式精细判定**（每模型打开 `{m}_rq6_greedy.json` 看累积曲线）：
- top-1 累积 ΔMA ≥ -85% → **模式 A** 确认
- top-1+top-2 累积 ≥ -95%（但 top-1 < 85%）→ **双层主导（A + 调节型）**
- top-5 才 ≤ -80% → **模式 B** 确认
- top-K 非单调（中间回升）→ **有抑制层**，A + 调节混合型

### 批次 3 — RQ6 高优先 13（~2.5h，~2 卡并行可半）

**说明**：中间 9 + 模式 B 4 = 13 个（去重后）。每个模型跑 **macro-SVD 和 top-K 两个**。

```bash
# === 模式 B 4 个：跑 macro-SVD 和 top-K ===
for m in gpt2 opt_6.7b qwen3_32b qwen3.5_27b; do
    # 根据该模型的起源层范围设置 origin_layers
    case $m in
        gpt2)          OL="0,1,2,3,4" ;;
        opt_6.7b)      OL="0,1,2,3" ;;
        qwen3_32b)     OL="40,41,42,43,44,45" ;;
        qwen3.5_27b)   OL="50,51,52,53,54,55" ;;
    esac
    python RQ6_single_layer_activation/exp6_macro_svd_full.py \
        --model "$m" --origin_layers "$OL" --nsamples 20 --seqlen 512 \
        --savedir "results/wikitext_run/RQ6/$m"
    python RQ6_single_layer_activation/exp6_single_layer_activation.py \
        --model "$m" --layers_to_scan all --nsamples 30 --seqlen 1024 \
        --savedir "results/wikitext_run/RQ6/$m"
done

# === MoE 1 个（作 MoE 异常对照）===
python RQ6_single_layer_activation/exp6_macro_svd_full.py \
    --model qwen3.5_35b_a3b --origin_layers "37,38,39,40,41" \
    --nsamples 20 --seqlen 512 --savedir results/wikitext_run/RQ6/qwen3.5_35b_a3b
python RQ6_single_layer_activation/exp6_single_layer_activation.py \
    --model qwen3.5_35b_a3b --layers_to_scan all --nsamples 30 --seqlen 1024 \
    --savedir results/wikitext_run/RQ6/qwen3.5_35b_a3b

# === 中间 9 个：同样两个脚本 ===
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

**验收与模式精细判定**：

| macro σ₁/σ₂ | remove top-1 ΔMA | 判定 |
|:-:|:-:|---|
| ≥ 3 | ≤ -80% | **真模式 B（多层协作）** |
| 2 – 3 | -50% ~ -80% | 中间倾 B |
| 1 – 2 | > -50% | 中间倾 A |
| MoE ≈ 1 | 任意 | **MoE 异常（expert 间 mark 不共享，Tier C 专项）** |

产出：25 模型 × (A / B / 中间-A / 中间-B / MoE) 精确分类表。

### 批次 4 — RQ3 + RQ4 起源层 23 个（~5h 串行，~2.5h 双卡）

```bash
# Cart 1：RQ3
bash run_rq345_origin_layer.sh "" rq3

# Cart 2（另起终端）：RQ4
bash run_rq345_origin_layer.sh "" rq4
```

**验收**（模式 A 10 个）：

| 实验 | 指标 | 阈值 | 参考 |
|---|---|:-:|:-:|
| RQ3 | `cohens_d` | **> 0.5** | 论文要求显著 |
| RQ3 | `p_value` | < 1e-10 | 30 样本下易达成 |
| RQ4 | `sigma_ratio` | **≥ 3.0**（GPT-J 应 ≥ 4.5）| GPT-J 论文 5.74 |
| RQ4 | `top_alignments[0].rank` | **< 50** | MA 维度必须 rank 前几 |

**模式 B 4 个**（预期结果弱，验证"非单层主导"）：

| 实验 | 预期范围 | 含义 |
|---|:-:|---|
| RQ3 `cohens_d` | 0 – 0.3 | 单层弱 → 功能词特异性在多层合力中才显著 |
| RQ4 `sigma_ratio` | < 2 | 单层谱不集中 |

**中间 9 个**：介于两者之间，**结合批次 3 的 macro σ₁/σ₂ 定性**。

### 批次 5 — RQ6 低优先 10 个模式 A 补齐（~1.5h）

```bash
for m in bloom_7b1 falcon_7b gptj_6b llama3.1_8b yi_9b \
         qwen2.5_0.5b qwen2.5_7b qwen2_7b qwen3_0.6b qwen3_1.7b; do
    OL=$(python -c "print(','.join(str(x) for x in range(0, 6)))")  # L0-L5 通用
    python RQ6_single_layer_activation/exp6_macro_svd_full.py \
        --model "$m" --origin_layers "$OL" --nsamples 20 --seqlen 512 \
        --savedir "results/wikitext_run/RQ6/$m"
    python RQ6_single_layer_activation/exp6_single_layer_activation.py \
        --model "$m" --layers_to_scan all --nsamples 30 --seqlen 1024 \
        --savedir "results/wikitext_run/RQ6/$m"
done
```

对 yi_9b (L1)、qwen2.5_0.5b (L0)：OL="0,1,2" 即可。qwen3_0.6b/1.7b (L2)：OL="0,1,2,3"。可以按需微调，但 L0-L5 对大多数模式 A 都够用。

**预期**：macro σ₁/σ₂ 不一定比单层大（因为 A 本来就是单层主导）。这正是论文分 A/B 的**对照证据**。

### 批次 6 — RQ5 全部（~3-4h，**最后做**）

#### 6.1 模式 A + 中间 所有单层（22 个，~3h）

```bash
# 排除 glm4_32b、llama2_13b 这两个依赖前置的
bash run_rq345_origin_layer.sh "" rq5
```

**验收**：

| 模式 | `delta_ma.top1_mean_pct` 预期 |
|---|:-:|
| A 模式（10 个）+ 中间偏 A | **≤ -85%** |
| 中间偏 B（glm4_9b, qwen3.5_9b）| -60% ~ -85% |
| B 模式（4 个）单层 | > -30%（**弱是对的**，说明需要 macro 版本）|
| MoE（qwen3_30b_a3b, qwen3.5_35b_a3b）单层 | 预期弱（作 MoE 异常对照）|

#### 6.2 模式 B 4 个的 macro v₁ 投影消除（~1h）

```bash
bash run_rq345_origin_layer.sh "gpt2 opt_6.7b qwen3_32b qwen3.5_27b" rq5_macro
```

**验收**：`delta_ma.top1_mean_pct ≤ -80%`。这对比单层版本（预期 > -30%），直接证明模式 B 的 MA 在多层合力里。

**可选扩展**：若批次 3 RQ6 确认 glm4_9b 是真 B（macro η=4.33 已高），也跑：
```bash
python RQ5_v_matrix_ablation/exp5_macro_v_ablation.py \
    --model glm4_9b --origin_layers "15,16,17,18,19" --nsamples 30 \
    --savedir results/wikitext_run/RQ5_macro/glm4_9b
```

#### 6.3 PPL 附加验证（信息分轨，可选）

目前 exp5_v_ablation.py 和 exp5_macro_v_ablation.py 没自动测 PPL。要验证"MA 消 -85% 但 PPL 不涨"，需要在脚本里加一行：
```python
from lib.eval_utils import eval_ppl
ppl_baseline = eval_ppl('wikitext', model, tokenizer, seed=0)
# ... ablation ...
ppl_ablated = eval_ppl('wikitext', model, tokenizer, seed=0)
```
这是加法修改、非必需；本轮可以先不加，等主实验数据出来后再补。

### 批次 7 — 数据补洞（~2h）

#### 7.1 llama2_13b（依赖批次 0.3 给出 L_origin）

假设批次 0.3 得到 L_origin_llama2_13b = X：
1. 把 `run_rq345_origin_layer.sh` 的 L_ORIGIN 数组加 `[llama2_13b]=X`
2. 跑：
```bash
bash run_rq345_origin_layer.sh "llama2_13b" all
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
    --model llama2_13b --nsamples 30 --dataset wikitext \
    --savedir results/wikitext_run/RQ2a/llama2_13b
python RQ6_single_layer_activation/exp6_macro_svd_full.py \
    --model llama2_13b --origin_layers "0,1,2,3,4" --nsamples 20 \
    --savedir results/wikitext_run/RQ6/llama2_13b
```

#### 7.2 glm4_32b（fp32 全套）

确认批次 0.2 的 RQ1 已有效后：
```bash
# 把 L_ORIGIN 的 glm4_32b 设好
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py --model glm4_32b \
    --nsamples 30 --savedir results/wikitext_run/RQ2a/glm4_32b
cd ../changeHead_massvieAcitve
python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py \
    --model glm4_32b --nsamples 20 --seqlen 4096 \
    --savedir "$PWD/../paper_experiments/results/wikitext_run/RQ2b/glm4_32b"
cd ../paper_experiments

# 之后 RQ3/4/5/6 同样方法跑
```

---

## 4. 并行化机会

| 批次 | 可并行维度 | 建议 |
|:-:|---|---|
| 0 | 三个修复项并行 | 3 终端 |
| 1 | 串行（Pilot） | 不并行 |
| 2.1 RQ2a | 19 个模型可分 2 卡 | 2 卡并行 |
| 2.2 RQ2c | 23 个可分 2 卡 | 2 卡并行 |
| 3 RQ6 | 13 个可分 2 卡 | 2 卡并行 |
| 4 | RQ3 / RQ4 **独立**，同时跑 | 双卡 |
| 5 | 单卡顺序 | 串行 |
| 6 | RQ5 单层顺序；6.2 macro 可和 6.1 并行 | 部分并行 |
| 7 | 串行 | 串行 |

---

## 5. 总工时估算

| 批次 | 串行工时 | 双卡并行 |
|:-:|:-:|:-:|
| 0 前置修复（剩余 3 项）| 1h | 0.5h |
| 1 Pilot | 0.5h | 0.5h |
| 2 RQ2 补齐 | 2h | 1.2h |
| 3 RQ6 高优先 | 2.5h | 1.5h |
| 4 RQ3 + RQ4 | 5h | **2.5h** |
| 5 RQ6 低优先 | 1.5h | 1.5h |
| 6 RQ5 全部（含 macro）| 4h | 2.5h |
| 7 补洞 | 2h | 2h |
| **合计** | **~18.5h** | **~12.2h** |

---

## 6. 交付物清单（全跑完后）

1. `results/wikitext_run/RQ{2a,2b,2c,3_origin,4_origin,5_origin,5_macro,6}/` 全部 25 模型新数据
2. 新 `ALL_EXPERIMENTS_SUMMARY_v2.json`（合并所有新结果）
3. 更新后的 `docs/PROGRESS_MATRIX.md`：所有 ◐/·/⚠ 变 ✓
4. 更新后的 `docs/TODO_EXPERIMENTS.md`：所有任务 ✓
5. **25 模型 × (A/B 模式 × Gen/Sup 调节) 精确二维分类表**（核心论点数据）
6. 模式 A 10 个起源层 RQ5 平均 ΔMA（应 ≤ -85%）
7. 模式 B 4 个 macro RQ5 平均 ΔMA（应 ≤ -80%）
8. MoE 2 个 macro σ₁/σ₂ ≈ 1.0 的异常证据（Tier C 伏笔）

---

## 7. 中止/回滚条件

| 情况 | 动作 |
|---|---|
| 批次 1 Pilot 不过验收（GPT-J RQ5 ΔMA > -70%）| **停**，查 lib.get_mlp_down_proj 对 GPT-J fc_out 的处理 |
| 批次 2 RQ2a 某模型 ΔMA > -90% | 单模型排查，不影响其他 |
| 批次 3 macro σ₁/σ₂ 全 < 2 | 重查 origin_layers 范围是否覆盖了起源层 |
| 批次 4 模式 A 的 Cohen's d 仍为负 | **严重异常**，回头确认 RQ2b 的 critical_layer 是不是起源层 |
| 批次 6.1 模式 A 的 RQ5 不到 -85% | 检查 SVD 算法（是否 full_matrices=False）+ random seed |
| 批次 6.2 模式 B macro RQ5 不到 -70% | 调大 origin_layers 范围，或检查 v₁ 提取的投影方向 |

---

## 8. 跑完后的论文映射

| 论文章节 | 数据来源 |
|---|---|
| Table 1（主结果）| 25 模型 RQ1-RQ6 核心指标汇总 |
| Section 3.1 MA 生成 | 批次 2（RQ2a + RQ2c）|
| Section 3.2 功能词 mark | 批次 4（RQ3 起源层）|
| Section 3.3 SVD 几何 | 批次 4（RQ4 起源层）|
| Section 3.4 因果验证 | 批次 6（RQ5 起源层 + RQ5 macro）|
| Section 3.5 多层协作（模式 B）| 批次 3（RQ6 macro-SVD + top-K）|
| Section 4 调节机制 | RQ1 Gen/Sup 分类 + RQ2c 累积曲线异常回升 |
| Section 5 信息分轨（可选）| 6.3 PPL vs ΔMA 对照 |

---

按本 v2 手册执行，约 **12.2h 双卡 / 18.5h 单卡** 可以产出完整数据。
