# Master Plan — RQ 重跑全局总览（2026-04-21）

> 本文件整合所有子 plan：
> - `DEBUG_SUMMARY.md`（7 bug 修复总结）
> - `SCRIPT_FIXES.md`（bug 修复详情）
> - `EXPERIMENT_PLAN.md §全局重跑计划汇总`（原 5 阶段 plan）
> - `SECONDARY_SERVER_PLAN.md`（另一台服务器的 6 模型）
> - `fixes/README.md`（部署包）
> - `.patrol_state.json` 和 cron `6c4728ac`（实时巡检）
>
> 如果只能读一篇文档，读这一篇。

---

## 一、样本划分（26 模型 = 24 dense + 2 MoE）

```
总 26 模型
├── Tier A+B: 24 dense 模型（主样本，本轮全部跑）
│   ├── Primary 本地权重（15 个，已在 server A 跑中）
│   ├── Secondary HF 下载（6 个，预备 server B 跑）
│   └── 悬挂（3 个：gpt2 + qwen2.5_0.5b 很小 + glm4_32b 无源）
└── Tier C: 2 MoE 模型（附录，本轮 skip）
    ├── qwen3_30b_a3b
    └── qwen3.5_35b_a3b
```

### 详细模型分布

| Server | 数量 | 模型 | 状态 |
|---|:-:|---|---|
| **Primary 117.50.223.194** | **15 dense** | qwen3_0.6b, qwen3_1.7b, qwen3_4b, llama2_7b_chat, qwen2_7b, qwen2.5_7b, qwen3_8b, llama3.1_8b, qwen3.5_9b, yi_9b, glm4_9b, qwen3_14b, qwen1.5_14b, qwen3.5_27b, qwen3_32b | 🟢 RQ3 跑中 |
| **Secondary 8.138.30.52** | **6 dense** | opt_6.7b, gptj_6b, bloom_7b1, falcon_7b, mistral_7b_v03, llama2_13b | ⏸ 待启动（按 SECONDARY_SERVER_PLAN）|
| 小模型待决 | 2 dense | gpt2, qwen2.5_0.5b | ⏸ 任一 server 都能快速补 |
| 无源 | 1 dense | glm4_32b | ⚠ 无本地 + 无 HF，需手动找权重 |
| MoE（Tier C）| 2 | qwen3_30b_a3b, qwen3.5_35b_a3b | ⏸ 本轮 skip |

**主样本** = 15 (primary) + 6 (secondary) = **21 个已规划**；gpt2/qwen2.5_0.5b 小模型补在任一 server = **最终 23**；glm4_32b 若拿不到 = **22 个主样本**。论文论点 "≥20 个支持" 仍满足。

---

## 二、已完成的阶段

### Stage 0：脚本 bug 修复（COMPLETED 2026-04-21）

7 个 bug 中 6 个已修、1 个推迟：

| Bug | 状态 | 详情 |
|:-:|:-:|---|
| B1 | ✅ | `add_token` 存所有 token（含内容词、结构 token）|
| B2 | ⏸ | MoE `SparseMoeBlock` 无 `.up_proj` — Tier C 专项 |
| B3 | ✅ | `get_mlp_submodules` 加 glm4/yi |
| B4 | ✅ | RQ6 `get_mlp_down_proj` 加 glm4/yi |
| B5 | ✅ | RQ6 `get_critical_layer` 读 L_ORIGIN.json |
| B6 | ✅ | RQ6 baseline 扫所有层找真 MA |
| B7 | ✅ | RQ2a `MLPDisableHook` 处理 MoE tuple |
| **B8** | ✅ | monkey_patch/modify_llama.py 匹配 transformers 5.x API（新发现） |
| **B9** | ✅ | `enable_custom_block/attention` 加 glm4/yi（新发现）|

参考：`DEBUG_SUMMARY.md`、`SCRIPT_FIXES.md`、`fixes/README.md`、`fixes/sentinel_test.md`

**部署包**：`paper_experiments/fixes/`（5 个 Python 文件 + 5 个 README + sentinel + pipeline driver + models_local.txt）

---

## 三、进行中 / 待启动阶段

### Stage 1：RQ1/RQ2 小补（4 模型）—— **悬挂**

| 任务 | 模型 | 说明 | 预计 |
|:-:|---|---|:-:|
| RQ2a 补 | opt_6.7b | ANOMALY 模型判因必需 | 3 min |
| RQ1 补 | qwen2_7b | `--nsamples 30→60` 修 baseline≈0 除零 | 15 min |
| RQ2a+2b 补 | llama2_13b | primary 本地无，在 secondary 跑 | 23 min |
| RQ2b+2c 补 | llama2_7b_chat | L_ORIGIN 缺失（RQ2c 没跑过）| 20 min |

**合计 ~1h**。分摊：
- **opt_6.7b**, **llama2_13b** → secondary server
- **qwen2_7b**, **llama2_7b_chat** → primary server（本地有）

### Stage 2：RQ3 + RQ4 结构 token 重跑（**进行中**）

**Primary**（🟢 跑中）：

```bash
# 已启动 06:31，预计 1.5-2h
nohup python3 paper_experiments/fixes/run_pipeline.py \
  --models_file paper_experiments/fixes/models_local.txt \
  --rqs RQ3 --nsamples 30 \
  > paper_experiments/logs/stage2_rq3_20260421_063106.log 2>&1 &
```

**进度**（每 10 min 自动巡检，cron `6c4728ac`）：
- Patrol #3 @ 06:52: **done=5/15**（qwen3_0.6b/1.7b/4b/2_7b ✓，llama2_7b_chat skip）
- 输出验证通过：content > func > struct（B1 生效），conc/align 各 7330 entries

**Primary 接下来**（RQ3 完成后）：

```bash
# RQ4 全 15 模型（预计 1.5-2h）
python3 paper_experiments/fixes/run_pipeline.py \
  --models_file paper_experiments/fixes/models_local.txt \
  --rqs RQ4 --nsamples 30
```

**Secondary**（⏸ 待启动，按 `SECONDARY_SERVER_PLAN.md`）：
- 一次性跑 `--rqs RQ1 RQ2a RQ2c RQ3 RQ4 RQ5s RQ5m RQ6`（48 runs，~3.5h）

### Stage 2 分析：Top-K aggregator（数据齐后写）

RQ3/RQ4 产出 JSON 里有 `word_stats.is_function/is_structural`，但**还需要一个离线脚本**：
- 读每个模型的 `exp5_detailed_results.json`
- 按 `|h₂·v₁|` 排序，统计 Top-10 / 50 / 100 里 func/struct/content 占比
- 输出跨模型汇总表

**计划**：Stage 2 数据齐后我写 `paper_experiments/fixes/analyze_topk.py`（~30 min）。

### Stage 3：RQ6 exp6 全 21 dense 重跑（⏸ 待）

```bash
# Primary
python3 paper_experiments/fixes/run_pipeline.py \
  --models_file paper_experiments/fixes/models_local.txt \
  --rqs RQ6 --nsamples 30
# Secondary: 已包含在 SECONDARY_SERVER_PLAN Phase 4 的 --rqs 列表里
```

**预计**：每 server ~2-3h。baseline 现在扫全层（B5+B6 修复），remove/keep_top_K 百分比可信。

### Stage 4：RQ5 single + macro 补（⏸ 待）

```bash
# Primary（15 模型）+ Secondary（6 模型）
python3 paper_experiments/fixes/run_pipeline.py \
  --models_file <primary or secondary>.txt \
  --rqs RQ5s RQ5m --nsamples 30
```

**预计**：每 server ~2h。

### Stage 5：MoE Tier C 专项（主结论定稿后，低优先级）

4 类 bug 修复（B2 + ...）+ per-expert 分析 + qwen3_30b/qwen3.5_35b 专项重跑。本轮**不做**。

---

## 四、全局时间预算

```
Stage 0 (修 bug)         ✅ DONE
Stage 1 (RQ1/RQ2 补)     ~1h (分摊 primary + secondary)
Stage 2a (RQ3 × 21)      primary 2h + secondary 0.75h（并行）
Stage 2b (RQ4 × 21)      primary 1.5h + secondary 0.75h（并行）
Stage 2c (Top-K 分析)    0.5h（CPU 本地）
Stage 3 (RQ6 × 21)       primary 2h + secondary 0.75h（并行）
Stage 4 (RQ5 × 21)       primary 2h + secondary 1h（并行）
─────────────────────────────────────────
总计（两 server 并行）    ~10-12h
单 server 串行            ~18-20h
```

**Secondary server 上一次性跑 Phase 4（RQ1-6 全 8 阶段）≈ 3.5h**；Primary 分段跑，按上述顺序。

---

## 五、实时监控

### Primary 巡检

- **Cron `6c4728ac`**：每 10 min 自动检查 + 报告
- **State file**：`.patrol_state.json`（本地）
- 手动查：
  ```bash
  sshpass -p 'j053v429E1a8LNQs' ssh -p 23 root@117.50.223.194 "tail -30 /root/changeHead_massvieAcitve/paper_experiments/logs/stage2_rq3_20260421_063106.log"
  ```

### Secondary（待启动后）

- tmux session：`rqsweep`
- 日志：`paper_experiments/logs/secondary_sweep.log`
- 新 cron（启动后可加）：每 10 min 巡检

### 汇总命令

```bash
# 跨两个 server 的统一进度
for server in "j053v429E1a8LNQs@117.50.223.194:23" "XinAn234\!@8.138.30.52:6007"; do
  IFS=: read pw host port <<< "$server"
  IFS=@ read pw user <<< "$pw"
  sshpass -p "$pw" ssh -p "$port" "$user@$host" "ls -d ~/ma*/paper_experiments/results/wikitext_run_2026_04_21/RQ*/*/  2>/dev/null | wc -l"
done
```

---

## 六、合并 & 收尾

### 数据合并

Primary 和 secondary 输出路径**完全一致**：
```
paper_experiments/results/wikitext_run_2026_04_21/
├── RQ1/<model>/...
├── RQ2a/<model>/...
├── RQ3/<model>/exp5_detailed_results.json
├── RQ4/<model>/exp3_detailed_results.json
├── RQ5s/<model>/...
├── RQ5m/<model>/...
└── RQ6/<model>/v_ablation_results.json
```

从 secondary rsync 到 primary：
```bash
rsync -avP ~/ma/paper_experiments/results/wikitext_run_2026_04_21/ \
    root@117.50.223.194:/root/changeHead_massvieAcitve/paper_experiments/results/wikitext_run_2026_04_21/
```

### 主结论判决标准（24 dense 模型，扣 MoE 2 个）

**H₁-H₅ 支持度**：

| H | 指标 | 阈值 | 当前/预期 |
|:-:|---|:-:|:-:|
| H₀（atten 非起源）| RQ1 residual% > 0 | 25/25（已证伪）| ✅ DONE |
| H₁（MLP 是起源）| RQ2a retain% ≤ 10 | ≥ 20/24 | ~20 预期 |
| H₂（结构 token mark）| RQ3 Top-K struct 占比 | > 50% | 待 Stage 2 |
| H₃（谱集中）| RQ4 σ₁/σ₂ 或 σ₁ 绝对值 | ≥ 3 或 σ₁ 大 | 待 Stage 2 |
| H₄（multi-layer 合力）| RQ6 remove_1 < 50%, keep_1 > 90% | ≥ 15/24 | 待 Stage 3 |
| H₅（v₁ 因果必要）| RQ5 ΔMA ≤ -85% | ≥ 15/24 | 待 Stage 4 |

**定稿标准**：≥ 20/24 模型支持 H₁-H₅ 主链；qwen3.5 dense 家族（Tier D 候选）+ opt_6.7b（Tier E）单独讨论。

---

## 七、关键文件快速索引

```
仓库根 /root/changeHead_massvieAcitve/（两 server 同布局）
├── paper_experiments/
│   ├── fixes/                              ← 修复部署包
│   │   ├── README.md                       入口
│   │   ├── lib/model_utils.py              B3+B9
│   │   ├── RQ2_mlp_source/.../exp2a...py  B7
│   │   ├── RQ3_function_words/.../exp5...py B1
│   │   ├── RQ6_v_ablation/.../exp6...py    B4+B5+B6
│   │   ├── monkey_patch/modify_llama.py    B8
│   │   ├── sentinel_test.sh                6 check
│   │   ├── run_pipeline.py                 subprocess driver
│   │   ├── models_local.txt                primary 15
│   │   └── models_secondary.txt            secondary 6（待创建）
│   ├── docs/
│   │   ├── MASTER_PLAN.md                  ← 本文
│   │   ├── SECONDARY_SERVER_PLAN.md        另台服务器详细
│   │   ├── DEBUG_SUMMARY.md                6+2 bug 修复总结
│   │   ├── SCRIPT_FIXES.md                 bug 逐个详情
│   │   ├── EXPERIMENT_PLAN.md              总讨论记录
│   │   └── EXPERIMENT_RESULTS.md           26 模型数据汇总
│   ├── origin_layer/output/L_ORIGIN.json  25 模型起源层
│   └── results/wikitext_run_2026_04_21/   统一结果目录
```

---

## 八、Session-level 巡检与任务

- Cron `6c4728ac`：每 10 min 自动巡检 primary（session 内）
- TodoWrite：跟踪本 plan 各阶段状态
- `.patrol_state.json`：巡检增量记录

**取消巡检**：说 "停止巡检" 或 `CronDelete 6c4728ac`
**任一 server 完成后**：告诉我，我发起合并 + Top-K 分析 + 主结论判决

---

## 变更日志

| 日期 | 改动 |
|---|---|
| 2026-04-21 | 初版，整合 5 个子 plan |
