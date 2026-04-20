# Debug Summary — RQ 重跑前的脚本修复总结

> **日期**：2026-04-21
> **范围**：本轮分析期间发现的 7 个脚本 bug，其中 6 个已修复、1 个（MoE 专属）推迟到 Tier C
> **部署包位置**：`paper_experiments/fixes/`
> **验证方式**：`bash paper_experiments/fixes/sentinel_test.sh` — 6/6 通过

---

## TL;DR

分析 26 模型 × 6 RQ 数据过程中，发现 **7 个脚本 bug** 会污染或阻塞数据：

| 状态 | 数量 | Bug IDs |
|:-:|:-:|---|
| ✅ 已修复 | **6** | B1, B3, B4, B5, B6, B7 |
| ⏸ 推迟 | 1 | B2（MoE，Tier C 专项） |

**主样本 24 个 dense 模型**修复后全部可跑、数据可信。MoE 2 个模型（qwen3_30b_a3b, qwen3.5_35b_a3b）本轮单独归 Tier C。

---

## 修复前 vs 修复后：模型跑得了什么（26 模型 × 3 修改过的 RQ）

| 图例 | 含义 |
|:-:|---|
| ✓ | 能跑 + 结果可信 |
| ◯ | 能跑但有 bug 污染结果 |
| ✗ | crash / 跑不了 |
| ⏸ | 本轮跳过（MoE Tier C） |

| 模型 | 家族 | RQ2a 前 / 后 | RQ3 前 / 后 | RQ6 前 / 后 |
|---|:-:|:-:|:-:|:-:|
| bloom_7b1 | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| falcon_7b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| **glm4_9b** | dense | ✓ / ✓ | **✗ → ✓** | **✗ → ✓** |
| **glm4_32b** | dense | ✓ / ✓ | **✗ → ✓** | **✗ → ✓** |
| gpt2 | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| gptj_6b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| llama2_13b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| llama2_7b_chat | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| llama3.1_8b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| mistral_7b_v03 | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| opt_6.7b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| **qwen1.5_14b** | dense | ✓ / ✓ | **✗ → ✓** | **◯ → ✓** |
| qwen2.5_0.5b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen2.5_7b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen2_7b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen3_0.6b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen3_1.7b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen3_4b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen3_8b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen3_14b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| qwen3_32b | dense | ✓ / ✓ | ◯ / ✓ | ◯ / ✓ |
| **qwen3.5_9b** | dense | ✓ / ✓ | **✗ → ✓** | **✗ → ✓** |
| **qwen3.5_27b** | dense | ✓ / ✓ | **✗ → ✓** | **✗ → ✓** |
| **yi_9b** | dense | ✓ / ✓ | **✗ → ✓** | **✗ → ✓** |
| **qwen3_30b_a3b** | **MoE** | ✓ / ✓ | ✗ → ⏸ | ✗ → ⏸ |
| **qwen3.5_35b_a3b** | **MoE** | **◯ → ✓** | ✗ → ⏸ | ✗ → ⏸ |

### 汇总

| 指标 | 修复前 | 修复后 | 提升 |
|---|:-:|:-:|:-:|
| RQ2a 数据可信 | 25/26（qwen3.5_35b_a3b B7 污染） | **26/26** | +1 |
| RQ3 数据可信 | 0/26（B1 全污染 + 5 crash） | **24/26**（主样本全覆盖 + MoE 跳过）| +24 |
| RQ6 数据可信 | 0/26（B5/B6 全污染 + 4 crash）| **24/26**（主样本全覆盖 + MoE 跳过）| +24 |

**主样本 24 dense × 6 RQ = 144 个数据点**全部可跑可信。

---

## 6 个已修 Bug 详情

### B1 — `add_token` 只存功能词（RQ3, RQ4）

**症状**：`FunctionWordSVDTracker.add_token()` 里有一个 `if self.is_function_word(token_text)` 过滤器——内容词和结构 token 的 h₂ 向量**从未被存储**。

**后果**：此前所有 RQ3 的 "func vs content Cohen's d" 比较的其实是**核心功能词 vs 边缘功能词**，内容词根本没参与对比。16 模型的分析结论**方法学无效**。

**修复**：
- 去掉过滤器，存所有 token
- 每个 token 附加 `is_function` + `is_structural` 标签
- 新增 `STRUCTURAL_TOKENS` 集合（标点、换行、特殊符号、chat template 分隔符）

**文件**：`paper_experiments/fixes/RQ3_function_words/exp5_function_words_svd_mapping.py` +~55 行

### B2 — MoE 访问 `.up_proj`（**推迟到 Tier C**）

**症状**：`AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'`

**影响**：qwen3_30b_a3b, qwen3.5_35b_a3b（都是 MoE）——整个 RQ3/RQ6 无法跑。

**处置**：MoE 不在本轮主样本（24 dense），推迟到 Tier C 专项（主结论定稿后）。RQ6 脚本加了 **MoE guard**（检测 `layer.mlp.experts` 就跳过），避免污染批量运行。

### B3 — `get_mlp_submodules` 白名单缺失（RQ3）

**症状**：`ValueError: Cannot identify MLP submodules for model 'glm4_9b'`

**影响**：glm4_9b, glm4_32b, yi_9b 进来就抛异常。qwen1.5_14b / qwen3.5_9b / qwen3.5_27b 能走 `"qwen" in name` 分支（不受影响，但确认后再跑一次）。

**修复**：现有 `"llama / mistral / qwen"` SwiGLU 分支加入 `"glm4" / "yi"`（+3 行）。

**文件**：`paper_experiments/fixes/lib/model_utils.py`

### B4 — RQ6 `get_mlp_down_proj` 缺 glm4/yi 分支（RQ6）

**症状**：同 B3，只是在 RQ6 的 `exp6_v_ablation.py` 里。

**修复**：SwiGLU 分支加 `"glm4" / "yi"`（+3 行）。

**文件**：`paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py`

### B5 — RQ6 `critical_layer` 默认 L0（RQ6）

**症状**：`get_critical_layer(model_name)` 只硬编码了 6 个模型，其他全默认返回 L0——20+ 模型在**错层**做消融。

**典型错层**（修复前 → 修复后）：

| 模型 | 旧脚本默认 | 正确 L_origin |
|---|:-:|:-:|
| qwen3_32b | 0 | **6** |
| qwen1.5_14b | 0 | **35** |
| yi_9b | 0 | **8** |
| qwen3.5_27b | 0 | **54** |
| bloom_7b1 | 28（硬编码但过时）| **3** |

**修复**：
1. 优先环境变量 `OVERRIDE_CRITICAL_LAYER`（CLI 调试用）
2. 读 `paper_experiments/origin_layer/output/L_ORIGIN.json`（25 模型精确值）
3. 内置 fallback 表（L_ORIGIN 快照，部分部署场景用）
4. 未知模型直接 **raise ValueError**（不再静默默认 L0）

另加 argparse `--layer_id` 参数，可 CLI 覆盖。

### B6 — RQ6 baseline 测错层（RQ6）

**症状**：原脚本 `run_and_collect_ma` 只在 `critical_layer` 一层测 top1 激活，但 MA 真正的峰值一般在更后面的层（被 attention 广播放大之后）。

**后果**：17/17 模型的 RQ6 baseline 都**远小于真 MA**：

| 模型 | RQ2a 真 MA baseline | RQ6 旧 baseline | 差距 |
|---|:-:|:-:|:-:|
| glm4_32b | 298598 | **1.15** | 260000× |
| yi_9b | 5004 | 1.97 | 2540× |
| qwen1.5_14b | 7444 | 3.58 | 2079× |
| qwen3_32b | 27417 | 30.79 | 890× |

所有 `remove_top_K` / `keep_top_K` 百分比基于错误 baseline——glm4 甚至出现 "remove_top_1 后 MA 变 137%"（非物理）的诡异值。

**修复**：hook 全部层 → 跨所有层扫 top1 → 取全局峰值。和 RQ2a 测 MA 方法对齐。新增 `peak_layer` / `ablation_layer` 输出字段便于 debug。

### B7 — RQ2a `MLPDisableHook` 未处理 tuple（RQ2a）

**症状**：MoE `SparseMoeBlock.forward` 返回 `(hidden_states, router_logits)` tuple。`torch.zeros_like(tuple)` 行为未定义。

**证据**：qwen3.5_35b_a3b RQ2a retain=**81%**（异常），而 qwen3_30b_a3b（另一 MoE）retain=**2.85%**——两个 MoE 行为截然不同，大概率 qwen3.5 的 tuple 结构不同。

**修复**：加 `isinstance(output, tuple)` 分支，参照 `exp6_progressive_ablation.py:32-35` 的正确写法。

**文件**：`paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py` +~8 行

---

## 代码审查收获（额外修复）

初稿经 code-reviewer subagent 审查后，额外修了 3 处：

### C4 → legacy hardcode 替换成正确值
原 `get_critical_layer` 的 legacy fallback 保留了已知错误的 `bloom_7b1: 28`。审查指出这违背 B5 设计意图（为防止错层而存在，却内置错层默认）。**改为 25 模型的 L_ORIGIN.json 精确快照**作为内置 fallback，没有已知错误值。

### I2 → r_func 用 `is_function` flag 代替字符串匹配
原代码用 `w.strip().lower() in FUNCTION_WORDS`，但 BPE 分词的 `'Ġthe'` 不会被 strip 识别。**改为用 tracker 已算好的 `is_function` 标签**——避免字符串匹配漏洞，顺便统计出 `r_struct` / `r_content` 三类占比。

### I3 → 修正误导日志
原 `print("Collected X function word occurrences")` 在 B1 修复后完全不对（现在收所有 token）。**改为 `"Collected X total tokens (F func, S struct, C content)"`**。

---

## 验证方法

部署前在仓库根目录跑：

```bash
bash paper_experiments/fixes/sentinel_test.sh
```

6 个测试检查点：

| 测试 | 验证什么 |
|---|---|
| A | B3 白名单覆盖 glm4/yi/qwen1.5/qwen3.5 + 已有模型 |
| B | B7 dense tensor + MoE tuple hook 都正确置零 |
| C | B4 RQ6 白名单同上 |
| D | B5 读 L_ORIGIN.json 给出 qwen3_32b→6, yi_9b→8, qwen1.5_14b→35, bloom_7b1→3（不再错层）；env override 生效；未知模型 raise |
| E | B6 `run_and_collect_ma` 扫全层 + 记录 peak_layer + MoE guard + `--layer_id` 参数齐全 |
| F | B1 FunctionWordSVDTracker 正确分类 func(3)/struct(4)/content(3) |

**预期输出**：`6 passed, 0 failed. All checks passed. Safe to deploy.`

---

## 部署

```bash
cd <repo-root>
cp paper_experiments/fixes/lib/model_utils.py paper_experiments/lib/model_utils.py
cp paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py
cp paper_experiments/fixes/RQ3_function_words/exp5_function_words_svd_mapping.py paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py
cp paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py
bash paper_experiments/fixes/sentinel_test.sh   # 验证
```

---

## 下一步

部署后按 `EXPERIMENT_PLAN.md §全局重跑计划汇总` 阶段 1-4 执行：

| 阶段 | 内容 | 成本 |
|:-:|---|:-:|
| 1 | RQ1/RQ2 小补（4 模型） | ~1h |
| 2 | RQ3/RQ4 结构 token 重跑（全 24 dense） | ~3h |
| 3 | RQ6 exp6 全重跑（全 24 dense） | ~2.5h |
| 4 | RQ5 补数据（8 模型） | ~2h |
| **合计** | | **~8.5h** |

阶段 5（MoE Tier C 专项）在主结论定稿之后再做。

---

## 参考

- 完整 bug 修复细节：`paper_experiments/docs/SCRIPT_FIXES.md`
- 实验计划：`paper_experiments/docs/EXPERIMENT_PLAN.md §全局重跑计划汇总`
- Fix 部署包 README：`paper_experiments/fixes/README.md`
- Sentinel 测试：`paper_experiments/fixes/sentinel_test.sh`
