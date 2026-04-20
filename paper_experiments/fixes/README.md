# 脚本修复部署包（Fixes Package）

> 本目录包含 7 个脚本 bug 的修复版本，用于本轮 RQ3/4/6 重做。**本地修好后用户自行拷贝到服务器**。

## 快速部署

把本目录下每个文件**按对应路径覆盖**到 `paper_experiments/` 对应位置（见下面的"文件替换对照表"）。然后跑 `sentinel_test.sh` 验证。

```bash
# 示例（用户在服务器上执行）
cd ~/ma  # 仓库根目录
# 对照 README 下面的表，逐个 cp
cp paper_experiments/fixes/lib/model_utils.py paper_experiments/lib/model_utils.py
cp paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py
# ... 其他文件同理
# 然后验证
bash paper_experiments/fixes/sentinel_test.sh
```

---

## Bug 列表（6 修 + 1 推迟 = 7 总）

> **B2（MoE 适配）** 本轮不修——MoE 2 个模型（qwen3_30b_a3b, qwen3.5_35b_a3b）已归 Tier C，不纳入主结论。等主结论定稿后另开 Tier C 专项脚本。
> **24 dense 模型主样本完全不受影响**。

| Bug | 症状 | 影响 RQ | 修复状态 |
|:-:|---|---|:-:|
| **B1** | `add_token` 只存功能词，丢内容词 | RQ3, RQ4 | ✅ 2026-04-21 |
| **B2** | MoE `SparseMoeBlock` 无 `.up_proj/.down_proj` | RQ3 | ⏸ 本轮不修（Tier C 专项）|
| **B3** | `get_mlp_submodules()` 缺 glm4/yi 白名单 | RQ3 | ✅ 2026-04-21 |
| **B4** | RQ6 `get_mlp_down_proj()` 缺 glm4/yi 分支 | RQ6 | ✅ 2026-04-21 |
| **B5** | RQ6 `get_critical_layer()` 默认 L0，不读 L_origin | RQ6 | ✅ 2026-04-21 |
| **B6** | RQ6 baseline 只在 critical_layer 测，非真 MA | RQ6 | ✅ 2026-04-21 |
| **B7** | RQ2a `MLPDisableHook` 未处理 tuple 输出 | RQ2a | ✅ 2026-04-21 |

---

## 文件替换对照表

| # | 修复文件（本包里的路径）| 替换到服务器的路径 | 修的 Bug |
|:-:|---|---|:-:|
| 1 | `fixes/lib/model_utils.py` | `paper_experiments/lib/model_utils.py` | B3 |
| 2 | `fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py` | `paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py` | B7 |
| 3 | `fixes/RQ3_function_words/exp5_function_words_svd_mapping.py` | `paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py` | B1（STRUCTURAL_TOKENS 已内联，无额外文件）|
| 4 | `fixes/RQ6_v_ablation/exp6_v_ablation.py` | `changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py` | B4 + B5 + B6 |
| 5 | `fixes/sentinel_test.sh` | （本地执行验证）| 验证 |

---

## 验证方法

修复后在服务器上跑 `sentinel_test.sh`——3 个 sentinel 模型（**gpt2** 代表 dense、**glm4_9b** 代表新加支持的家族、**qwen3_30b_a3b** 代表 MoE）× 3 个改动 RQ（**RQ2a / RQ3 / RQ6**），快速检查无回归。

预期结果：
- 全部**不报错**
- RQ3 的 `word_stats` 里 **content words 数量 > func words 数量**（B1 修复验证）
- RQ6 的 **baseline 和 RQ2a baseline 量级接近**（不是差几千倍；B6 修复验证）
- MoE 模型**不再 AttributeError**（B2/B4 修复验证）

详见 `sentinel_test.sh` 注释。

---

## 重跑顺序（修复后）

按依赖链跑，对应 `EXPERIMENT_PLAN.md §全局重跑计划汇总` 的阶段 1-4：

1. **阶段 1**：RQ1/RQ2 小补（opt_6.7b RQ2a、qwen2_7b RQ1、llama2_13b RQ2a+2b、llama2_7b_chat RQ2b+2c）—— ~1h
2. **阶段 2**：RQ3+RQ4 全 26 模型结构 token 重跑 —— ~3h
3. **阶段 3**：RQ6 exp6 全 26 模型重跑 —— ~2.5h
4. **阶段 4**：RQ5 补 8 缺口模型 —— ~2h

**合计 ~8.5h**（可并行则更短）。

Tier C（MoE）专项放最后，等主结论定稿（阶段 5）。

---

## 变更日志

| 日期 | 内容 |
|---|---|
| 2026-04-21 | 初建骨架 + README |
