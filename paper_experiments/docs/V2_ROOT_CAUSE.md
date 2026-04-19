# V2 数据根因诊断：层错位是主因

> 基于 `results/ALL_EXPERIMENTS_SUMMARY_v2.json`（2026-04-19）
> 本文档回答一个问题：**当前实验不符合预期的根本原因是什么？**

## 一句话结论

> **~2/3 的问题是同一件事：RQ3 / RQ4 / RQ5 单层实验的 `--layer_id` 仍然用了 peak 层（或 v1 里陈旧的 critical_layer），没有更新到 exp2c 的 `l_origin_from_step1`。**

修这 1 件事杀 30+ 处指标异常；剩余 6-7 件独立问题单独处理。

## 一、证据：错层导致的指标异常

### 1.1 RQ3 Cohen's d 负或近零（13/18 个完成模型）

| 模型 | 模式 | cohens_d | 预期 | 偏差 |
|---|:-:|:-:|:-:|:-:|
| qwen2_7b | A | **−1.46** | > 0.5 | −2.0 |
| qwen2.5_7b | A | **−1.34** | > 0.5 | −1.8 |
| qwen3.5_27b | B | −1.01 | 0 | −1.0 |
| qwen3.5_9b | 中 | **−0.89** | > 0.3 | −1.2 |
| qwen3_8b | 中 | −0.55 | > 0.3 | −0.85 |
| qwen2.5_0.5b | A | **−0.50** | > 0.5 | −1.0 |
| yi_9b | A | **−0.31** | > 0.5 | −0.8 |
| qwen3_0.6b | A | **−0.28** | > 0.5 | −0.8 |
| qwen3_1.7b | A | −0.15 | > 0.5 | −0.65 |
| qwen3_30b_a3b | MoE | −0.05 | — | — |
| llama3.1_8b | A | +0.37 | > 0.5 | −0.15 |
| glm4_9b | 中 | +0.24 | > 0.3 | 弱 |

**物理解释（见 README §II 步骤 3）**：
- 起源层：MLP 刚把 mark 写到功能词，d ≫ 0
- attention 广播后：内容词被"染色"，差距稀释
- peak 层：部分模型内容词 v₁ 投影反超功能词 → d 变负

**9 个负值是"attention 广播"的直接实证**——不是 bug，但意味着数据不能用于支撑"功能词触发 MA"的论点。

### 1.2 RQ4 σ₁/σ₂ 全部低于预期

模式 A 起源层 W_down 应 σ₁/σ₂ ≥ 3（GPT-J 论文实测 5.74）。v2 实际：

```
模式 A (10 个)：1.13 – 1.58  ← 全部远低于 3
中间   (7 个) ：1.39 – 2.58  ← 应在 2-3 之间
B     (3 个) ：1.45 – 1.77  ← 弱是对的
```

**没有一个模式 A 模型达标**——说明 RQ4 根本没在起源层跑。peak 层 MLP 的 W_down 不需要谱集中。

### 1.3 RQ5 单层 u_attribution 对模式 A 偏低

| 模型 | u_attribution% | 预期 |
|---|:-:|:-:|
| qwen3_1.7b | 57% | > 80% |
| qwen3_0.6b | 64% | > 80% |
| qwen2.5_0.5b | 67% | > 80% |

## 二、对比证据：用正确起源层做的 RQ5b 全部成功

当 exp5b 用 `exp2c.final_disabled_set` 作为 origin_layers 时，效果立刻显现：

| 模型 | origin_layers（来自 exp2c）| ΔMA |
|---|:-:|:-:|
| llama3.1_8b | [0,1] | **−99.8%** |
| qwen3_1.7b | [0,1,2] | −99.8% |
| qwen3_14b | [0,2,3,4,5,6] | −99.5% |
| qwen3_4b | [6,15] | −99.7% |
| qwen3_8b | [0,2,3,5,6,7] | −99.6% |
| yi_9b | [8,18,23,24,35] | −99.2% |
| qwen3_32b | [4,5,6,19,28,29] | **−86.3%**（模式 B 实锤）|
| **glm4_9b** | [0,1] | **−81.8%**（中间→真 B 确认）|

**11 个 RQ5b 中 8 个强通过**，失败的 3 个（qwen1.5_14b、qwen3.5_9b、qwen3.5_27b）全是 DISPERSED 模式，是 origin_layers 选择策略的问题，不是方法错。

**结论**：只要用对起源层，方法本身完全正确。

## 三、起源层定位方案对比

| 方案 | 数据源 | 问题 |
|---|---|---|
| A. **peak_layer** | `exp1.peak_layer` | ❌ 错层：MA 观测最大层 ≠ 写入层。v1 脚本就是这样错的 |
| B. **exp2.critical_layer** | v1 JSON 单层最强消融层 | ⚠️ 对 CONCENTRATED/FEW-SOURCE 可用，对 DISPERSED **严重失真**（例：glm4_9b 给出 L17 但实际 L1）|
| C. **exp2c.l_origin_from_step1** | 贪心累积第一步选的层 | ✅ **推荐**：单层实验（RQ3/4/5 single）的正确起源 |
| D. **exp2c.final_disabled_set** | 贪心完整消融集合 | ✅ **推荐**：macro 实验（RQ5b/RQ6 macro）的正确原点集合 |

## 四、错层导致的新旧起源层差异

部分模型 v1 critical_layer（方案 B）和 v2 exp2c（方案 C）差异**巨大**：

| 模型 | v1 critical_layer | v2 exp2c step1 | 差距 | 影响 |
|---|:-:|:-:|:-:|---|
| **glm4_9b** | 17 | **1** | 16 层 | 几乎是两个不同实验 |
| qwen3.5_27b | 54 | 50 | 4 | 可能漏掉真起源 |
| qwen3.5_9b | 26 | 22 | 4 | 可能漏掉 |
| glm4_32b | 0 | 0 | 0 | 同 |
| qwen3_32b | 43 | 6 | **37** | **v1 彻底错**，真起源在早期 |
| llama3.1_8b | 1 | 1 | 0 | 同 |

6 个模型中 **3 个的差距 ≥ 4 层，2 个差距超过 15 层**——说明 v1 的 critical_layer 在 DISPERSED 模型上不可靠。

## 五、修复行动

### 立即执行（~4h 成本，解决 ~30 处异常）

1. **用 `exp2c.l_origin_from_step1` 更新 `run_rq345_origin_layer.sh` 的 `L_ORIGIN` 表**
2. **重跑 RQ3 / RQ4 / RQ5（single）全 25 个**
3. **预期**：
   - 模式 A 的 Cohen's d 从负翻正到 > 0.5
   - 模式 A 的 σ₁/σ₂ 从 1.x 翻到 ≥ 3
   - 模式 A 的 RQ5 单层 ΔMA ≤ −85%

### 和层错无关的 6 件独立事（需另外处理）

| # | 问题 | 修法 |
|:-:|---|---|
| 1 | exp2a 全 null | 补跑 RQ2a disable-all-MLP |
| 2 | 6 个原始模型（bloom/falcon/gpt2/gptj/mistral/opt）RQ3-6 全空 | 起源层一起跑（前述 1 的一部分）|
| 3 | llama2_13b 缺 RQ2b | 在老仓库跑 exp2b，确定 critical_layer |
| 4 | qwen2_7b RQ1 = +∞ | nsamples 30→60 |
| 5 | glm4_32b fp32 全流程 | loader 已修好，重跑 |
| 6 | MoE 单层 ΔMA=0% | Tier C 专项（本轮暂缓）|

### DISPERSED 模式的 origin_layers 策略

3 个 RQ5b 弱结果都是 DISPERSED 模型。对 DISPERSED：
- **不要**把 `final_disabled_set` 原样当 origin_layers 用——它是消融顺序，不是"完整起源"
- **应该**尝试：`[step1, step2, ..., steps_to_kill]` 的**前半部分**（优先级最高几层）
- 或者用 `final_disabled_set` 的**前 40% 层**

见 `determine_origin_layer.py` 的 `derive_origin_layers()` 函数。

## 六、代码工具

- `determine_origin_layer.py`（本轮新增）：从 v2 JSON 自动推导每个模型的起源层，输出可直接粘贴到 bash 的 `L_ORIGIN` 表
- 用法：`python determine_origin_layer.py > L_ORIGIN_v2.sh`

---

**一句话总结**：现在的数据看起来"处处都错"，其实是**同一个层错导致的连锁反应**。改 1 处层号表，约 30+ 处指标异常会自动变成预期值。
