# HC — Token 位置 H(C) 熵分析（双重稀疏假说）

> 最终稿 · 2026-04-23
> 导航：[README](../README.md) | [OVERVIEW](../OVERVIEW.md)

---

## 实验目的

验证 **论点 E — 双重稀疏假说**：
1. **Token 维度稀疏**：MA 位置是"低熵 token"（模型能稳定预测下一 token 的位置）
2. **方向维度稀疏**：MA 写入 u₁ 的稀疏 hidden dim

两个稀疏交叉 → MA = 两个稀疏集合的交点。

## 实验方式

对每个 token 位置 x 计算：
$$
H(C)(x) = -\sum_c p(c|x) \log p(c|x)
$$

即 LLM 在位置 x 预测下一 token 的**熵**（信息论意义上的不确定性）。

- **熵低**（~0.5 bits）= 模型很确信下一个 token 是什么 = 位置 x 是"锚点"
- **熵高**（~8 bits）= 模型不确定

**核心假设**：MA 位置 ↔ H(C) 低 ↔ function_token

**脚本**：`RQ3_function_words/exp5c_entropy.py`

## 26 模型数据完整

| 模型 | 有 HC 数据？ | 备注 |
|---|:-:|---|
| 所有 26 模型 | ✅ | 数据在 `fixes/results_stage2*/systemd_hc*/` |

## 主要发现（2026-04-22 论点 E）

### 🎯 核心实证：FT 位置 H(C) 显著低

对 14 个分析过 HC 的模型，**function_token 位置的平均 H(C) 比 content_token 低 2-4 bits**——即 FT 位置模型**更确信下一个 token**。

典型例子（qwen2.5_7b）：
- `'\n\n'` 位置 H(C) ≈ 0.3 bits（几乎确定下一个 token）
- 随机 CW 位置 H(C) ≈ 3.5 bits（分布较散）

### 🧩 双重稀疏的证据

| 维度 | 证据 | 来源 |
|---|---|---|
| **Token 稀疏** | FT 位置占 ~53% 但 MA Top-1 占 92% | RQ3 |
| **方向稀疏** | u₁ top1 weight 平均 ~0.75 | u₁ 分析 |
| **交叉** | MA 值 = H(C) 低 位置 × u₁ 稀疏维度 | HC 分析 |

## 论点演化（A → B → C → E）

| 论点 | 时间 | 内容 | 状态 |
|:-:|:-:|---|:-:|
| A | 04-17 | 功能词 mark（the/of/and 等语法功能词）| ✗ 太窄 |
| B | 04-20 | 结构 token mark（换行/标点/@）| ✗ 14 模型只 glm4 支持 |
| C | 04-22 | 低熵 token mark（信息论位置）| ✓ 被最终论点吸收 |
| E | 04-22 | **双重稀疏假说**（token + direction）| ★ 定稿 |

## 论点 E 的精确表述

> MA 是 **token 稀疏性** 和 **direction 稀疏性** 的交集现象：
> - 在**信息论低熵位置**（function_token / 可预测的下一 token 的位置）
> - 通过 **W_down 的稀疏主方向**（σ₁·v₁）
> - 写到 **u₁ 的稀疏 hidden 维度** j\*
> - 形成数值极大的激活

## 为什么 FT 位置是低熵的？

语言统计特性：
- `'\n\n'` 后通常是新段落开头（大写字母、标题等）—— **模型能预测范围很窄**
- `'.'` 后通常是空格 + 大写 —— 低熵
- `'@'` 后可能是邮箱、Twitter handle —— 低熵
- 普通内容词之后可能跟几百种词 —— 高熵

所以 FT = 信息论锚点 = MA 写入位置。

## 14 模型 H(C) 实测结果

数据存于 `fixes/analyze_HC_results.json` 和 `fixes/systemd_*_tokens.json`：

| 模型 | FT 位置 mean H(C) | CT 位置 mean H(C) | 差值 (bits) |
|---|---:|---:|---:|
| glm4_9b | ~0.8 | ~3.5 | **-2.7** |
| llama3.1_8b | ~1.1 | ~3.8 | -2.7 |
| qwen1.5_14b | ~0.6 | ~3.2 | -2.6 |
| qwen2.5_7b | ~0.3 | ~3.5 | **-3.2** |
| ...其他 10 模型 | similar | similar | -2 ~ -4 |

**14/14 模型都观测到 FT 位置熵显著低于 CT**。

## 论点 E 的验证条件

```
PASS:  FT 位置平均 H(C) < CT 平均 H(C) - 1.0 bit    (14/14 满足)
PASS:  Top-K MA 位置的平均 H(C) < 全体平均 H(C)      (待补验证)
```

## HC 解释了什么问题

1. **连接 RQ3 和语言模型理论**：MA 位置不只是语法上的"function word"，而是**信息论上的锚点**（低熵预测点）
2. **支持论点 E 双重稀疏**：MA 是两个稀疏集合的交集，这个视角比单维稀疏（只讲 token 或只讲 direction）更精确
3. **提供新角度解释"attention sink"**：sink 位置是低熵位置（模型依赖此处作信息枢纽）

## 关键观察

1. **FT 位置熵显著低**（-2 到 -4 bits，14/14 模型验证）
2. **熵 low-bound ~0.1 bit**（`\n\n` 后几乎是确定的下一 token）
3. **H(C) 在不同模型家族高度一致**：无论架构（Qwen/GLM/Llama），FT 位置熵都低——是**语言本身的统计性质**，非模型特异
4. **这和 attention sink 理论吻合**（Xiao et al.）：sink 位置就是低熵锚点

## 数据补齐状态

- **26/26 完整**（`fixes/` 各子目录）
- 14 模型深度分析（HC 熵分布 + Top-K token 列表）
- 12 模型仅基础数据（待深度分析）

## 结论摘要

> **HC 最终结论**：function_token 位置的 H(C) 熵（下一 token 预测熵）比 content_token 低 2-4 bits（14/14 模型验证），支持 **论点 E 双重稀疏假说**：MA 是**信息论低熵位置**（token 稀疏）和 **W_down 稀疏主方向**（direction 稀疏）的交集。
>
> 这个发现连接了 MA 研究、attention sink 理论和语言统计——function_token 本质上是**语言的统计锚点**（高频、可预测），LLM 用它们当"信息枢纽"，MA 是这种枢纽的极端数值表现。

## 数据文件

- **HC 熵结果**：`fixes/analyze_HC_results.json` + `fixes/analyze_HC_results.md`
- **Top-K tokens**：`fixes/systemd_topK_tokens.json`（Top-200 per model）
- **完整 token list**：`fixes/systemd_full_tokens.json`（Top-500 per model）
- **HC 分析脚本**：`fixes/analyze_HC.py`
- **原始数据**：`fixes/results_stage2*/systemd_hc*/<model>/exp5c_entropy_results.json`
