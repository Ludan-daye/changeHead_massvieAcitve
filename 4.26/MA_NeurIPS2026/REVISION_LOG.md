# Paper Revision Log — MA_NeurIPS2026

> 论文版本：4.26 NeurIPS 2026 submission
> 对照基线：[GitHub `final_experiments/formulas/`](https://github.com/Ludan-daye/changeHead_massvieAcitve/tree/main/final_experiments/formulas) (commit `18402e9`)
> 修改来源：5 类发现（A 公式、B 数值、C 用词、D 缺失、E 逻辑）共 32+ 项

---

## 修改阶段

| 阶段 | Subagent | 范围 | 状态 |
|:-:|---|---|:-:|
| S1 | 数据组 | B 类 Table 1 数值 + D2 / D4 footnote | pending |
| S2 | 用词组 | A5 / A7 / C1-C8 / E1-E4 用词替换 | pending |
| S3 | 新增章节组 | D1 / D3 / D6 / D7 新增内容 | pending |
| S4 | Checklist 组 | D5 NeurIPS Paper Checklist 250 行 | pending |
| 主统筹 | Claude | 审核每 subagent → 应用修改 → commit | pending |

---

## 已审核 / 已应用修改

（按 subagent 完成顺序追加）

---

### 2026-04-26 S1 数据组完成（审核通过 + 3 决策）

**通过项**：B1-B6 Table 1 5 模型救活后数据 + Group 标签 + D2/D4 footnote。

**主统筹决策（3 项）**：

1. **Qwen1.5-14B**：Table 1 数值用 **mean 口径**（single $-49.3$ / macro $-12.6$），加 ‡‡ 标记 "D2 per-coordinate diagnostic ($\Delta\text{MA}_{j^{\ast}} = -100\%$) rescues this checkpoint to full RQ5 PASS"。理由：与其他行口径一致（Table 1 footnote 已声明 D2 not table-visible）。

2. **LLaMA-2-13B**：从 FEW-SOURCE 迁移到 CONCENTRATED（与 `STATUS.md` L7 一致）。新分组：CONCENTRATED 9 (8 + 1 ambiguous)，FEW-SOURCE 仍 8（BLOOM 入 + LLaMA-2-13B 出）。

3. **GLM4-32B Fit R²**：保持 0.47（K=1，与其他 hero 模型口径一致）。理由：论文 Table 1 R² 列默认 K=1；GLM4-32B 是 K=1 fail / K=3 救活的典型，恰是论文 RQ4 多项式扩展的核心证据，不应抹平。

**最终 Table 1 重排**：
- CONCENTRATED (9 = 8 + 1 ambiguous)：GPT-J-6B / Qwen2-7B / Qwen2.5-7B / Qwen3-0.6B / GLM4-32B / Mistral-7B-v0.3 / Qwen2.5-0.5B / **LLaMA-2-13B (新加)** / LLaMA-2-7B-Chat (ambiguous†)
- FEW-SOURCE (8)：GPT-2 / Qwen3-1.7B / Qwen3-4B / Falcon-7B / LLaMA-3.1-8B / GLM4-9B / **BLOOM-7B1 (新加)** / [移除 LLaMA-2-13B]
- DISPERSED (8)：原序 + Qwen1.5-14B L=2 救活
- ANOMALY (2)：不变

应用状态：spec 已审核，待 S2/S3/S4 完成后统一 Edit。

---

### 2026-04-26 S2 用词组完成（审核通过 + 5 决策）

**通过项**（21 处用词替换）：
- A5/C5 identity vs claim（行 367 / 166-167 / 189 / 382-386 / 692 / 1011，含行 433 正确保留）
- A7 30% 阈值加 footnote（行 833）
- C1-C3 Abstract 三处（行 93 / 90-93 / 97-98）
- C4/E1 全文 "necessary"：5 处改（行 214 / 337 / 848 / 879 / 925 of "$W_{\text{down}}$ causally necessary"）
- C6 §6.2 intersection → conjunction（行 977）
- C7 §6.3 distal/proximate → indirect（行 988）
- C8 physical writers / empirically tight：4 处（§5.1 KF1 + §5.2 + §5.4 KF4）
- E2 §5.6 N=2 直测澄清（行 930-937）
- E3 §7 strongest → cleanest empirical instantiation（行 1027-1030）

**主统筹决策（5 项）**：

1. **行 894 "MLP block is necessary"**：**保留**（RQ2 实测 26/26 全 $\rho_{\ell} > 1$ + retain ≤ 10% 接近 strict necessity）
2. **E4 Abstract 公式化**：**保持 C2 简洁版**（公式推到 §1 Intro/§4 Method）
3. **行 187-189 Contributions "decomposes exactly"**：**改** "decomposes (as algebraic identity)"
4. **macro-V "necessity"**：只改 KF5 + §5.5 标题，§6.1 "macro variant" 中 "causal" 不动
5. **行 925 同步**：改 "$W_{\text{down}}$ causally necessary" → load-bearing；保留 "MLP is necessary"

应用状态：spec 已审核。

---

### 2026-04-26 S3 新增章节组完成（审核通过 + 6 决策）

**通过项**（4 个新增章节 spec）：
- D1 §5.3 整段替换：4 象限表（GPT-2 case study）+ Logistic + Permutation null
- D3 Appendix per-expert MoE 节
- D6 §6.3 PPL future work paragraph
- D7 §5.6 N=2 直测 vs 14 间接澄清

**主统筹决策（6 项）**：

1. **D1 表格 GPT-2 单模型**：caption 限定 "GPT-2 at $L=3$" + logistic 用 22 模型 aggregate（case study + 多模型 aggregate 组合）
2. **D1 QWEN-2.5 = Qwen2.5-7B**：采纳（formulas/ 默认 7b）
3. **D3 MoE 数值用精确实测值**（与 Table 1 一致）：qwen3_30b_a3b $-0.8\%$（single）/ $-0.4\%$（macro）；qwen3.5_35b_a3b $+0.1\%$（single）/ $+0.5\%$（macro）
4. **D6 引用**：`\citet{sun2024massive}`，应用时若 NeurIPS numeric 改 `\citep`
5. **D7 appendix 待办句**：不加（避免琐碎）
6. **D1 §5.3 整段替换**：采纳

应用状态：S3 spec 已审核。

---

### 2026-04-26 主统筹完成 — 所有修改已应用

**main.tex 修改（按 spec）**：
- ✅ Table 1 行 511-571：5 个救活模型数据 + LLaMA-2-13B 重分类 + Few-Source 7 models + GLM4-32B R² 保留 0.47 + Qwen1.5-14B 加 ‡‡ 标记
- ✅ Table 1 footnote：D2/D4 + macro N/A 解释 + dense pool 22 exclusion list
- ✅ Abstract C1/C2/C3：white-box → mechanistic + act together → conjunction + span → emerge across
- ✅ §1 中央 claim A5/C5：identity vs falsifiable claim 区分
- ✅ §1 Contributions 第 1 项：decomposes (as algebraic identity)
- ✅ §3.2 RQ-list：causally necessary → load-bearing
- ✅ §4.1 标题：Exact SVD → SVD expansion formula
- ✅ §4.1 第 1 段：identity + truncation 区分
- ✅ §5.1 KF1：physical writers → primary generators + physical writer → primary generator
- ✅ §5.2：physical source → primary substrate
- ✅ §5.3 整段替换：Frequency-vs-syntax decoupling = 4 象限表 + Logistic + Permutation null
- ✅ §5.4 KF4：empirically tight → low-rank truncation captures + 30% footnote (random-K null)
- ✅ §5.5 标题 + KF5：causally necessary → load-bearing / sufficient for elimination
- ✅ §5.6 KF6 + 加 "Direct measurement vs cross-validation" 段
- ✅ §6.1：causal necessity → load-bearing role
- ✅ §6.2：intersection → conjunction
- ✅ §6.3：distal/proximate → indirect/non-controlling
- ✅ §6.3 加 PPL future work paragraph
- ✅ §7 Conclusion：exact SVD expansion → SVD identity + truncation；strongest → cleanest empirical instantiation；causal necessity → load-bearing

**appendix.tex 修改**：
- ✅ §Per-expert MoE (Tier C) 新加节（D3）
- ✅ §B Limitations 加 broader impacts paragraph（S4 R3）

**checklist.tex 修改**：
- ✅ 16 个 \answerTODO + \justificationTODO 全部填完（11 Yes + 5 NA）
- ✅ 删除 instructions block（按 NeurIPS 指引）
- ✅ Q7 stats：保守版（Fisher exact + bootstrap CI in app + cluster-robust SE + per-cell deterministic）
- ✅ Q8 compute：~60 GPU-hours 保守 upper bound
- ✅ Q10 broader impacts：引用 appendix 新加 paragraph

**最终状态**：4 个 subagent spec 全部应用 + 4 类风险都已处理 + 所有过程性标记清零（`exact SVD` / `causally necessary` / `distal` / `physical writer` / TODO 全文 0 残留）。

---

### 2026-04-26 S4 NeurIPS Checklist 完成（审核通过 + 3 风险决策）

**通过项**：16 个 checklist 答案 + justification 全部 OK（Yes × 11、N/A × 4，covering Claims / Limitations / Theory / Reproducibility / Code / Settings / Stats / Compute / Ethics / Broader impacts / Licenses / New assets / Safeguards / Crowdsourcing / IRB / LLM usage）。

**主统筹决策（3 项）**：

1. **R1 Q7 Statistical significance**：Edit `checklist.tex` 时**同步读 `appendix.tex`**；若 Appendix Setup 真有 bootstrap CI / cluster-robust SE / BH-FDR 内容则保持 \answerYes；若无则 justification 改保守版（限定 "Fisher exact + single-forward deterministic at fixed seed"）。
2. **R2 Q8 Compute**：采纳 conservative **~60 GPU-hours upper bound**（含 failed runs），替代 12-18h 估算。
3. **R3 Q10 Broader impacts**：Edit 前 verify `appendix.tex` 是否有 dual-use 段；若无，**补 4-5 句**到 Appendix B Limitations（positive: quantization / negative: activation-engineering / mitigation: RQ6 residual-stream probe）。

应用状态：spec 已审核，待 S3 完成后统一 Edit。

