# RQ1 — Attention 消融

**实验目的**：证伪假设 H₀「attention 是 MA 起源」。测量 attention 全消融后 MA 是否归零——若未归零则 H₀ 被否定。

---

## 方法 / How to reproduce

**方法**：
1. Baseline：原始模型跑 WikiText-2，收集每 token 每层 top-1 激活 (`baseline/results.json`)
2. Intervention：在每个 attention 模块注册 forward-hook 使输出为零，跑同样数据 (`all_heads_disabled/results.json`)
3. 对比：`comparison/EXPERIMENT_1_SUMMARY.txt` + 4 张诊断图

```bash
# 主脚本位置：见 code/ 子目录 或 ../../paper_experiments/RQ1_attention_contribution/
# 起源层自动读 ../../paper_experiments/origin_layer/output/L_ORIGIN.sh
bash ../../paper_experiments/run_rq345_origin_layer.sh "<model>" <rq_id>
```

---

## 关键指标

**关键指标**：
- `residual% = disabled_top1 / baseline_top1 × 100`（主证伪指标）
- `ΔMA% = (disabled - baseline) / baseline × 100`（方向指标）

---

## 结论

**要证明什么 / 本轮证明了什么**：
- **主结论**（H₀ 证伪）：**atten_h 不是 MA 的起源**——26/26 有数据模型中，关 attention 后 MA 都有残留。最小残留 residual% = **1.69%**（`gptj_6b`）。**没有任何一个模型归零** → MA 必须有另一来源（→ RQ2 指向 MLP）。
- **副结论（影响方向）**：
  - **Generative**（ΔMA<0）：17 个——attention 是放大器/广播器
  - **Suppressive**（ΔMA>0）：8 个——attention 是抑制器/稳态器

---

## 每模型结果表（26 模型）

| # | 模型 | cat | L | 结果 | 数据目录 |
|:-:|---|---|:-:|---|---|
| 1 | bloom_7b1 | CONC | 7 | ✅ Gen ΔMA=-98.3% | [`results/bloom_7b1/`](results/bloom_7b1/) |
| 2 | falcon_7b | FS | 3 | ✅ Gen ΔMA=-65% | [`results/falcon_7b/`](results/falcon_7b/) |
| 3 | glm4_32b | CONC | 0 | ✅ Sup (fp32 分支) | [`results/glm4_32b/`](results/glm4_32b/) |
| 4 | glm4_9b | FS | 1 | ✅ Gen | [`results/glm4_9b/`](results/glm4_9b/) |
| 5 | gpt2 | FS | 3 | ✅ Gen | [`results/gpt2/`](results/gpt2/) |
| 6 | gptj_6b | CONC | 2 | ✅ Gen, residual=1.69% (最小) | [`results/gptj_6b/`](results/gptj_6b/) |
| 7 | llama2_13b | FS | 0 | ✅ | [`results/llama2_13b/`](results/llama2_13b/) |
| 8 | llama2_7b_chat | — | 1 | ✅ | [`results/llama2_7b_chat/`](results/llama2_7b_chat/) |
| 9 | llama3.1_8b | FS | 1 | ✅ Gen | [`results/llama3.1_8b/`](results/llama3.1_8b/) |
| 10 | mistral_7b_v03 | CONC | 0 | ✅ | [`results/mistral_7b_v03/`](results/mistral_7b_v03/) |
| 11 | opt_6.7b | ANOM | 1 | ✅ | [`results/opt_6.7b/`](results/opt_6.7b/) |
| 12 | qwen1.5_14b | DISP | 2 | ✅ | [`results/qwen1.5_14b/`](results/qwen1.5_14b/) |
| 13 | qwen2.5_0.5b | CONC | 0 | ✅ Gen | [`results/qwen2.5_0.5b/`](results/qwen2.5_0.5b/) |
| 14 | qwen2.5_7b | CONC | 3 | ✅ Sup | [`results/qwen2.5_7b/`](results/qwen2.5_7b/) |
| 15 | qwen2_7b | CONC | 3 | ✅ (ΔMA=+∞ → nsamples=60 修正) | [`results/qwen2_7b/`](results/qwen2_7b/) |
| 16 | qwen3.5_27b | DISP | 54 | ✅ | [`results/qwen3.5_27b/`](results/qwen3.5_27b/) |
| 17 | qwen3.5_35b_a3b | FS MoE | 9 | ✅ (hybrid_attn 修复后) | [`results/qwen3.5_35b_a3b/`](results/qwen3.5_35b_a3b/) |
| 18 | qwen3.5_9b | DISP | 22 | ✅ | [`results/qwen3.5_9b/`](results/qwen3.5_9b/) |
| 19 | qwen3_0.6b | CONC | 2 | ✅ | [`results/qwen3_0.6b/`](results/qwen3_0.6b/) |
| 20 | qwen3_1.7b | FS | 2 | ✅ | [`results/qwen3_1.7b/`](results/qwen3_1.7b/) |
| 21 | qwen3_14b | DISP | 6 | ✅ | [`results/qwen3_14b/`](results/qwen3_14b/) |
| 22 | qwen3_30b_a3b | DISP MoE | 1 | ✅ | [`results/qwen3_30b_a3b/`](results/qwen3_30b_a3b/) |
| 23 | qwen3_32b | DISP | 6 | ✅ | [`results/qwen3_32b/`](results/qwen3_32b/) |
| 24 | qwen3_4b | FS | 6 | ✅ | [`results/qwen3_4b/`](results/qwen3_4b/) |
| 25 | qwen3_8b | DISP | 6 | ✅ | [`results/qwen3_8b/`](results/qwen3_8b/) |
| 26 | yi_9b | DISP | 8 | ✅ | [`results/yi_9b/`](results/yi_9b/) |


---

## 每模型详细分析

每个模型子目录下有独立的 `analysis.md`：

```
results/<model>/
├── analysis.md           ← 此模型详细分析
├── <data files>
└── ...
```

---

## 参考

- 根总览：[`../README.md`](../README.md)
- 整体 PASS/FAIL 矩阵：[`../STATUS.md`](../STATUS.md)
- 综合结论：[`../../CONCLUSIONS.md`](../../CONCLUSIONS.md)
- 原始代码：[`../../paper_experiments/`](../../paper_experiments/)
- 项目日志：[`../../../CLAUDE.md`](../../../CLAUDE.md)
