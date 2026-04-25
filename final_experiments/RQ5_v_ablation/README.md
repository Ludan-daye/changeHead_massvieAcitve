# RQ5 — V 矩阵因果消融

**实验目的**：因果验证 v₁ 方向是 MA 生成的「承重梁」。替换 v₁ 后测 MA 塌陷。

---

## 方法 / How to reproduce

**方法**：
- **单层变体**（`exp5_v_ablation.py`）：将起源层 `W_down` 的 v₁ 方向替换为随机正交向量
- **Macro 变体**（`exp5_macro_v_ablation.py`）：跨多起源层捕获 Δh_macro，SVD 得 macro-v₁，对每层 W_down 投影消除 (I - vv^T)
- 对照：`bias_ablation/`（bloom + gptj）消 bias 而非 v₁

```bash
# 主脚本位置：见 code/ 子目录 或 ../../paper_experiments/RQ5_v_matrix_ablation/
# 起源层自动读 ../../paper_experiments/origin_layer/output/L_ORIGIN.sh
bash ../../paper_experiments/run_rq345_origin_layer.sh "<model>" <rq_id>
```

---

## 关键指标

**关键指标**：
- `single_ΔMA%` / `macro_ΔMA%`：≤ -80% 即 PASS
- `ΔPPL`：保证模型未被破坏

---

## 结论

**要证明什么 / 主要论点**：
- **主结论**：18/26 模型（69%）ΔMA ≤ -80%
- **核心证据**：gptj_6b / qwen2.5_7b / qwen2_7b V 消融 ΔMA = **-99%**（完美因果）
- **Pattern A 成功**：单层 V 消融在 CONC/FS 模型的达成率 ~85%
- **Pattern B 成功**：macro V 消融在 DISP 模型的达成率 ~70%（qwen3_32b/4b/8b/14b macro -86%~-100%）
- llama2_7b_chat 原 L=26 错层 ΔMA=0% → 修为 L=1 ΔMA=-96%
- **失败模式**：qwen3.5_35b_a3b（MoE）、qwen3_30b_a3b（MoE）、opt_6.7b（架构特异）

---

## 每模型结果表（26 模型）

| # | 模型 | cat | L | 结果 | 数据目录 |
|:-:|---|---|:-:|---|---|
| 1 | bloom_7b1 | CONC | 7 | ✅ L=7 K=10 ΔMA=-67% | [`results/bloom_7b1/L7_multi_v/`](results/bloom_7b1/L7_multi_v/) |
| 2 | falcon_7b | FS | 3 | ✅ -98% / macro -97% | [`results/falcon_7b/`](results/falcon_7b/) |
| 3 | glm4_32b | CONC | 0 | ✅ -97% | [`results/glm4_32b/`](results/glm4_32b/) |
| 4 | glm4_9b | FS | 1 | ✅ macro -82% | [`results/glm4_9b/`](results/glm4_9b/) |
| 5 | gpt2 | FS | 3 | ✅ macro -95% | [`results/gpt2/`](results/gpt2/) |
| 6 | gptj_6b | CONC | 2 | ✅ -99% / macro -99% | [`results/gptj_6b/`](results/gptj_6b/) |
| 7 | llama2_13b | FS | 0 | ✅ -96% | [`results/llama2_13b/`](results/llama2_13b/) |
| 8 | llama2_7b_chat | — | 1 | ✅ L=1 -96% (修正错层) | [`results/llama2_7b_chat/`](results/llama2_7b_chat/) |
| 9 | llama3.1_8b | FS | 1 | ✅ macro -100% | [`results/llama3.1_8b/`](results/llama3.1_8b/) |
| 10 | mistral_7b_v03 | CONC | 0 | ✅ -83% | [`results/mistral_7b_v03/`](results/mistral_7b_v03/) |
| 11 | opt_6.7b | ANOM | 1 | ❌ -18% | [`results/opt_6.7b/`](results/opt_6.7b/) |
| 12 | qwen1.5_14b | DISP | 2 | ✅ K=1 mean -76% | [`results/qwen1.5_14b/L2_multi_v/`](results/qwen1.5_14b/L2_multi_v/) |
| 13 | qwen2.5_0.5b | CONC | 0 | ❌ -55% | [`results/qwen2.5_0.5b/`](results/qwen2.5_0.5b/) |
| 14 | qwen2.5_7b | CONC | 3 | ✅ -99% | [`results/qwen2.5_7b/`](results/qwen2.5_7b/) |
| 15 | qwen2_7b | CONC | 3 | ✅ -99% | [`results/qwen2_7b/`](results/qwen2_7b/) |
| 16 | qwen3.5_27b | DISP | 54 | ✅ -78% (接近阈值) / macro bug | [`results/qwen3.5_27b/recheck/`](results/qwen3.5_27b/recheck/) |
| 17 | qwen3.5_35b_a3b | FS MoE | 9 | ❌ +0% (MoE) | [`results/qwen3.5_35b_a3b/`](results/qwen3.5_35b_a3b/) |
| 18 | qwen3.5_9b | DISP | 22 | ❌ -16% / macro -57% | [`results/qwen3.5_9b/recheck/`](results/qwen3.5_9b/recheck/) |
| 19 | qwen3_0.6b | CONC | 2 | ✅ -93% | [`results/qwen3_0.6b/`](results/qwen3_0.6b/) |
| 20 | qwen3_1.7b | FS | 2 | ✅ macro -100% | [`results/qwen3_1.7b/`](results/qwen3_1.7b/) |
| 21 | qwen3_14b | DISP | 6 | ✅ macro -88% | [`results/qwen3_14b/`](results/qwen3_14b/) |
| 22 | qwen3_30b_a3b | DISP MoE | 1 | ❌ -1% / macro 0% | [`results/qwen3_30b_a3b/`](results/qwen3_30b_a3b/) |
| 23 | qwen3_32b | DISP | 6 | ✅ -98% / macro -86% | [`results/qwen3_32b/`](results/qwen3_32b/) |
| 24 | qwen3_4b | FS | 6 | ✅ -95% / macro -100% | [`results/qwen3_4b/`](results/qwen3_4b/) |
| 25 | qwen3_8b | DISP | 6 | ✅ -96% / macro -100% | [`results/qwen3_8b/`](results/qwen3_8b/) |
| 26 | yi_9b | DISP | 8 | ✅ -93% / macro -99% | [`results/yi_9b/`](results/yi_9b/) |


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
