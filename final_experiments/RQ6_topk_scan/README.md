# RQ6 — Top-K 恢复（macro-SVD 多层聚合）

**实验目的**：测量「仅保留起源层 Top-K 激活」或 macro-SVD 多层 Δh 累加的 MA 恢复率。

---

## 方法 / How to reproduce

**方法**：
- **6.1 macro-SVD**：跨多层累加 Δh，做 SVD
- **6.2/6.3 top-K**：删除或保留 top-K MA 测 baseline 恢复
- **6.4 progressive**（≡ RQ2c）：贪心累积消融

```bash
# 主脚本位置：见 code/ 子目录 或 ../../paper_experiments/RQ6_single_layer_activation/
# 起源层自动读 ../../paper_experiments/origin_layer/output/L_ORIGIN.sh
bash ../../paper_experiments/run_rq345_origin_layer.sh "<model>" <rq_id>
```

---

## 关键指标

**关键指标**：
- `recovery%`：top-K 或 macro 恢复后 / baseline MA
- CONC 期望 ≥ 30%，多层期望 < 30% 一致性

---

## 结论

**要证明什么 / 主要论点**：
- **主结论**：仅 2/26 模型期望高 recovery 达成（gptj_6b 76%, llama3.1_8b 49%）
- **gptj_6b 独特**：CONCENTRATED + Parallel architecture（attention 和 MLP 并行计算）→ 单层 top-K 足够恢复
- **llama3.1_8b 独特**：FEW-SOURCE 但 L=1 RQ6 也过 → 极早层 + 后续 residual 稳态
- **多数模型 residual stream 依赖**：单层 top-K 不够恢复

---

## 每模型结果表（26 模型）

| # | 模型 | cat | L | 结果 | 数据目录 |
|:-:|---|---|:-:|---|---|
| 1 | bloom_7b1 | CONC | 7 | — | [`results/bloom_7b1/`](results/bloom_7b1/) |
| 2 | falcon_7b | FS | 3 | — | [`results/falcon_7b/`](results/falcon_7b/) |
| 3 | glm4_32b | CONC | 0 | — | [`results/glm4_32b/`](results/glm4_32b/) |
| 4 | glm4_9b | FS | 1 | — | [`results/glm4_9b/`](results/glm4_9b/) |
| 5 | gpt2 | FS | 3 | — | [`results/gpt2/`](results/gpt2/) |
| 6 | gptj_6b | CONC | 2 | ✅ 76% (CONC + Parallel 架构) | [`results/gptj_6b/`](results/gptj_6b/) |
| 7 | llama2_13b | FS | 0 | — | [`results/llama2_13b/`](results/llama2_13b/) |
| 8 | llama2_7b_chat | — | 1 | — | [`results/llama2_7b_chat/`](results/llama2_7b_chat/) |
| 9 | llama3.1_8b | FS | 1 | ✅ 49% (FS 唯一过) | [`results/llama3.1_8b/`](results/llama3.1_8b/) |
| 10 | mistral_7b_v03 | CONC | 0 | — | [`results/mistral_7b_v03/`](results/mistral_7b_v03/) |
| 11 | opt_6.7b | ANOM | 1 | — | [`results/opt_6.7b/`](results/opt_6.7b/) |
| 12 | qwen1.5_14b | DISP | 2 | — | [`results/qwen1.5_14b/`](results/qwen1.5_14b/) |
| 13 | qwen2.5_0.5b | CONC | 0 | — | [`results/qwen2.5_0.5b/`](results/qwen2.5_0.5b/) |
| 14 | qwen2.5_7b | CONC | 3 | — | [`results/qwen2.5_7b/`](results/qwen2.5_7b/) |
| 15 | qwen2_7b | CONC | 3 | — | [`results/qwen2_7b/`](results/qwen2_7b/) |
| 16 | qwen3.5_27b | DISP | 54 | — | [`results/qwen3.5_27b/`](results/qwen3.5_27b/) |
| 17 | qwen3.5_35b_a3b | FS MoE | 9 | — | [`results/qwen3.5_35b_a3b/`](results/qwen3.5_35b_a3b/) |
| 18 | qwen3.5_9b | DISP | 22 | — | [`results/qwen3.5_9b/`](results/qwen3.5_9b/) |
| 19 | qwen3_0.6b | CONC | 2 | — | [`results/qwen3_0.6b/`](results/qwen3_0.6b/) |
| 20 | qwen3_1.7b | FS | 2 | — | [`results/qwen3_1.7b/`](results/qwen3_1.7b/) |
| 21 | qwen3_14b | DISP | 6 | — | [`results/qwen3_14b/`](results/qwen3_14b/) |
| 22 | qwen3_30b_a3b | DISP MoE | 1 | — | [`results/qwen3_30b_a3b/`](results/qwen3_30b_a3b/) |
| 23 | qwen3_32b | DISP | 6 | — | [`results/qwen3_32b/`](results/qwen3_32b/) |
| 24 | qwen3_4b | FS | 6 | — | [`results/qwen3_4b/`](results/qwen3_4b/) |
| 25 | qwen3_8b | DISP | 6 | — | [`results/qwen3_8b/`](results/qwen3_8b/) |
| 26 | yi_9b | DISP | 8 | — | [`results/yi_9b/`](results/yi_9b/) |


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
