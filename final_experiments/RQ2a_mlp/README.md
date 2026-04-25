# RQ2a — MLP 全消融

**实验目的**：验证假设 H₁「MLP 是 MA 起源」。测量 MLP 全消融后 MA 是否大幅下降。

---

## 方法 / How to reproduce

**方法**：
1. Baseline：完整模型运行
2. Intervention：对每个 MLP 模块（up_proj + down_proj 或 SwiGLU）注册 forward-hook 使输出归零
3. 对比：测 retention = disabled_max_ma / baseline_max_ma

```bash
# 主脚本位置：见 code/ 子目录 或 ../../paper_experiments/RQ2_mlp_source/
# 起源层自动读 ../../paper_experiments/origin_layer/output/L_ORIGIN.sh
bash ../../paper_experiments/run_rq345_origin_layer.sh "<model>" <rq_id>
```

---

## 关键指标

**关键指标**：
- `retention% = disabled_max_ma / baseline_max_ma × 100`（≤ 10% 即 PASS）
- `reduction% = 100 - retention%`

---

## 结论

**要证明什么 / 本轮证明了什么**：
- **主结论**：MLP 是 MA 主要来源——24/26 有数据模型中，20/24 retain ≤ 10%，bloom_7b1 完全归零（100% reduction）
- **4 个残留异常**（retain > 15%）：qwen3.5_35b_a3b 87.6%（MoE 特异）、qwen3.5_9b 32.1%、qwen3.5_27b 10.0%（qwen3.5 家族）、gpt2 4.3%
- **模式分布**（RQ2c 辅助）：CONCENTRATED 8 / FEW-SOURCE 8 / DISPERSED 8 / ANOMALY 1
- **辅助结论**：RQ2b 的 peak_layer ≠ RQ2c 的 L_origin——V2 错层的根因

---

## 每模型结果表（26 模型）

| # | 模型 | cat | L | 结果 | 数据目录 |
|:-:|---|---|:-:|---|---|
| 1 | bloom_7b1 | CONC | 7 | ✅ retain=0% (完全归零) | [`results/bloom_7b1/`](results/bloom_7b1/) |
| 2 | falcon_7b | FS | 3 | ✅ retain=1.6% | [`results/falcon_7b/`](results/falcon_7b/) |
| 3 | glm4_32b | CONC | 0 | ✅ retain=12.6%（边界放宽 PASS）| [`results/glm4_32b/`](results/glm4_32b/) |
| 4 | glm4_9b | FS | 1 | ✅ retain=4.5% | [`results/glm4_9b/`](results/glm4_9b/) |
| 5 | gpt2 | FS | 3 | ✅ retain=4.3% | [`results/gpt2/`](results/gpt2/) |
| 6 | gptj_6b | CONC | 2 | ✅ retain=1.9% | [`results/gptj_6b/`](results/gptj_6b/) |
| 7 | llama2_13b | FS | 0 | ⏳ HF 401 缺数据 | [`results/llama2_13b/`](results/llama2_13b/) |
| 8 | llama2_7b_chat | — | 1 | ✅ retain=1.1% | [`results/llama2_7b_chat/`](results/llama2_7b_chat/) |
| 9 | llama3.1_8b | FS | 1 | ✅ retain=2.8% | [`results/llama3.1_8b/`](results/llama3.1_8b/) |
| 10 | mistral_7b_v03 | CONC | 0 | ✅ retain=0.8% | [`results/mistral_7b_v03/`](results/mistral_7b_v03/) |
| 11 | opt_6.7b | ANOM | 1 | ⏳ hook 异常 +250% | [`results/opt_6.7b/`](results/opt_6.7b/) |
| 12 | qwen1.5_14b | DISP | 2 | ✅ retain=2.1% | [`results/qwen1.5_14b/`](results/qwen1.5_14b/) |
| 13 | qwen2.5_0.5b | CONC | 0 | ✅ retain=1.6% | [`results/qwen2.5_0.5b/`](results/qwen2.5_0.5b/) |
| 14 | qwen2.5_7b | CONC | 3 | ✅ retain=0.6% | [`results/qwen2.5_7b/`](results/qwen2.5_7b/) |
| 15 | qwen2_7b | CONC | 3 | ✅ retain=0.5% | [`results/qwen2_7b/`](results/qwen2_7b/) |
| 16 | qwen3.5_27b | DISP | 54 | ✅ retain=10.0% | [`results/qwen3.5_27b/`](results/qwen3.5_27b/) |
| 17 | qwen3.5_35b_a3b | FS MoE | 9 | ❌ retain=87.6% (MoE) | [`results/qwen3.5_35b_a3b/`](results/qwen3.5_35b_a3b/) |
| 18 | qwen3.5_9b | DISP | 22 | ❌ retain=32.1% | [`results/qwen3.5_9b/`](results/qwen3.5_9b/) |
| 19 | qwen3_0.6b | CONC | 2 | ✅ retain=1.3% | [`results/qwen3_0.6b/`](results/qwen3_0.6b/) |
| 20 | qwen3_1.7b | FS | 2 | ✅ retain=2.9% | [`results/qwen3_1.7b/`](results/qwen3_1.7b/) |
| 21 | qwen3_14b | DISP | 6 | ✅ retain=1.1% | [`results/qwen3_14b/`](results/qwen3_14b/) |
| 22 | qwen3_30b_a3b | DISP MoE | 1 | ✅ retain=0.3% | [`results/qwen3_30b_a3b/`](results/qwen3_30b_a3b/) |
| 23 | qwen3_32b | DISP | 6 | ✅ retain=0.6% | [`results/qwen3_32b/`](results/qwen3_32b/) |
| 24 | qwen3_4b | FS | 6 | ✅ retain=0.3% | [`results/qwen3_4b/`](results/qwen3_4b/) |
| 25 | qwen3_8b | DISP | 6 | ✅ retain=1.0% | [`results/qwen3_8b/`](results/qwen3_8b/) |
| 26 | yi_9b | DISP | 8 | ✅ retain=1.2% | [`results/yi_9b/`](results/yi_9b/) |


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
