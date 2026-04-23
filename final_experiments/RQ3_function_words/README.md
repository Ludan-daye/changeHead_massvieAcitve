# RQ3 — Function Token 定位

**实验目的**：验证 MA 极值位置是否落在 function_token（广义：功能词 + 标点 + 换行 + 特殊符号）。

---

## 方法 / How to reproduce

**方法**：
1. 在起源层捕获 h₂，按 |h₂ · v₁| 排序
2. 计算 Top-K 中 FT 比例、Top-1 MA token 是否 FT
3. Cohen's d：FT vs 内容词的投影差异

```bash
# 主脚本位置：见 code/ 子目录 或 ../../paper_experiments/RQ3_function_words/
# 起源层自动读 ../../paper_experiments/origin_layer/output/L_ORIGIN.sh
bash ../../paper_experiments/run_rq345_origin_layer.sh "<model>" <rq_id>
```

---

## 关键指标

**关键指标**：
- `Top-1 MA token` 是否 FT
- `Cohen's d`（FT vs 内容词投影差异，负值为错层证据）

---

## 结论

**要证明什么 / 本轮证明了什么**：
- **主结论**：24/26 模型的 Top-1 MA token 是 function_token（广义定义含 \n\n、标点、符号）
- **Top-1 实例**：gptj/qwen2_7b/qwen3_0.6b = `\n\n`（换行）；yi_9b = `''`（空白）；bloom = `' k'`
- **2 个 FAIL**：llama2_7b_chat（R²=0 异常）、qwen3.5_35b_a3b（MoE）
- **重要修正**：2026-04-23 is_FT() bug 修复——空白 token 正确识别为 FT 后，mistral、yi_9b 的 Top-1 都是 `''`

---

## 每模型结果表（26 模型）

| # | 模型 | cat | L | 结果 | 数据目录 |
|:-:|---|---|:-:|---|---|
| 1 | bloom_7b1 | CONC | 7 | ✅ Top-1 MA = FT | [`results/bloom_7b1/`](results/bloom_7b1/) |
| 2 | falcon_7b | FS | 3 | ✅ | [`results/falcon_7b/`](results/falcon_7b/) |
| 3 | glm4_32b | CONC | 0 | ✅ | [`results/glm4_32b/`](results/glm4_32b/) |
| 4 | glm4_9b | FS | 1 | ✅ | [`results/glm4_9b/`](results/glm4_9b/) |
| 5 | gpt2 | FS | 3 | 🟡 R²=0.55 (弱) | [`results/gpt2/`](results/gpt2/) |
| 6 | gptj_6b | CONC | 2 | ✅ Top-1='\n\n' | [`results/gptj_6b/`](results/gptj_6b/) |
| 7 | llama2_13b | FS | 0 | ✅ | [`results/llama2_13b/`](results/llama2_13b/) |
| 8 | llama2_7b_chat | — | 1 | ❌ R²=0 异常 | [`results/llama2_7b_chat/`](results/llama2_7b_chat/) |
| 9 | llama3.1_8b | FS | 1 | ✅ | [`results/llama3.1_8b/`](results/llama3.1_8b/) |
| 10 | mistral_7b_v03 | CONC | 0 | ✅ | [`results/mistral_7b_v03/`](results/mistral_7b_v03/) |
| 11 | opt_6.7b | ANOM | 1 | ✅ | [`results/opt_6.7b/`](results/opt_6.7b/) |
| 12 | qwen1.5_14b | DISP | 2 | ✅ | [`results/qwen1.5_14b/`](results/qwen1.5_14b/) |
| 13 | qwen2.5_0.5b | CONC | 0 | ✅ | [`results/qwen2.5_0.5b/`](results/qwen2.5_0.5b/) |
| 14 | qwen2.5_7b | CONC | 3 | ✅ | [`results/qwen2.5_7b/`](results/qwen2.5_7b/) |
| 15 | qwen2_7b | CONC | 3 | ✅ Top-1='\n\n' | [`results/qwen2_7b/`](results/qwen2_7b/) |
| 16 | qwen3.5_27b | DISP | 54 | ✅ | [`results/qwen3.5_27b/`](results/qwen3.5_27b/) |
| 17 | qwen3.5_35b_a3b | FS MoE | 9 | ❌ | [`results/qwen3.5_35b_a3b/`](results/qwen3.5_35b_a3b/) |
| 18 | qwen3.5_9b | DISP | 22 | ✅ | [`results/qwen3.5_9b/`](results/qwen3.5_9b/) |
| 19 | qwen3_0.6b | CONC | 2 | ✅ Top-1='\n\n' | [`results/qwen3_0.6b/`](results/qwen3_0.6b/) |
| 20 | qwen3_1.7b | FS | 2 | ✅ | [`results/qwen3_1.7b/`](results/qwen3_1.7b/) |
| 21 | qwen3_14b | DISP | 6 | ✅ | [`results/qwen3_14b/`](results/qwen3_14b/) |
| 22 | qwen3_30b_a3b | DISP MoE | 1 | ✅ | [`results/qwen3_30b_a3b/`](results/qwen3_30b_a3b/) |
| 23 | qwen3_32b | DISP | 6 | ✅ | [`results/qwen3_32b/`](results/qwen3_32b/) |
| 24 | qwen3_4b | FS | 6 | ✅ | [`results/qwen3_4b/`](results/qwen3_4b/) |
| 25 | qwen3_8b | DISP | 6 | ✅ | [`results/qwen3_8b/`](results/qwen3_8b/) |
| 26 | yi_9b | DISP | 8 | ✅ Top-1='' | [`results/yi_9b/`](results/yi_9b/) |


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
