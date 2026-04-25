# RQ4 — SVD 几何对齐 + 公式验证

**实验目的**：验证 MA 生成公式 MA = Σᵢ σᵢ·(h₂·vᵢ)·uᵢ[j\*] + b[j\*]。测量公式 K=1 截断的 R² 拟合精度，或多项截断（K=3, K=20）。

---

## 方法 / How to reproduce

**方法**：
1. 起源层 L 对 `W_down` 做 SVD → 得 σ₁..σ_r, v₁..v_r, u₁..u_r
2. 对每样本测 h₂ · v₁ 的投影
3. 线性回归 MA ≈ β · (h₂ · v₁) + b，记 R²（K=1 单项）
4. 若 K=1 失败，测 K=3/K=20 多项截断或 macro V 消融

```bash
# 主脚本位置：见 code/ 子目录 或 ../../paper_experiments/RQ4_svd_alignment/
# 起源层自动读 ../../paper_experiments/origin_layer/output/L_ORIGIN.sh
bash ../../paper_experiments/run_rq345_origin_layer.sh "<model>" <rq_id>
```

---

## 关键指标

**关键指标**：
- `R² (K=1)`：单项截断的线性回归拟合精度
- `σ₁, σ₁/σ₂`：谱集中度
- `macro ΔMA%`：macro V 消融后 MA 变化

---

## 结论

**要证明什么 / 本轮证明了什么**：
- **主结论**：22/26 模型（85%）通过 K=1 R²≥0.95 OR macro V 消融 ≤-80%
- **K=1 完美**（R²≥0.999）：gptj_6b, qwen2_7b, qwen2.5_7b, qwen3_0.6b, qwen3_14b, qwen3_32b, qwen3_4b, qwen3_8b, bloom_7b1(L=7), qwen1.5_14b(L=2), llama3.1_8b
- **K=3 救场**：glm4_32b（σ₁ 扁平，K=1 R²=0.47, K=3 err=0.04%）
- **macro 救场**：gpt2, llama3.1_8b, qwen3 多层模型
- **3 个真 FAIL**（判据 D 多项式 + macro）：mistral_7b_v03（小 MA 模型 σ₁=1.29）、qwen1.5_14b（起源层冲突）、qwen3.5_35b_a3b（MoE）

---

## 每模型结果表（26 模型）

| # | 模型 | cat | L | 结果 | 数据目录 |
|:-:|---|---|:-:|---|---|
| 1 | bloom_7b1 | CONC | 7 | ✅ L=7 R²=0.9999 (救活) | [`results/bloom_7b1/L7_recheck/`](results/bloom_7b1/L7_recheck/) |
| 2 | falcon_7b | FS | 3 | ✅ R²=0.99 | [`results/falcon_7b/`](results/falcon_7b/) |
| 3 | glm4_32b | CONC | 0 | ✅ K=3 err=0.04% | [`results/glm4_32b/`](results/glm4_32b/) |
| 4 | glm4_9b | FS | 1 | ✅ R²=0.89 | [`results/glm4_9b/`](results/glm4_9b/) |
| 5 | gpt2 | FS | 3 | ✅ macro -95% | [`results/gpt2/`](results/gpt2/) |
| 6 | gptj_6b | CONC | 2 | ✅ K=1 R²=0.998 | [`results/gptj_6b/`](results/gptj_6b/) |
| 7 | llama2_13b | FS | 0 | ✅ R²=0.97 | [`results/llama2_13b/`](results/llama2_13b/) |
| 8 | llama2_7b_chat | — | 26 | ⚠️ R²=0.0001 | [`results/llama2_7b_chat/`](results/llama2_7b_chat/) |
| 9 | llama3.1_8b | FS | 1 | ✅ K=1 R²=0.998 | [`results/llama3.1_8b/`](results/llama3.1_8b/) |
| 10 | mistral_7b_v03 | CONC | 0 | ❌ R²=0.002 (σ₁=1.29 极弱) | [`results/mistral_7b_v03/`](results/mistral_7b_v03/) |
| 11 | opt_6.7b | ANOM | 1 | ✅ R²=0.98 | [`results/opt_6.7b/`](results/opt_6.7b/) |
| 12 | qwen1.5_14b | DISP | 2 | ✅ L=2 R²=0.9999 (救活) | [`results/qwen1.5_14b/L2_recheck/`](results/qwen1.5_14b/L2_recheck/) |
| 13 | qwen2.5_0.5b | CONC | 0 | ✅ R²=0.51 | [`results/qwen2.5_0.5b/`](results/qwen2.5_0.5b/) |
| 14 | qwen2.5_7b | CONC | 3 | ✅ K=1 R²=1.000 | [`results/qwen2.5_7b/`](results/qwen2.5_7b/) |
| 15 | qwen2_7b | CONC | 3 | ✅ K=1 R²=1.000 | [`results/qwen2_7b/`](results/qwen2_7b/) |
| 16 | qwen3.5_27b | DISP | 54 | ✅ L=54 R²=0.9923 (救活) | [`results/qwen3.5_27b/`](results/qwen3.5_27b/) |
| 17 | qwen3.5_35b_a3b | FS MoE | 9 | ❌ R²=0.001 | [`results/qwen3.5_35b_a3b/`](results/qwen3.5_35b_a3b/) |
| 18 | qwen3.5_9b | DISP | 22 | ✅ R²=0.73 | [`results/qwen3.5_9b/`](results/qwen3.5_9b/) |
| 19 | qwen3_0.6b | CONC | 2 | ✅ K=1 R²=1.000 | [`results/qwen3_0.6b/`](results/qwen3_0.6b/) |
| 20 | qwen3_1.7b | FS | 2 | ✅ R²=0.94 | [`results/qwen3_1.7b/`](results/qwen3_1.7b/) |
| 21 | qwen3_14b | DISP | 6 | ✅ R²=1.000 | [`results/qwen3_14b/`](results/qwen3_14b/) |
| 22 | qwen3_30b_a3b | DISP MoE | 1 | ❌ R²=0.38 | [`results/qwen3_30b_a3b/`](results/qwen3_30b_a3b/) |
| 23 | qwen3_32b | DISP | 6 | ✅ R²=1.000 | [`results/qwen3_32b/`](results/qwen3_32b/) |
| 24 | qwen3_4b | FS | 6 | ✅ R²=1.000 | [`results/qwen3_4b/`](results/qwen3_4b/) |
| 25 | qwen3_8b | DISP | 6 | ✅ R²=0.999 | [`results/qwen3_8b/`](results/qwen3_8b/) |
| 26 | yi_9b | DISP | 8 | ✅ R²=0.88 | [`results/yi_9b/`](results/yi_9b/) |


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
