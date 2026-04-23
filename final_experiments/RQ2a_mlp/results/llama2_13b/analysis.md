# llama2_13b — RQ2a（MLP 消融）

**模型分类**：`SINGLE`（今日 2026-04-24 重分类，原 FS 是 RQ2c 误判）| **真起源层 L**：`0`（peak_layer = L=22）

**本 RQ 在问什么**：MLP 全消融：禁用全部 MLP 后 MA 是否归零（验证 'MLP 是 MA 起源'）

---

## 关键指标（2026-04-24 RQ2a 实跑，78s/sample × 60 samples）

| 指标 | 值 |
|---|---:|
| baseline Top1 Peak @ L=22 | **1282.80** |
| disabled Top1 @ L=22 | **49.28** |
| **retain%** | **3.84%** |
| **reduction%** | **-96.16%** |
| baseline Primary dim @ L=3 | 69.91 |
| disabled Primary dim | 13.90 (-80.13%) |

**判据**：retain ≤ 10% → **✅ PASS**

**结论**：MLP 几乎完全承担 MA 生成；关全 MLP 后 MA 从 1282 → 49（-96%），强证据支持"MLP 是 MA 起源"主论点。

---

## 数据文件

- `data/EXPERIMENT_2A_SUMMARY.txt` — 完整实验总结
- `data/baseline_results.json` — 40 层 baseline 每层 MA/mlp_output_mean/attn_output_mean/比率
- `data/all_mlp_disabled_results.json` — 全 MLP 关后的实测

## 脚本

`paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py`（OPT hook fix 后版本，同样适用 LLaMA 架构）

## 备注

- 今日（2026-04-24）把 Meta 原生 PyTorch 权重用 `transformers/models/llama/convert_llama_weights_to_hf.py` 转成 HF 格式（主服本地 56s 完成），解决之前 HF 401 缺数据问题
- 转换后权重路径（主服）：`model_weights/llama2_13b_hf/`

---

## 总评

**此模型 × 此 RQ**：✅ PASS

**此模型综合评分**：**5/5 ⭐**

参见 [../README.md](../README.md) 和 [../../STATUS.md](../../STATUS.md)。
