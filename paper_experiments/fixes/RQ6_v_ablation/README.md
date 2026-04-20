# `fixes/RQ6_v_ablation/` — exp6_v_ablation.py 修复

## 修了什么（3 个 bug 合修）

### Bug B4：`get_mlp_down_proj` 缺 glm4/yi 分支

原代码不识别 glm4/yi，会抛 `ValueError: Unknown model`。

**修法**：SwiGLU 分支加 `"glm4" / "yi"`（+3 行）。

### Bug B5：`get_critical_layer` 默认 L0

原代码只硬编码了 6 个模型，其他默认返回 L0。导致 **20+ 模型在错层做 V 消融**。

典型错层（旧默认 → 正确值）：
- qwen3_32b: L0 → **L6**
- qwen1.5_14b: L0 → **L35**
- yi_9b: L0 → **L8**
- qwen3.5_27b: L0 → **L54**
- bloom_7b1: L28（过时硬编码）→ **L3**

**修法**（4 层优先级）：
1. 环境变量 `OVERRIDE_CRITICAL_LAYER`（CLI 调试）
2. 读 `paper_experiments/origin_layer/output/L_ORIGIN.json`（25 模型精确）
3. 内置 fallback 表（L_ORIGIN 的快照，部分部署用）
4. 未知模型 **raise ValueError**（不再静默默认 L0）

另加 argparse `--layer_id` 参数。

### Bug B6：baseline 只测 critical_layer 非真 MA

原代码只在 `critical_layer` 一层测 top1 激活。但 MA 的峰值一般在**更后面的层**（被 attention 广播放大后）。结果：

| 模型 | RQ2a 真 MA baseline | RQ6 旧 baseline | 差距 |
|---|:-:|:-:|:-:|
| glm4_32b | 298598 | **1.15** | 260000× |
| yi_9b | 5004 | 1.97 | 2540× |
| qwen3_32b | 27417 | 30.79 | 890× |

所有 `remove_top_K` / `keep_top_K` 百分比基于错误 baseline，glm4 甚至出现 "remove_top_1 后 MA 变 137%"（非物理）。

**修法**：hook 全部层 → 跨所有层扫 top1 → 取全局峰值。和 RQ2a 对齐。新增 `peak_layer` / `ablation_layer` 输出字段。

### 附加：MoE guard

在 `run_ablation_experiment()` 开头检测 `hasattr(layers[0].mlp, 'experts')`，MoE 模型**优雅跳过**（返回 `None`，不产出 json）。避免批量跑时 crash。

## 部署

这个脚本在 `changeHead_massvieAcitve` submodule 里（历史遗留），不在 paper_experiments 下：

```bash
cd <repo-root>
cp paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py \
   changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py
```

## 怎么跑

### 方式 A：自动读 L_ORIGIN.json（推荐）

```bash
cd <repo-root>
python changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py \
    --model qwen3_32b \
    --nsamples 30 \
    --savedir results/wikitext_run/RQ6/qwen3_32b
# 脚本自动查 paper_experiments/origin_layer/output/L_ORIGIN.json，用 L=6
```

### 方式 B：CLI 显式指定（调试 / 覆盖）

```bash
python exp6_v_ablation.py \
    --model qwen3_32b \
    --layer_id 6 \
    --nsamples 30 \
    --savedir /tmp/test
```

### 方式 C：环境变量覆盖

```bash
OVERRIDE_CRITICAL_LAYER=6 python exp6_v_ablation.py \
    --model qwen3_32b --nsamples 30 --savedir /tmp/test
```

### 批量跑 24 dense 模型

```bash
# source L_ORIGIN（用 bash 4+ / homebrew bash）
source paper_experiments/origin_layer/output/L_ORIGIN.sh

for model in bloom_7b1 falcon_7b glm4_9b glm4_32b gpt2 gptj_6b \
             llama2_13b llama2_7b_chat llama3.1_8b mistral_7b_v03 opt_6.7b \
             qwen1.5_14b qwen2.5_0.5b qwen2.5_7b qwen2_7b \
             qwen3_0.6b qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b qwen3_32b \
             qwen3.5_9b qwen3.5_27b yi_9b; do
    L="${L_ORIGIN[$model]}"
    python changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py \
        --model $model --layer_id "$L" --nsamples 30 \
        --savedir results/wikitext_run/RQ6/$model
done
# MoE 2 个（qwen3_30b_a3b / qwen3.5_35b_a3b）脚本内自动 skip
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必填 | 模型名 |
| `--layer_id` | **None**（自动读 L_ORIGIN.json）| 起源层覆盖 |
| `--nsamples` | 5 | 采样数（**建议 ≥ 30** 降低 variance）|
| `--k_values` | `[1, 5, 10, 50, 100]` | remove/keep top-K 的 K 值列表 |
| `--savedir` | 必填 | 输出目录 |

## 预期输出

```
results/wikitext_run/RQ6/<model>/
└── v_ablation_results.json
```

### `v_ablation_results.json` 结构

```json
{
  "model": "qwen3_32b",
  "critical_layer": 6,
  "weight_shape": [4096, 14336],
  "singular_values_top10": [...],
  "sigma_ratio": 1.83,
  "baseline": {
    "top1": 27417.6,          // ← 修复后：真 MA（和 RQ2a 一致）
    "top1_values": [...],
    "top1_avg": 27400.2,
    "peak_layer": 53,         // ← 修复后新增：MA 实际峰值层
    "ablation_layer": 6       // ← 修复后新增：哪层被消融
  },
  "ablation_results": {
    "remove_top_k": {
      "1": {"value": ..., "pct_of_baseline": 34.1, ...},
      ...
    },
    "keep_top_k": {
      "1": {"value": ..., "pct_of_baseline": 97.9, ...},
      ...
    }
  }
}
```

### MoE 模型输出

```
results/wikitext_run/RQ6/qwen3_30b_a3b/
(无输出，脚本打印 "⚠ qwen3_30b_a3b is a MoE model. Skipping..." 直接返回)
```

## 验证修复

```bash
# sentinel 检查（Test C/D/E）
bash paper_experiments/fixes/sentinel_test.sh

# 手动验证 glm4_9b 的 baseline
# 修复前: baseline ≈ 4.58（错的，只在 L0 测）
# 修复后: baseline ≈ 2250（对的，和 RQ2a 一致）
python -c "
import json
d = json.load(open('results/wikitext_run/RQ6/glm4_9b/v_ablation_results.json'))
rq2a = json.load(open('results/wikitext_run/RQ2a/glm4_9b/baseline/results.json'))
rq6_bl = d['baseline']['top1']
rq2a_bl = max(v['top1_mean'] for v in rq2a.values() if isinstance(v, dict))
ratio = rq2a_bl / rq6_bl
print(f'RQ2a baseline: {rq2a_bl:.1f}')
print(f'RQ6 baseline:  {rq6_bl:.1f}')
print(f'Ratio: {ratio:.1f}× (should be < 10×)')
"
```

## MoE 模型

qwen3_30b_a3b + qwen3.5_35b_a3b 自动 skip。Tier C 专项另写 `exp6_v_ablation_moe_per_expert.py`，主结论定稿后再做。

## 下游影响

原 `v_ablation_results.json` 的老字段**完全保留**。新增 `peak_layer` / `ablation_layer` 字段（可选读）。`baseline.top1` 数量级**变大**（和 RQ2a 一致），所有 `pct_of_baseline` 百分比**值变化**（原 137% 会回到 <50%，原 95% 可能微调）。

老脚本消费者（plotting、summary 聚合）读 `pct_of_baseline` 时可能需要重新校准阈值。
