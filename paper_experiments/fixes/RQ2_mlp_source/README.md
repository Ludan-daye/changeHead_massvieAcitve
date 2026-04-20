# `fixes/RQ2_mlp_source/` — exp2a_mlp_feasibility_test.py 修复

## 修了什么

**Bug B7**：`MLPDisableHook.__call__` 原本对 MoE 模型的 tuple 返回值处理错误。

### 旧代码（错）

```python
def __call__(self, module, input, output):
    if self.mode == 'disable_all':
        return torch.zeros_like(output)   # ← 对 tuple 行为未定义
```

### 症状

- MoE 模型（如 qwen3.5_35b_a3b）的 `SparseMoeBlock.forward` 返回 `(hidden_states, router_logits)` tuple
- `torch.zeros_like(tuple)` 不会正确置零——部分输出保留，qwen3.5_35b_a3b RQ2a 出现 `retain=81%` 异常（其他 MoE 如 qwen3_30b_a3b 是 2.85% 正常）

## 修复方法

加 `isinstance(output, tuple)` 分支：

```python
def __call__(self, module, input, output):
    if self.mode == 'disable_all':
        # 处理 MoE tuple: 置零 hidden，保留 router_logits
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        # Dense: 简单置零
        return torch.zeros_like(output)
    else:
        return output
```

参照 `exp6_progressive_ablation.py:32-35` 的正确写法。

## 部署

```bash
cd <repo-root>
cp paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
   paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py
```

## 怎么跑（以 qwen3.5_35b_a3b 为例）

```bash
cd paper_experiments
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
    --model qwen3.5_35b_a3b \
    --nsamples 30 \
    --savedir results/wikitext_run/RQ2a/qwen3.5_35b_a3b
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必填 | 模型名（见 `lib/model_dict.py`）|
| `--nsamples` | 30 | 采样 wikitext 样本数 |
| `--dataset` | wikitext | 数据集 |
| `--seed` | 0 | 随机种子 |
| `--savedir` | 必填 | 输出目录 |
| `--access_token` | — | HuggingFace token（某些模型需要）|

## 预期输出

```
results/wikitext_run/RQ2a/qwen3.5_35b_a3b/
├── baseline/results.json              # 原模型每层 top1/median
├── all_mlp_disabled/results.json      # 关 MLP 后每层 top1/median
├── comparison/
│   ├── exp2a_top1_comparison.png
│   ├── exp2a_layerwise_breakdown.png
│   ├── exp2a_percentage_change_heatmap.png
│   ├── exp2a_critical_dimensions.png
│   └── EXPERIMENT_2A_SUMMARY.txt      # 人类可读汇总
```

## 修复验证

```bash
# 方法 1：跑 sentinel（自动 Test B）
bash paper_experiments/fixes/sentinel_test.sh

# 方法 2：qwen3.5_35b_a3b 修复前后对比
# 修复前：retain=81.08%
# 修复后：retain 应该 < 10%（和 qwen3_30b_a3b 的 2.85% 接近）
```

## 对非 MoE 模型的影响

**零影响**——非 tuple 分支走原逻辑，行为完全不变。

## 什么都没改的东西

- `run_experiment` / 数据流 / 输出格式完全不变
- 其他 hook 逻辑不变
- argparse 不变（无新增参数）
