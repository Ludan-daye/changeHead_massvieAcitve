# RQ2: MLP 是 MA 的物理来源

## 文件

- `exp2a_mlp_feasibility_test.py` — MLP vs Attention 幅度对比
- `exp2c_mlp_internal_analysis.py` — MLP 内部 4 阶段追踪

## 实验做什么

**exp2a**：在每层同时捕获 Attention 和 MLP 的输出，计算 max(|H_mlp|)/max(|H_attn|) 比值。同时禁用全部 MLP 观察 MA 是否消失。

**exp2c**：在目标层 MLP 内部 4 个检查点注册 hook（输入 → Up-projection → Activation → Down-projection），追踪 MA 在哪个阶段被放大。

## 产出数据

**exp2a**：
- `baseline/results.json` — 各层激活统计 + MLP/Attn 比值
- `all_mlp_disabled/results.json` — 禁用 MLP 后的统计
- `comparison/` — 对比图 + 报告

**exp2c**：
- `exp2c_detailed_results.json` — 4 阶段激活统计 + 权重分析
- 3 张分析图 + 文本报告

## 运行

```bash
cd paper_experiments

# exp2a: MLP vs Attention 幅度对比
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
    --model gpt2 --nsamples 30 --savedir results/RQ2/exp2a/gpt2

# exp2c: MLP 内部追踪（指定目标层）
python RQ2_mlp_source/exp2c_mlp_internal_analysis.py \
    --model gpt2 --layer_id 2 --nsamples 30 --savedir results/RQ2/exp2c/gpt2
```

## 各模型的目标层（--layer_id）

| 模型 | layer_id |
|------|----------|
| GPT-2 | 2 |
| LLaMA-2-13B | 22 |
| BLOOM-7B1 | 12 |
| GPT-J-6B | 0 |
| OPT-6.7B | 25 |
| Falcon-7B | 0 |
| Mistral-7B | 0 |
