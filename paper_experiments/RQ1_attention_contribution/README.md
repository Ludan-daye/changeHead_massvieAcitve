# RQ1: Attention 是否产生 Massive Activation

## 文件

- `exp1_feasibility_test.py` — 唯一实验脚本

## 实验做什么

禁用所有 Attention Head（输出置零），对比禁用前后各层的 Top1 激活值。如果 MA 消失说明 Attention 是来源，如果不变说明来源在别处（MLP）。

## 产出数据

- `table1_rq1.json` — Table 1 需要的数据：Key Layer、Base MA、ΔTop1(%)
- `baseline/results.json` — 基线各层激活统计
- `all_heads_disabled/results.json` — 干预后各层激活统计
- `comparison/` — 4 张对比图 + 文本报告

## 运行

```bash
cd paper_experiments

# GPT-2（最快，用于验证）
python RQ1_attention_contribution/exp1_feasibility_test.py \
    --model gpt2 --nsamples 30 --savedir results/RQ1/gpt2

# LLaMA-2-13B（需要 access token）
python RQ1_attention_contribution/exp1_feasibility_test.py \
    --model llama2_13b --nsamples 30 --access_token YOUR_TOKEN \
    --savedir results/RQ1/llama2_13b

# 论文覆盖的全部 8 个模型，每个跑一次
for model in gpt2 llama2_13b bloom_7b gptj_6b qwen2.5_7b opt_7b falcon_7b mistral_7b; do
    python RQ1_attention_contribution/exp1_feasibility_test.py \
        --model $model --nsamples 30 --savedir results/RQ1/$model
done
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | gpt2 | 模型名（见 lib/model_dict.py） |
| --nsamples | 30 | 样本数 |
| --dataset | wikitext | 数据集 |
| --seed | 0 | 随机种子 |
| --savedir | results/exp1_feasibility_test/ | 输出目录 |
| --access_token | — | HuggingFace token（LLaMA 等需要） |
