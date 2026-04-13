# RQ3: 功能词是 MA 的触发因素

## 文件

- `exp5_function_words_svd_mapping.py` — 唯一实验脚本

## 实验做什么

对每个 token 判断是否为功能词（the/of/and 等），计算功能词触发 MA 的比例（R_func）。同时分析功能词在 W_down 的 SVD 空间中的集中度、稳定性和 v₁ 对齐度。

## 产出数据

- `table1_rq3.json` — Table 1 需要的数据：Func.(%)
- `exp5_detailed_results.json` — 完整分析结果（集中度/不对称/稳定性/对齐）
- 4 张分析图 + 文本报告

## 运行

```bash
cd paper_experiments

python RQ3_function_words/exp5_function_words_svd_mapping.py \
    --model gpt2 --layer_id 2 --nsamples 50 --savedir results/RQ3/gpt2

# LLaMA-2（注意 layer_id 不同）
python RQ3_function_words/exp5_function_words_svd_mapping.py \
    --model llama2_13b --layer_id 22 --nsamples 50 \
    --access_token YOUR_TOKEN --savedir results/RQ3/llama2_13b
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | gpt2 | 模型名 |
| --layer_id | 2 | MA 触发层（不同模型不同，见 RQ2 的表） |
| --nsamples | 50 | 样本数 |
| --seqlen | 1024 | 序列长度 |
| --savedir | results/exp5_svd_mapping/ | 输出目录 |
