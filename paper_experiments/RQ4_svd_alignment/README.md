# RQ4: SVD 几何对齐解释 MA 的产生机制

## 文件

- `exp3_svd_alignment_analysis.py` — 唯一实验脚本

## 实验做什么

1. 对目标层 W_down 做 SVD 分解，提取 σ₁、σ₂、η=σ₁/σ₂、v₁
2. 前向推理收集每个 token 的中间表征 h₂，计算 cos(h₂, v₁)
3. 对比功能词 vs 内容词的对齐度分布
4. 回归分析：log(MA) ~ log(|h₂·v₁|)，得到 R²
5. 自动分类 MA 类型（Type I/II/III）

## 产出数据

- `table1_rq4.json` — Table 1 需要的数据：σ₁/σ₂、Cos Sim、MA Type、R²
- `exp3_detailed_results.json` — 完整 SVD 分析 + 回归结果
- 5 张分析图 + 文本报告

## 运行

```bash
cd paper_experiments

python RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
    --model gpt2 --layer_id 2 --nsamples 50 --savedir results/RQ4/gpt2

# Qwen2.5（注意需要找到正确的 MA 触发层）
python RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
    --model qwen2.5_7b --layer_id 0 --nsamples 50 --savedir results/RQ4/qwen2.5
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | gpt2 | 模型名 |
| --layer_id | 2 | MA 触发层 |
| --nsamples | 50 | 样本数 |
| --seqlen | 1024 | 序列长度 |
| --savedir | results/exp3_svd_alignment/ | 输出目录 |
