# RQ5: V 矩阵几何方向是 MA 的因果必要条件

## 文件

- `exp5_v_ablation.py` — 真实 V-matrix 消融实验（**核心脚本**）
- `exp5_mock_validation.py` — 合成数据验证（用于方法学验证，非论文数据）
- `exp5_validation_report.py` — 验证报告生成（辅助）

## 实验做什么

1. 测量 baseline MA（正常模型）
2. 对 W_down 做 SVD：W = U @ Σ @ V^T
3. 用 QR 分解随机高斯矩阵生成 V_rand（随机正交矩阵）
4. 用 W_ablated = U @ Σ @ V_rand^T 替换 W_down
5. 测量消融后 MA
6. 恢复原始权重
7. 计算 ΔMA 并与理论预测（Eq. 10）对比

## 产出数据

- `{model}_v_ablation_results.json` — 完整结果：
  - baseline/ablated MA 值
  - ΔMA 百分比（Table 1 的 ΔMA 列）
  - σ₁、σ₂、η
  - 理论预测值

## 运行

```bash
cd paper_experiments

# GPT-2
python RQ5_v_matrix_ablation/exp5_v_ablation.py \
    --model gpt2 --layer_id 2 --nsamples 30 --savedir results/RQ5/gpt2

# 全部 8 个模型
declare -A LAYERS=( [gpt2]=2 [llama2_13b]=22 [bloom_7b]=12 [gptj_6b]=0 [qwen2.5_7b]=0 [opt_7b]=25 [falcon_7b]=0 [mistral_7b]=0 )
for model in "${!LAYERS[@]}"; do
    python RQ5_v_matrix_ablation/exp5_v_ablation.py \
        --model $model --layer_id ${LAYERS[$model]} --nsamples 30 \
        --savedir results/RQ5/$model
done
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --model | gpt2 | 模型名 |
| --layer_id | 2 | MA 触发层 |
| --nsamples | 30 | 样本数 |
| --seed | 0 | 随机种子 |
| --savedir | results/exp5_v_ablation/ | 输出目录 |
