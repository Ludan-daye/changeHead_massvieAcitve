# Paper Experiments

论文 "Function Words as Geometric Anchors" 实验代码。每个 RQ 一个文件夹，文件夹内只有实验脚本和使用说明。

## 环境

```bash
conda create -n massive-activations python=3.9 -y
conda activate massive-activations
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 目录结构

```
paper_experiments/
├── RQ1_attention_contribution/   ← Attention 是否产生 MA
│   ├── exp1_feasibility_test.py
│   └── README.md
├── RQ2_mlp_source/               ← MLP 是 MA 的物理来源
│   ├── exp2a_mlp_feasibility_test.py
│   ├── exp2c_mlp_internal_analysis.py
│   └── README.md
├── RQ3_function_words/           ← 功能词是 MA 的触发因素
│   ├── exp5_function_words_svd_mapping.py
│   └── README.md
├── RQ4_svd_alignment/            ← SVD 几何对齐解释 MA 机制
│   ├── exp3_svd_alignment_analysis.py
│   └── README.md
├── RQ5_v_matrix_ablation/        ← V 矩阵是 MA 的因果必要条件
│   ├── exp5_v_ablation.py
│   └── README.md
├── lib/                          ← 共享库（模型加载/数据/评估/绘图）
├── monkey_patch/                 ← 激活捕获 hook
├── main_llm.py                   ← LLM 统一入口
├── main_vit.py                   ← ViT 统一入口
└── requirements.txt
```

## 快速开始

```bash
cd paper_experiments

# 用 GPT-2 跑全部 5 个实验
python RQ1_attention_contribution/exp1_feasibility_test.py --model gpt2 --savedir results/RQ1/gpt2
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py --model gpt2 --savedir results/RQ2/gpt2
python RQ2_mlp_source/exp2c_mlp_internal_analysis.py --model gpt2 --layer_id 2 --savedir results/RQ2c/gpt2
python RQ3_function_words/exp5_function_words_svd_mapping.py --model gpt2 --layer_id 2 --savedir results/RQ3/gpt2
python RQ4_svd_alignment/exp3_svd_alignment_analysis.py --model gpt2 --layer_id 2 --savedir results/RQ4/gpt2
python RQ5_v_matrix_ablation/exp5_v_ablation.py --model gpt2 --layer_id 2 --savedir results/RQ5/gpt2
```

## Table 1 数据产出

每个实验会输出一个 `table1_rqX.json`，合并即可填充论文 Table 1：

| Table 1 列 | 来源文件 | JSON 字段 |
|---|---|---|
| Key Layer | `table1_rq1.json` | `key_layer` |
| ΔTop1 | `table1_rq1.json` | `delta_top1_pct` |
| Base MA | `table1_rq1.json` | `base_ma` |
| Func.(%) | `table1_rq3.json` | `func_pct` |
| σ₁/σ₂ | `table1_rq4.json` | `sigma1_sigma2` |
| Cos Sim | `table1_rq4.json` | `cos_sim` |
| ΔMA | `{model}_v_ablation_results.json` | `delta_ma.top1_mean_pct` |

## 支持的模型

| --model 参数 | 模型 | MA 触发层 (--layer_id) |
|---|---|---|
| gpt2 | GPT-2 (124M) | 2 |
| llama2_13b | LLaMA-2-13B | 22 |
| bloom_7b | BLOOM-7B1 | 12 |
| gptj_6b | GPT-J-6B | 0 |
| qwen2.5_7b | Qwen2.5-7B | 0 |
| opt_7b | OPT-6.7B | 25 |
| falcon_7b | Falcon-7B | 0 |
| mistral_7b | Mistral-7B | 0 |

需要 access token 的模型加 `--access_token YOUR_TOKEN`。
