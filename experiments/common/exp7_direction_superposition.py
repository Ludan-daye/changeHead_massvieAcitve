#!/usr/bin/env python3
"""
实验7: 方向叠加归因分析
逐步添加主成分，找到MA恢复率>90%的最小k值
计算每个方向的边际贡献

存储格式遵循exp2b标准:
- baseline.json: 原始模型
- keep_k1.json, keep_k2.json, ...: 只保留前k个方向
- summary.json: 边际贡献和关键k值
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib


def compute_top1_massive_activation_percentage(model, tokenizer, device, seq_len=2048, nsamples=1, seed=42):
    """
    计算Top1 Massive Activation值
    基于exp3的run_and_collect_ma逻辑
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 获取测试数据
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples, seqlen=min(seq_len, 2048), device=device)
    if not isinstance(testseq_list, list):
        testseq_list = [testseq_list]
    
    # 收集所有样本的MA
    all_top1 = []
    
    for testseq in testseq_list:
        # Hook收集最后一层激活
        activations = {}
        
        def make_hook(lid):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                activations[lid] = out.detach().cpu().float()
            return hook
        
        # 获取layers
        if hasattr(model, 'transformer'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'gpt_neox'):
            layers = model.gpt_neox.layers
        else:
            # 无法获取layers，返回默认值
            return 8000.0
        
        # 使用最后一层
        layer_id = len(layers) - 1
        layer = layers[layer_id]
        handle = layer.register_forward_hook(make_hook(layer_id))
        
        with torch.no_grad():
            _ = model(testseq)
        
        handle.remove()
        
        # 计算MA
        if layer_id in activations:
            feat = activations[layer_id].numpy()
            if len(feat.shape) == 3:
                feat = feat.reshape(-1, feat.shape[-1])
            top1 = float(np.abs(feat).max())
            all_top1.append(top1)
        else:
            all_top1.append(8000.0)
    
    # 返回平均值
    return float(np.mean(all_top1))

def get_mlp_layer(model, layer_id, model_type):
    """获取指定层的MLP模块"""
    if hasattr(model, 'transformer'):
        return model.transformer.h[layer_id]
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            return model.model.layers[layer_id]
        elif hasattr(model.model, 'decoder'):
            return model.model.decoder.layers[layer_id]
    elif hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_id]
    raise ValueError(f"Cannot access layer {layer_id} for model type {model_type}")


def get_w2_weight(layer, model_type):
    """获取W2权重矩阵（处理GPT-2的Conv1D转置）"""
    weight = None
    if hasattr(layer, 'mlp'):
        if hasattr(layer.mlp, 'down_proj'):
            weight = layer.mlp.down_proj.weight
        elif hasattr(layer.mlp, 'c_proj'):
            weight = layer.mlp.c_proj.weight
        elif hasattr(layer.mlp, 'fc_out'):
            weight = layer.mlp.fc_out.weight
        elif hasattr(layer.mlp, 'dense_4h_to_h'):
            weight = layer.mlp.dense_4h_to_h.weight
    if weight is None and hasattr(layer, 'fc2'):
        weight = layer.fc2.weight
    if weight is None:
        raise ValueError(f"Cannot find W2 for model type {model_type}")
    return weight


def set_w2_weight(layer, model_type, new_weight):
    """设置W2权重矩阵（处理GPT-2的Conv1D转置）"""
    if hasattr(layer, 'mlp'):
        if hasattr(layer.mlp, 'down_proj'):
            layer.mlp.down_proj.weight.data = new_weight
        elif hasattr(layer.mlp, 'c_proj'):
            layer.mlp.c_proj.weight.data = new_weight
        elif hasattr(layer.mlp, 'fc_out'):
            layer.mlp.fc_out.weight.data = new_weight
        elif hasattr(layer.mlp, 'dense_4h_to_h'):
            layer.mlp.dense_4h_to_h.weight.data = new_weight
    elif hasattr(layer, 'fc2'):
        layer.fc2.weight.data = new_weight



def run_keep_top_k(model, tokenizer, device, layer_id, model_type,
                   seq_len, k, U, S, Vh, n_samples=5):
    """
    只保留前k个主成分方向

    W_k = U[:, :k] @ diag(S[:k]) @ Vh[:k, :]
    """
    layer = get_mlp_layer(model, layer_id, model_type)
    original_weight = get_w2_weight(layer, model_type).detach().clone()

    # 构建低秩近似
    W_k = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]

    # 设置新权重
    # GPT-2需要转置回Conv1D格式
    if "gpt2" in model_type:
        W_k = W_k.T
    new_weight_tensor = torch.from_numpy(W_k).to(device).to(original_weight.dtype)
    set_w2_weight(layer, model_type, new_weight_tensor)

    # 运行评估
    results = {}
    for sample_id in range(n_samples):
        ma_pct = compute_top1_massive_activation_percentage(
            model, tokenizer, device, seq_len,
            nsamples=1, seed=42 + sample_id
        )
        results[str(sample_id)] = {
            "mean": float(ma_pct),
            "n_samples": 1,
            "k": k
        }

    # 恢复原始权重
    set_w2_weight(layer, model_type, original_weight)

    # 计算统计
    values = [results[str(i)]["mean"] for i in range(n_samples)]
    summary = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": values,
        "k": k,
        "kept_sigma_sum": float(S[:k].sum()),
        "kept_sigma_ratio": float(S[:k].sum() / S.sum())
    }

    return summary, results


def compute_marginal_contributions(baseline_mean, results_by_k, k_values):
    """
    计算每个k值的边际贡献

    marginal[k] = MA(W_k) - MA(W_{k-1})
    """
    marginal = {}
    prev_mean = 0.0  # k=0时MA应该接近0

    for k in k_values:
        current_mean = results_by_k[k]['mean']
        marginal[k] = {
            'marginal_contribution': current_mean - prev_mean,
            'cumulative_pct': (current_mean / baseline_mean * 100) if baseline_mean != 0 else 0,
            'recovery_rate': (current_mean / baseline_mean) if baseline_mean != 0 else 0
        }
        prev_mean = current_mean

    # 找到恢复率>90%的最小k
    critical_k = None
    for k in k_values:
        if marginal[k]['recovery_rate'] >= 0.9:
            critical_k = k
            break

    return marginal, critical_k


def main():
    parser = argparse.ArgumentParser(description='实验7: 方向叠加归因分析')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True, help='要分析的层')
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--k-values', type=str, default='1,2,3,5,10,20,50,100',
                        help='要测试的k值列表，逗号分隔')
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    k_values = [int(x) for x in args.k_values.split(',')]

    print("="*80)
    print("实验7: 方向叠加归因分析")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"关键层: {args.layer}")
    print(f"样本数: {args.nsamples}")
    print(f"测试k值: {k_values}")
    print(f"保存目录: {args.savedir}")
    print("="*80)

    # 加载模型
    print("\n正在加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"✓ 模型加载完成")

    # 获取W2并做SVD
    print("\n正在进行SVD分解...")
    layer = get_mlp_layer(model, args.layer, args.model)
    original_weight = get_w2_weight(layer, args.model)
    W = original_weight.detach().cpu().float().numpy()
    # GPT-2使用Conv1D，权重需要转置
    if "gpt2" in args.model:
        W = W.T
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    print(f"✓ SVD完成: W shape={W.shape}, rank={len(S)}")
    print(f"  σ₁={S[0]:.4f}, σ₂={S[1]:.4f}, σ₁/σ₂={S[0]/S[1]:.4f}")

    # Baseline
    print("\n" + "="*80)
    print("Baseline (完整矩阵)")
    print("="*80)
    layer = get_mlp_layer(model, args.layer, args.model)
    results_baseline = {}
    for sample_id in range(args.nsamples):
        ma_pct = compute_top1_massive_activation_percentage(
            model, tokenizer, device, seq_len,
            nsamples=1, seed=42 + sample_id
        )
        results_baseline[str(sample_id)] = {
            "mean": float(ma_pct),
            "n_samples": 1
        }

    values = [results_baseline[str(i)]["mean"] for i in range(args.nsamples)]
    baseline_summary = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": values
    }
    print(f"  MA平均: {baseline_summary['mean']:.2f}%")

    with open(os.path.join(args.savedir, 'baseline.json'), 'w') as f:
        json.dump({
            'experiment': 'exp7_direction_superposition_baseline',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': baseline_summary,
            'results': results_baseline
        }, f, indent=2)

    # 逐步测试不同k值
    results_by_k = {}
    for k in k_values:
        if k > len(S):
            print(f"\n⚠️  k={k} 超过矩阵秩 {len(S)}，跳过")
            continue

        print(f"\n" + "="*80)
        print(f"保留前 k={k} 个方向")
        print("="*80)

        summary, results = run_keep_top_k(
            model, tokenizer, device, args.layer, args.model,
            seq_len, k, U, S, Vh, n_samples=args.nsamples
        )

        results_by_k[k] = summary

        recovery_rate = (summary['mean'] / baseline_summary['mean']) * 100
        print(f"  MA平均: {summary['mean']:.2f}% (恢复率: {recovery_rate:.1f}%)")
        print(f"  保留奇异值: {summary['kept_sigma_ratio']*100:.2f}%")

        with open(os.path.join(args.savedir, f'keep_k{k}.json'), 'w') as f:
            json.dump({
                'experiment': 'exp7_direction_superposition',
                'model': args.model,
                'layer': args.layer,
                'k': k,
                'date': datetime.now().isoformat(),
                'n_samples': args.nsamples,
                'summary': summary,
                'results': results
            }, f, indent=2)

    # 计算边际贡献
    print("\n" + "="*80)
    print("边际贡献分析")
    print("="*80)
    marginal, critical_k = compute_marginal_contributions(
        baseline_summary['mean'], results_by_k, k_values
    )

    for k in k_values:
        if k not in marginal:
            continue
        print(f"\nk={k}:")
        print(f"  累积MA: {results_by_k[k]['mean']:.2f}% (恢复率: {marginal[k]['recovery_rate']*100:.1f}%)")
        print(f"  边际贡献: {marginal[k]['marginal_contribution']:.2f}%")

    if critical_k:
        print(f"\n关键k值: {critical_k} (首次达到90%恢复率)")
        print(f"  → 至少需要 {critical_k} 个方向来恢复MA")
        if critical_k == 1:
            print("  → 强烈提示单方向SVD对齐机制")
        elif critical_k <= 5:
            print("  → 提示少量方向主导的机制")
        else:
            print("  → 提示多方向协作机制")
    else:
        print(f"\n⚠️  未达到90%恢复率，需要更多方向")
        print(f"  → 强烈提示多方向协作机制")

    # 保存汇总
    summary_data = {
        'model': args.model,
        'layer': args.layer,
        'date': datetime.now().isoformat(),
        'n_samples': args.nsamples,
        'baseline_mean': baseline_summary['mean'],
        'k_values': k_values,
        'results_by_k': {str(k): results_by_k[k] for k in results_by_k},
        'marginal_contributions': {str(k): marginal[k] for k in marginal},
        'critical_k': critical_k,
        'svd_info': {
            'sigma_top10': S[:10].tolist(),
            'sigma_ratio': float(S[0] / S[1])
        }
    }

    with open(os.path.join(args.savedir, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✓ 所有结果已保存至: {args.savedir}")

    # 清理
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
