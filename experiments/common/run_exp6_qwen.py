#!/usr/bin/env python3
"""
为Qwen2.5-7B重新运行Exp6实验
Keep Top-k V方向实验
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# 禁用代理
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import lib

# 配置
MODEL_NAME = 'qwen2.5_7b'
CRITICAL_LAYER = 3
NSAMPLES = 5
K_VALUES = [1, 2, 3, 5, 10, 20, 50, 100]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, f'results/experiments/exp6/{MODEL_NAME}')


def run_and_get_ma(model, layers, testseq, layer_id):
    """运行模型并获取MA"""
    activations = {}

    def make_hook(lid):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations[lid] = out.detach().cpu().float()
        return hook

    handle = layers[layer_id].register_forward_hook(make_hook(layer_id))

    with torch.no_grad():
        _ = model(testseq)

    handle.remove()

    feat = activations[layer_id].numpy()
    if len(feat.shape) == 3:
        feat = feat.reshape(-1, feat.shape[-1])

    top1 = np.abs(feat).max()
    return float(top1)


def main():
    print("=" * 80)
    print(f"Exp6: Qwen2.5-7B Keep Top-k V实验")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建args
    class Args:
        model = MODEL_NAME

    args = Args()

    # 加载模型
    print(f"\n加载模型: {MODEL_NAME}")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    print(f"✓ 模型加载完成 (层数: {len(layers)})")

    # 获取测试数据
    print(f"\n加载测试数据 (nsamples={NSAMPLES})...")
    testseq_list = lib.get_data(tokenizer, nsamples=NSAMPLES,
                                 seqlen=min(seq_len, 2048), device=device)
    if not isinstance(testseq_list, list):
        testseq_list = [testseq_list]

    # 获取权重
    layer = layers[CRITICAL_LAYER]
    down_proj = layer.mlp.down_proj
    W_original = down_proj.weight.data.cpu().float().numpy()
    print(f"权重shape: {W_original.shape}")

    # SVD分解
    print("\nSVD分解...")
    U, S, Vh = np.linalg.svd(W_original, full_matrices=False)
    print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
    sigma_ratio = float(S[0] / S[1])
    print(f"σ₁/σ₂ = {sigma_ratio:.2f}")
    print(f"Top 10 奇异值: {S[:10].tolist()}")

    # 1. Baseline
    print(f"\n{'='*60}")
    print("Baseline (原始权重)")
    print("="*60)
    baseline_mas = []
    for i, testseq in enumerate(testseq_list):
        ma = run_and_get_ma(model, layers, testseq, CRITICAL_LAYER)
        baseline_mas.append(ma)
        print(f"  样本 {i+1}: MA = {ma:.2f}")
    baseline_avg = np.mean(baseline_mas)
    print(f"\n✓ Baseline MA平均: {baseline_avg:.2f}")

    # 保存baseline
    baseline_data = {
        'model': MODEL_NAME,
        'layer': CRITICAL_LAYER,
        'ma_values': baseline_mas,
        'ma_avg': float(baseline_avg)
    }
    with open(os.path.join(OUTPUT_DIR, 'baseline.json'), 'w') as f:
        json.dump(baseline_data, f, indent=2)

    # 2. Keep Top-k实验
    print(f"\n{'='*60}")
    print("Keep Top-k V方向实验")
    print("="*60)

    results_by_k = {}

    for k in K_VALUES:
        print(f"\n  k={k}: 只保留前{k}个V方向...")

        # 消融权重: W' = U[:, :k] @ diag(S[:k]) @ Vh[:k, :]
        W_ablated = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]

        # 临时替换权重
        with torch.no_grad():
            down_proj.weight.data = torch.tensor(W_ablated,
                                                 dtype=down_proj.weight.dtype,
                                                 device=down_proj.weight.device)

        # 测试
        ablated_mas = []
        for testseq in testseq_list:
            ma = run_and_get_ma(model, layers, testseq, CRITICAL_LAYER)
            ablated_mas.append(ma)

        ablated_avg = np.mean(ablated_mas)
        ablated_std = np.std(ablated_mas)
        ablated_min = np.min(ablated_mas)
        ablated_max = np.max(ablated_mas)

        kept_sigma_sum = float(np.sum(S[:k]))
        kept_sigma_ratio = float(np.sum(S[:k]) / np.sum(S))

        print(f"    MA平均: {ablated_avg:.2f} (std={ablated_std:.2f})")
        print(f"    保留奇异值: {kept_sigma_sum:.2f} ({kept_sigma_ratio*100:.2f}%)")

        # 保存k结果
        k_data = {
            'k': k,
            'mean': float(ablated_avg),
            'std': float(ablated_std),
            'min': float(ablated_min),
            'max': float(ablated_max),
            'values': ablated_mas,
            'kept_sigma_sum': kept_sigma_sum,
            'kept_sigma_ratio': kept_sigma_ratio
        }

        with open(os.path.join(OUTPUT_DIR, f'keep_k{k}.json'), 'w') as f:
            json.dump(k_data, f, indent=2)

        results_by_k[str(k)] = k_data

        # 恢复原始权重
        with torch.no_grad():
            down_proj.weight.data = torch.tensor(W_original,
                                                 dtype=down_proj.weight.dtype,
                                                 device=down_proj.weight.device)

    # 3. 计算边际贡献
    print(f"\n{'='*60}")
    print("计算边际贡献")
    print("="*60)

    marginal_contributions = {}
    cumulative_ma = 0

    for i, k in enumerate(K_VALUES):
        ma = results_by_k[str(k)]['mean']
        if i == 0:
            marginal = ma
        else:
            marginal = ma - results_by_k[str(K_VALUES[i-1])]['mean']

        cumulative_pct = (ma / baseline_avg * 100) if baseline_avg != 0 else 0
        recovery_rate = (ma / baseline_avg) if baseline_avg != 0 else 0

        marginal_contributions[str(k)] = {
            'marginal_contribution': float(marginal),
            'cumulative_pct': float(cumulative_pct),
            'recovery_rate': float(recovery_rate)
        }

        print(f"  k={k}: 边际={marginal:+.2f}, 累积={cumulative_pct:.1f}%, 恢复率={recovery_rate:.3f}")

    # 4. 生成summary
    print(f"\n{'='*60}")
    print("生成Summary")
    print("="*60)

    # 找critical_k (首次达到baseline 95%的k)
    critical_k = None
    for k in K_VALUES:
        if results_by_k[str(k)]['mean'] >= baseline_avg * 0.95:
            critical_k = k
            break

    summary = {
        'model': MODEL_NAME,
        'layer': CRITICAL_LAYER,
        'date': datetime.now().isoformat(),
        'n_samples': NSAMPLES,
        'baseline_mean': float(baseline_avg),
        'k_values': K_VALUES,
        'results_by_k': results_by_k,
        'marginal_contributions': marginal_contributions,
        'critical_k': critical_k,
        'svd_info': {
            'sigma_top10': S[:10].tolist(),
            'sigma_ratio': sigma_ratio
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary已保存")
    print(f"  Baseline MA: {baseline_avg:.2f}")
    print(f"  σ₁/σ₂: {sigma_ratio:.2f}")
    print(f"  Critical k: {critical_k}")

    print(f"\n{'='*80}")
    print(f"✅ Qwen2.5-7B Exp6完成！")
    print(f"保存位置: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
