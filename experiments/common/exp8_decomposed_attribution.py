#!/usr/bin/env python3
"""
实验8: 分解归因分析
将W₂的作用分解为"方向"(U×V)和"放大"(Σ)两部分

数学分解:
  W₂ @ h₂ = (U @ Σ @ Vᵀ) @ h₂
           = U @ (Σ @ (Vᵀ @ h₂))
  阶段1: α = Vᵀ @ h₂       (输入投影)
  阶段2: β = Σ @ α         (奇异值放大)
  阶段3: output = U @ β    (输出映射)

存储格式遵循exp2b标准:
- baseline.json: 原始模型
- ablate_direction.json: 随机化方向，保留放大
- ablate_magnitude.json: 均匀化放大，保留方向
- ablate_both.json: 同时随机化
- summary.json: 归因百分比
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

def create_random_orthogonal(shape, device='cpu'):
    """创建随机正交矩阵"""
    random_matrix = torch.randn(shape, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q


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



def run_intervention(model, tokenizer, device, layer_id, model_type,
                     seq_len, U, S, Vh, intervention_type, n_samples=5):
    """
    运行分解干预实验

    intervention_type:
    - 'baseline': 原始 W = U @ Σ @ Vᵀ
    - 'ablate_direction': 随机化方向 W = U_rand @ Σ @ V_rand
    - 'ablate_magnitude': 均匀化放大 W = U @ ones @ Vᵀ
    - 'ablate_both': 同时破坏 W = U_rand @ ones @ V_rand
    """
    layer = get_mlp_layer(model, layer_id, model_type)
    original_weight = get_w2_weight(layer, model_type).detach().clone()

    if intervention_type == 'baseline':
        W_new = U @ np.diag(S) @ Vh

    elif intervention_type == 'ablate_direction':
        # 随机化方向，保留放大
        U_rand = create_random_orthogonal(U.shape, device='cpu').numpy()
        V_rand = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        W_new = U_rand @ np.diag(S) @ V_rand

    elif intervention_type == 'ablate_magnitude':
        # 均匀化放大，保留方向
        # 用所有奇异值的平均值
        S_uniform = np.ones_like(S) * S.mean()
        W_new = U @ np.diag(S_uniform) @ Vh

    elif intervention_type == 'ablate_both':
        # 同时破坏
        U_rand = create_random_orthogonal(U.shape, device='cpu').numpy()
        V_rand = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        S_uniform = np.ones_like(S) * S.mean()
        W_new = U_rand @ np.diag(S_uniform) @ V_rand

    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

    # 设置新权重
    # GPT-2需要转置回Conv1D格式
    if "gpt2" in model_type:
        W_new = W_new.T
    new_weight_tensor = torch.from_numpy(W_new).to(device).to(original_weight.dtype)
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
            "intervention": intervention_type
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
        "values": values
    }

    return summary, results


def compute_attribution(baseline, ablate_dir, ablate_mag, ablate_both):
    """
    计算分解归因

    方向贡献 = baseline - ablate_direction
    放大贡献 = baseline - ablate_magnitude
    交互 = baseline - direction - magnitude + ablate_both
    """
    baseline_val = baseline['mean']

    direction_effect = baseline_val - ablate_dir['mean']
    magnitude_effect = baseline_val - ablate_mag['mean']

    expected_both = baseline_val - direction_effect - magnitude_effect
    interaction = ablate_both['mean'] - expected_both

    # 归因百分比
    dir_pct = (direction_effect / baseline_val * 100) if baseline_val != 0 else 0
    mag_pct = (magnitude_effect / baseline_val * 100) if baseline_val != 0 else 0
    interaction_pct = (interaction / baseline_val * 100) if baseline_val != 0 else 0

    return {
        'baseline': baseline_val,
        'ablate_direction_mean': ablate_dir['mean'],
        'ablate_magnitude_mean': ablate_mag['mean'],
        'ablate_both_mean': ablate_both['mean'],
        'direction_effect': direction_effect,
        'magnitude_effect': magnitude_effect,
        'interaction_effect': interaction,
        'direction_attribution_pct': dir_pct,
        'magnitude_attribution_pct': mag_pct,
        'interaction_pct': interaction_pct,
        'total_explained': dir_pct + mag_pct + interaction_pct,
        'interpretation': {
            'direction_dominance': 'high' if dir_pct > 50 else 'medium' if dir_pct > 20 else 'low',
            'magnitude_dominance': 'high' if mag_pct > 50 else 'medium' if mag_pct > 20 else 'low'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='实验8: 分解归因分析')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True, help='要分析的层')
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("="*80)
    print("实验8: 分解归因分析 (方向 vs 放大)")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"关键层: {args.layer}")
    print(f"样本数: {args.nsamples}")
    print(f"保存目录: {args.savedir}")
    print("="*80)

    # 加载模型
    print("\n正在加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"✓ 模型加载完成")

    # SVD分解
    print("\n正在进行SVD分解...")
    layer = get_mlp_layer(model, args.layer, args.model)
    original_weight = get_w2_weight(layer, args.model)
    W = original_weight.detach().cpu().float().numpy()
    # GPT-2使用Conv1D，权重需要转置
    if "gpt2" in args.model:
        W = W.T
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    print(f"✓ SVD完成")
    print(f"  奇异值统计: min={S.min():.4f}, mean={S.mean():.4f}, max={S.max():.4f}")
    print(f"  σ₁/σ₂ = {S[0]/S[1]:.4f}")

    # 四种条件
    print("\n" + "="*80)
    print("1. Baseline (原始模型)")
    print("="*80)
    baseline_summary, baseline_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        U, S, Vh, 'baseline', n_samples=args.nsamples
    )
    print(f"  MA平均: {baseline_summary['mean']:.2f}%")

    with open(os.path.join(args.savedir, 'baseline.json'), 'w') as f:
        json.dump({
            'experiment': 'exp8_decomposed_baseline',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': baseline_summary,
            'results': baseline_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("2. 消融方向 (U_random @ Σ @ V_random)")
    print("="*80)
    ablate_dir_summary, ablate_dir_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        U, S, Vh, 'ablate_direction', n_samples=args.nsamples
    )
    change = ablate_dir_summary['mean'] - baseline_summary['mean']
    print(f"  MA平均: {ablate_dir_summary['mean']:.2f}% (变化: {change:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_direction.json'), 'w') as f:
        json.dump({
            'experiment': 'exp8_ablate_direction',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_dir_summary,
            'results': ablate_dir_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("3. 消融放大 (U @ Σ_uniform @ Vᵀ)")
    print("="*80)
    ablate_mag_summary, ablate_mag_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        U, S, Vh, 'ablate_magnitude', n_samples=args.nsamples
    )
    change = ablate_mag_summary['mean'] - baseline_summary['mean']
    print(f"  MA平均: {ablate_mag_summary['mean']:.2f}% (变化: {change:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_magnitude.json'), 'w') as f:
        json.dump({
            'experiment': 'exp8_ablate_magnitude',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_mag_summary,
            'results': ablate_mag_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("4. 同时消融 (U_random @ Σ_uniform @ V_random)")
    print("="*80)
    ablate_both_summary, ablate_both_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        U, S, Vh, 'ablate_both', n_samples=args.nsamples
    )
    change = ablate_both_summary['mean'] - baseline_summary['mean']
    print(f"  MA平均: {ablate_both_summary['mean']:.2f}% (变化: {change:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_both.json'), 'w') as f:
        json.dump({
            'experiment': 'exp8_ablate_both',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_both_summary,
            'results': ablate_both_results
        }, f, indent=2)

    # 计算归因
    print("\n" + "="*80)
    print("分解归因分析")
    print("="*80)
    attribution = compute_attribution(
        baseline_summary, ablate_dir_summary,
        ablate_mag_summary, ablate_both_summary
    )

    print(f"\n基线MA: {attribution['baseline']:.2f}%")
    print(f"\n方向(U×V)贡献: {attribution['direction_attribution_pct']:.2f}%")
    print(f"  重要性: {attribution['interpretation']['direction_dominance']}")
    print(f"\n放大(Σ)贡献: {attribution['magnitude_attribution_pct']:.2f}%")
    print(f"  重要性: {attribution['interpretation']['magnitude_dominance']}")
    print(f"\n交互效应: {attribution['interaction_pct']:.2f}%")
    print(f"总解释率: {attribution['total_explained']:.2f}%")

    # 解释
    print("\n机制解释:")
    if attribution['direction_attribution_pct'] > 70:
        print("  → MA主要由方向决定，放大效应次要")
        print("  → 关键在于U和V的特定结构")
    elif attribution['magnitude_attribution_pct'] > 70:
        print("  → MA主要由奇异值放大决定，方向次要")
        print("  → 关键在于Σ的数值大小")
    else:
        print("  → MA需要方向和放大的协同")
        print("  → 两者缺一不可")

    # 保存汇总
    summary_data = {
        'model': args.model,
        'layer': args.layer,
        'date': datetime.now().isoformat(),
        'n_samples': args.nsamples,
        'attribution': attribution,
        'svd_info': {
            'sigma_stats': {
                'min': float(S.min()),
                'mean': float(S.mean()),
                'max': float(S.max())
            },
            'sigma_ratio': float(S[0] / S[1]),
            'sigma_top10': S[:10].tolist()
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
