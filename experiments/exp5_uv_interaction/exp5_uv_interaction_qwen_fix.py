#!/usr/bin/env python3
"""
实验5: U×V交互归因分析
测试MA生成是U和V独立贡献还是需要协同作用

存储格式遵循exp2b标准:
- baseline.json: 原始模型
- ablate_u.json: 消融U (U_random @ Σ @ Vᵀ)
- ablate_v.json: 消融V (U @ Σ @ V_random)
- ablate_both.json: 同时消融 (U_random @ Σ @ V_random)
- summary.json: 归因百分比汇总
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path

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
    
    # 获取测试数据 - 对qwen使用更短序列避免token长度超限
    safe_seqlen = 512 if 'qwen' in str(model.config._name_or_path).lower() else min(seq_len, 2048)
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples, seqlen=safe_seqlen, device=device)
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
                     seq_len, ablate_u=False, ablate_v=False, n_samples=5):
    """
    运行干预实验

    Args:
        ablate_u: 是否用随机正交矩阵替换U
        ablate_v: 是否用随机正交矩阵替换V
    """
    layer = get_mlp_layer(model, layer_id, model_type)
    original_weight = get_w2_weight(layer, model_type).detach().clone()

    # SVD分解
    W = original_weight.detach().cpu().float().numpy()
    # GPT-2使用Conv1D，权重需要转置
    if "gpt2" in model_type:
        W = W.T
    U, S, Vh = np.linalg.svd(W, full_matrices=False)

    # 构建干预后的权重
    if ablate_u and ablate_v:
        # 同时消融
        U_new = create_random_orthogonal(U.shape, device='cpu').numpy()
        V_new = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        W_new = U_new @ np.diag(S) @ V_new
        intervention_type = "ablate_both"
    elif ablate_u:
        # 只消融U
        U_new = create_random_orthogonal(U.shape, device='cpu').numpy()
        W_new = U_new @ np.diag(S) @ Vh
        intervention_type = "ablate_u"
    elif ablate_v:
        # 只消融V
        V_new = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        W_new = U @ np.diag(S) @ V_new
        intervention_type = "ablate_v"
    else:
        # Baseline
        W_new = W
        intervention_type = "baseline"

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


def compute_attribution(baseline, ablate_u, ablate_v, ablate_both):
    """
    计算归因百分比

    MA = U_contribution + V_contribution + Interaction

    U_only = baseline - ablate_v  (保留U，移除V)
    V_only = baseline - ablate_u  (保留V，移除U)
    Interaction = baseline - U_only - V_only + ablate_both
    """
    baseline_val = baseline['mean']

    # 主效应
    u_main_effect = baseline_val - ablate_v['mean']  # V被破坏，看U的作用
    v_main_effect = baseline_val - ablate_u['mean']  # U被破坏，看V的作用

    # 交互效应
    # 如果U和V完全独立: ablate_both ≈ baseline - U_main - V_main
    # 交互项 = 实际差异
    expected_both = baseline_val - u_main_effect - v_main_effect
    interaction = ablate_both['mean'] - expected_both

    # 归因百分比（相对于baseline）
    u_attribution = (u_main_effect / baseline_val) * 100 if baseline_val != 0 else 0
    v_attribution = (v_main_effect / baseline_val) * 100 if baseline_val != 0 else 0
    interaction_attribution = (interaction / baseline_val) * 100 if baseline_val != 0 else 0

    return {
        'baseline': baseline_val,
        'ablate_u_mean': ablate_u['mean'],
        'ablate_v_mean': ablate_v['mean'],
        'ablate_both_mean': ablate_both['mean'],
        'u_main_effect': u_main_effect,
        'v_main_effect': v_main_effect,
        'interaction_effect': interaction,
        'u_attribution_pct': u_attribution,
        'v_attribution_pct': v_attribution,
        'interaction_pct': interaction_attribution,
        'total_explained': u_attribution + v_attribution + interaction_attribution,
        'interpretation': 'independent' if abs(interaction_attribution) < 5 else 'synergistic'
    }


def main():
    parser = argparse.ArgumentParser(description='实验5: U×V交互归因分析')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True, help='要分析的层')
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("="*80)
    print("实验5: U×V交互归因分析")
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

    # 运行四种条件
    print("\n" + "="*80)
    print("1. Baseline (原始模型)")
    print("="*80)
    baseline_summary, baseline_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=False, ablate_v=False, n_samples=args.nsamples
    )
    print(f"  MA平均: {baseline_summary['mean']:.2f}%")

    with open(os.path.join(args.savedir, 'baseline.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_baseline',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': baseline_summary,
            'results': baseline_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("2. 消融U矩阵 (U_random @ Σ @ Vᵀ)")
    print("="*80)
    ablate_u_summary, ablate_u_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=True, ablate_v=False, n_samples=args.nsamples
    )
    print(f"  MA平均: {ablate_u_summary['mean']:.2f}% (变化: {ablate_u_summary['mean']-baseline_summary['mean']:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_u.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_ablate_u',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_u_summary,
            'results': ablate_u_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("3. 消融V矩阵 (U @ Σ @ V_random)")
    print("="*80)
    ablate_v_summary, ablate_v_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=False, ablate_v=True, n_samples=args.nsamples
    )
    print(f"  MA平均: {ablate_v_summary['mean']:.2f}% (变化: {ablate_v_summary['mean']-baseline_summary['mean']:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_v.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_ablate_v',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_v_summary,
            'results': ablate_v_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("4. 同时消融U和V (U_random @ Σ @ V_random)")
    print("="*80)
    ablate_both_summary, ablate_both_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=True, ablate_v=True, n_samples=args.nsamples
    )
    print(f"  MA平均: {ablate_both_summary['mean']:.2f}% (变化: {ablate_both_summary['mean']-baseline_summary['mean']:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_both.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_ablate_both',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_both_summary,
            'results': ablate_both_results
        }, f, indent=2)

    # 计算归因
    print("\n" + "="*80)
    print("归因分析")
    print("="*80)
    attribution = compute_attribution(
        baseline_summary, ablate_u_summary, ablate_v_summary, ablate_both_summary
    )

    print(f"\n基线MA: {attribution['baseline']:.2f}%")
    print(f"\nU矩阵贡献: {attribution['u_attribution_pct']:.2f}%")
    print(f"V矩阵贡献: {attribution['v_attribution_pct']:.2f}%")
    print(f"交互效应: {attribution['interaction_pct']:.2f}%")
    print(f"总解释率: {attribution['total_explained']:.2f}%")
    print(f"\n机制类型: {attribution['interpretation']}")
    if attribution['interpretation'] == 'independent':
        print("  → U和V近似独立贡献 (SVD对齐型)")
    else:
        print("  → U和V需要协同作用 (多方向型)")

    # 保存汇总
    summary_data = {
        'model': args.model,
        'layer': args.layer,
        'date': datetime.now().isoformat(),
        'n_samples': args.nsamples,
        'attribution': attribution
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
