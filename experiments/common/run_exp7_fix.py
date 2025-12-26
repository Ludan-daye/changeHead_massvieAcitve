#!/usr/bin/env python3
"""
重新生成Exp7数据 - 修复OPT、Qwen、GPT-J
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from datetime import datetime

# 禁用代理
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import lib


def compute_ma(model, tokenizer, device, seq_len, nsamples, layer_id):
    """计算MA值（在指定层）"""
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples,
                                 seqlen=min(seq_len, 2048), device=device)
    if not isinstance(testseq_list, list):
        testseq_list = [testseq_list]

    all_top1 = []

    for testseq in testseq_list:
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
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                layers = model.model.layers
            elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
                layers = model.model.decoder.layers
            else:
                raise ValueError("Cannot find layers in model.model")
        elif hasattr(model, 'gpt_neox'):
            layers = model.gpt_neox.layers
        else:
            raise ValueError("Cannot find layers")

        # 使用指定层
        layer = layers[layer_id]
        handle = layer.register_forward_hook(make_hook(layer_id))

        with torch.no_grad():
            _ = model(testseq)

        handle.remove()

        if layer_id in activations:
            feat = activations[layer_id].numpy()
            if len(feat.shape) == 3:
                feat = feat.reshape(-1, feat.shape[-1])
            top1 = float(np.abs(feat).max())
            all_top1.append(top1)

    return all_top1


def create_random_orthogonal(shape, device='cpu'):
    """创建随机正交矩阵"""
    random_matrix = torch.randn(shape, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q


def get_mlp_layer(model, layer_id):
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
    raise ValueError(f"Cannot access layer {layer_id}")


def get_w2_weight(layer, model_name):
    """获取W2权重矩阵"""
    if hasattr(layer, 'mlp'):
        if hasattr(layer.mlp, 'down_proj'):
            return layer.mlp.down_proj.weight
        elif hasattr(layer.mlp, 'c_proj'):
            return layer.mlp.c_proj.weight
        elif hasattr(layer.mlp, 'fc_out'):
            return layer.mlp.fc_out.weight
        elif hasattr(layer.mlp, 'dense_4h_to_h'):
            return layer.mlp.dense_4h_to_h.weight
    if hasattr(layer, 'fc2'):
        return layer.fc2.weight
    raise ValueError(f"Cannot find W2 for model {model_name}")


def set_w2_weight(layer, model_name, new_weight):
    """设置W2权重矩阵"""
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


def run_intervention(model, tokenizer, device, layer_id, model_name,
                     seq_len, U, S, Vh, intervention_type, n_samples):
    """运行干预实验"""
    layer = get_mlp_layer(model, layer_id)
    original_weight = get_w2_weight(layer, model_name).detach().clone()

    if intervention_type == 'baseline':
        W_new = U @ np.diag(S) @ Vh

    elif intervention_type == 'ablate_direction':
        # 随机化方向，保留放大
        U_rand = create_random_orthogonal(U.shape, device='cpu').numpy()
        V_rand = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        W_new = U_rand @ np.diag(S) @ V_rand

    elif intervention_type == 'ablate_magnitude':
        # 均匀化放大，保留方向
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

    # 检查是否有NaN
    if np.isnan(W_new).any():
        print(f"  ⚠️ WARNING: W_new contains NaN values!")
        return [np.nan] * n_samples

    # 设置新权重
    if "gpt2" in model_name:
        W_tensor = torch.tensor(W_new.T, dtype=original_weight.dtype, device=original_weight.device)
    else:
        W_tensor = torch.tensor(W_new, dtype=original_weight.dtype, device=original_weight.device)

    set_w2_weight(layer, model_name, W_tensor)

    # 计算MA（在干预层）
    ma_values = compute_ma(model, tokenizer, device, seq_len, n_samples, layer_id)

    # 恢复原始权重
    set_w2_weight(layer, model_name, original_weight)

    return ma_values


def get_critical_layer(model_name):
    """获取关键层"""
    layers = {
        'gptj_6b': 0,
        'opt_7b': 3,
        'qwen2.5_7b': 3,
    }
    return layers.get(model_name, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    MODEL_NAME = args.model
    CRITICAL_LAYER = get_critical_layer(MODEL_NAME)
    NSAMPLES = 5
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, f'results/experiments/exp7/{MODEL_NAME}')

    print("=" * 80)
    print(f"Exp7: {MODEL_NAME} 方向与幅度归因分析")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print(f"\n加载模型: {MODEL_NAME}")

    class Args:
        model = MODEL_NAME

    model_args = Args()
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(model_args)
    model.eval()
    print(f"✓ 模型加载完成 (层数: {len(layers)})")

    # 获取权重并SVD分解
    layer = get_mlp_layer(model, CRITICAL_LAYER)
    W_original = get_w2_weight(layer, MODEL_NAME).data.cpu().float().numpy()

    if "gpt2" in MODEL_NAME:
        W_original = W_original.T

    print(f"\n权重shape: {W_original.shape}")
    print("SVD分解...")
    U, S, Vh = np.linalg.svd(W_original, full_matrices=False)
    print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")

    sigma_ratio = float(S[0] / S[1])
    print(f"σ₁/σ₂ = {sigma_ratio:.2f}")
    print(f"Top 10 奇异值: {S[:10].tolist()}")

    # 运行4种干预
    interventions = ['baseline', 'ablate_direction', 'ablate_magnitude', 'ablate_both']
    results = {}

    for intervention in interventions:
        print(f"\n{'='*60}")
        print(f"运行: {intervention}")
        print("="*60)

        ma_values = run_intervention(model, tokenizer, device, CRITICAL_LAYER,
                                     MODEL_NAME, seq_len, U, S, Vh,
                                     intervention, NSAMPLES)

        ma_mean = np.mean(ma_values)
        ma_std = np.std(ma_values)

        print(f"  MA值: {ma_values}")
        print(f"  平均: {ma_mean:.2f}, 标准差: {ma_std:.2f}")

        results[intervention] = {
            'experiment': f'exp8_{intervention}',
            'model': MODEL_NAME,
            'layer': CRITICAL_LAYER,
            'date': datetime.now().isoformat(),
            'n_samples': NSAMPLES,
            'summary': {
                'mean': float(ma_mean),
                'std': float(ma_std),
                'min': float(np.min(ma_values)),
                'max': float(np.max(ma_values)),
                'values': [float(v) for v in ma_values]
            },
            'results': {
                str(i): {
                    'mean': float(ma_values[i]),
                    'n_samples': 1,
                    'intervention': intervention
                } for i in range(len(ma_values))
            }
        }

        # 保存单个结果
        with open(os.path.join(OUTPUT_DIR, f'{intervention}.json'), 'w') as f:
            json.dump(results[intervention], f, indent=2)

    # 计算归因
    print(f"\n{'='*60}")
    print("计算归因")
    print("="*60)

    baseline = results['baseline']['summary']['mean']
    ablate_dir = results['ablate_direction']['summary']['mean']
    ablate_mag = results['ablate_magnitude']['summary']['mean']
    ablate_both = results['ablate_both']['summary']['mean']

    direction_effect = baseline - ablate_dir
    magnitude_effect = baseline - ablate_mag
    interaction_effect = baseline - ablate_both - direction_effect - magnitude_effect

    dir_pct = (direction_effect / baseline * 100) if baseline != 0 else 0
    mag_pct = (magnitude_effect / baseline * 100) if baseline != 0 else 0
    int_pct = (interaction_effect / baseline * 100) if baseline != 0 else 0
    total_explained = dir_pct + mag_pct + int_pct

    print(f"  Baseline: {baseline:.2f}")
    print(f"  Direction effect: {direction_effect:+.2f} ({dir_pct:+.1f}%)")
    print(f"  Magnitude effect: {magnitude_effect:+.2f} ({mag_pct:+.1f}%)")
    print(f"  Interaction: {interaction_effect:+.2f} ({int_pct:+.1f}%)")
    print(f"  Total explained: {total_explained:.1f}%")

    # 判断主导类型
    if abs(dir_pct) > 20:
        dir_dom = "high"
    elif abs(dir_pct) > 10:
        dir_dom = "medium"
    else:
        dir_dom = "low"

    if abs(mag_pct) > 20:
        mag_dom = "high"
    elif abs(mag_pct) > 10:
        mag_dom = "medium"
    else:
        mag_dom = "low"

    # 保存summary
    summary = {
        'model': MODEL_NAME,
        'layer': CRITICAL_LAYER,
        'date': datetime.now().isoformat(),
        'n_samples': NSAMPLES,
        'attribution': {
            'baseline': float(baseline),
            'ablate_direction_mean': float(ablate_dir),
            'ablate_magnitude_mean': float(ablate_mag),
            'ablate_both_mean': float(ablate_both),
            'direction_effect': float(direction_effect),
            'magnitude_effect': float(magnitude_effect),
            'interaction_effect': float(interaction_effect),
            'direction_attribution_pct': float(dir_pct),
            'magnitude_attribution_pct': float(mag_pct),
            'interaction_pct': float(int_pct),
            'total_explained': float(total_explained),
            'interpretation': {
                'direction_dominance': dir_dom,
                'magnitude_dominance': mag_dom
            }
        },
        'svd_info': {
            'sigma_stats': {
                'min': float(S.min()),
                'mean': float(S.mean()),
                'max': float(S.max())
            },
            'sigma_ratio': float(sigma_ratio),
            'sigma_top10': [float(s) for s in S[:10]]
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ {MODEL_NAME} Exp7完成！")
    print(f"保存位置: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
