#!/usr/bin/env python3
"""
实验4: MLP SVD对齐分析 - 通用版本
分析MLP输出权重的SVD结构，验证massive activation的数学来源
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib
from lib.model_utils import is_llama_model


def get_mlp_weights(layer, model_type):
    """获取MLP输出层权重"""
    if "qwen" in model_type or is_llama_model(model_type) or "deepseek" in model_type:
        # LLaMA/Qwen使用 SwiGLU: gate_proj, up_proj, down_proj
        # down_proj 是输出投影层
        if hasattr(layer.mlp, 'down_proj'):
            return layer.mlp.down_proj.weight.data.cpu().float().numpy()
    elif "opt" in model_type:
        # OPT使用 fc1, fc2
        if hasattr(layer, 'fc2'):
            return layer.fc2.weight.data.cpu().float().numpy()
    elif "gpt2" in model_type or "gptj" in model_type:
        # GPT-2/GPT-J使用 c_proj
        if hasattr(layer.mlp, 'c_proj'):
            return layer.mlp.c_proj.weight.data.cpu().float().numpy()
    elif "bloom" in model_type:
        # BLOOM使用 dense_4h_to_h
        if hasattr(layer.mlp, 'dense_4h_to_h'):
            return layer.mlp.dense_4h_to_h.weight.data.cpu().float().numpy()
    
    raise ValueError(f"Cannot find MLP output weights for model type: {model_type}")


def compute_svd(W, name=""):
    """计算SVD并返回结果"""
    print(f"  {name} shape: {W.shape}")
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    
    print(f"  Top 5 Singular Values: {S[:5].round(2)}")
    if len(S) > 1:
        print(f"  Ratio σ₁/σ₂: {S[0]/S[1]:.4f}")
        print(f"  Ratio σ₁/σ₃: {S[0]/S[2]:.4f}" if len(S) > 2 else "")
    
    return {'U': U, 'S': S, 'Vh': Vh, 'shape': W.shape}


def analyze_alignment(U, massive_dims, hidden_size):
    """分析SVD向量与massive activation维度的对齐"""
    # U[:, 0] 是第一主奇异向量，维度为 [hidden_size]
    u1 = U[:, 0]
    
    results = {}
    for dim in massive_dims:
        if dim < len(u1):
            # 该维度在u1中的值
            val = abs(u1[dim])
            # 该维度在u1中的排名
            rank = (np.abs(u1) >= val).sum()
            results[dim] = {'value': float(val), 'rank': int(rank)}
            print(f"    Dim {dim}: |u1[{dim}]| = {val:.6f}, Rank = {rank}/{hidden_size}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2.5_7b')
    parser.add_argument('--layers', type=str, default='0,1,2,3', help='要分析的层，逗号分隔')
    parser.add_argument('--savedir', type=str, default='results/models/qwen2.5_7b/exp4')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    layer_ids = [int(x) for x in args.layers.split(',')]
    
    print("="*60)
    print("实验4: MLP SVD对齐分析")
    print(f"模型: {args.model}")
    print(f"分析层: {layer_ids}")
    print("="*60)
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"模型层数: {len(layers)}, Hidden Size: {hidden_size}")
    
    # 根据实验1的结果，Qwen的massive activation维度是447和138
    massive_dims = [447, 138]  # 可以根据实验1结果调整
    
    results = {}
    
    for layer_id in layer_ids:
        if layer_id >= len(layers):
            print(f"Layer {layer_id} 超出范围，跳过")
            continue
            
        print(f"\n{'='*40}")
        print(f"Layer {layer_id} MLP SVD 分析")
        print(f"{'='*40}")
        
        layer = layers[layer_id]
        
        try:
            W = get_mlp_weights(layer, args.model)
            svd = compute_svd(W, "down_proj/fc2")
            
            print(f"\n  分析与Massive Activation维度的对齐:")
            alignment = analyze_alignment(svd['U'], massive_dims, hidden_size)
            
            results[layer_id] = {
                'singular_values': svd['S'][:20].tolist(),
                'ratio_s1_s2': float(svd['S'][0] / svd['S'][1]) if len(svd['S']) > 1 else None,
                'alignment': alignment
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[layer_id] = {'error': str(e)}
    
    # 保存结果
    with open(os.path.join(args.savedir, 'svd_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 奇异值分布
    ax1 = axes[0]
    for layer_id in layer_ids:
        if layer_id in results and 'singular_values' in results[layer_id]:
            sv = results[layer_id]['singular_values']
            ax1.plot(range(len(sv)), sv, marker='o', label=f'Layer {layer_id}')
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('MLP Output Layer Singular Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. σ₁/σ₂ 比值对比
    ax2 = axes[1]
    ratios = []
    labels = []
    for layer_id in layer_ids:
        if layer_id in results and 'ratio_s1_s2' in results[layer_id]:
            ratios.append(results[layer_id]['ratio_s1_s2'])
            labels.append(f'Layer {layer_id}')
    
    bars = ax2.bar(labels, ratios, color='steelblue')
    if ratios:
        max_idx = np.argmax(ratios)
        bars[max_idx].set_color('red')
    ax2.set_ylabel('σ₁/σ₂ Ratio')
    ax2.set_title('Singular Value Dominance (Higher = More Dominant Direction)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.savedir, 'svd_analysis.png'), dpi=150)
    plt.close()
    
    # 打印总结
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    print("\n关键发现:")
    for layer_id in layer_ids:
        if layer_id in results and 'ratio_s1_s2' in results[layer_id]:
            ratio = results[layer_id]['ratio_s1_s2']
            print(f"  Layer {layer_id}: σ₁/σ₂ = {ratio:.4f}")
            if ratio > 2:
                print(f"    → 存在主导奇异方向！")
    
    print(f"\n结果已保存至: {args.savedir}")

if __name__ == "__main__":
    main()
