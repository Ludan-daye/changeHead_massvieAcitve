#!/usr/bin/env python3
"""
实验4: Attention层 SVD分析
分析Attention输出权重的SVD结构
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib
from lib.model_utils import is_llama_model


def get_attention_weights(layer, model_type):
    """获取Attention输出投影层权重（只获取dense/o_proj，跳过大矩阵QKV）"""
    weights = {}
    
    if "bloom" in model_type:
        # BLOOM: self_attention.dense (输出投影)
        if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'dense'):
            weights['dense'] = layer.self_attention.dense.weight.data.cpu().float().numpy()
        # 跳过QKV，矩阵太大SVD太慢
    elif "qwen" in model_type or is_llama_model(model_type) or "mistral" in model_type:
        # LLaMA/Qwen/Mistral: self_attn.o_proj
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            weights['o_proj'] = layer.self_attn.o_proj.weight.data.cpu().float().numpy()
    elif "gpt2" in model_type:
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_proj'):
            weights['c_proj'] = layer.attn.c_proj.weight.data.cpu().float().numpy()
    
    return weights


def compute_svd(W, name=""):
    """计算SVD"""
    print(f"  {name} shape: {W.shape}")
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    
    print(f"  Top 5 Singular Values: {S[:5].round(2)}")
    if len(S) > 1:
        print(f"  σ₁/σ₂ = {S[0]/S[1]:.4f}")
    
    return {'U': U, 'S': S, 'Vh': Vh, 'shape': W.shape}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bloom_7b1')
    parser.add_argument('--layers', type=str, default='26,27,28,29', help='要分析的层')
    parser.add_argument('--savedir', type=str, default='results/models/bloom_7b1/exp4')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    layer_ids = [int(x) for x in args.layers.split(',')]
    
    print("="*60)
    print("实验4: Attention SVD分析")
    print(f"模型: {args.model}")
    print(f"分析层: {layer_ids}")
    print("="*60)
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"模型层数: {len(layers)}, Hidden Size: {hidden_size}")
    
    results = {}
    
    for layer_id in layer_ids:
        if layer_id >= len(layers):
            print(f"Layer {layer_id} 超出范围，跳过")
            continue
            
        print(f"\n{'='*40}")
        print(f"Layer {layer_id} Attention SVD 分析")
        print(f"{'='*40}")
        
        layer = layers[layer_id]
        weights = get_attention_weights(layer, args.model)
        
        layer_results = {}
        for name, W in weights.items():
            print(f"\n  分析 {name}:")
            svd = compute_svd(W, name)
            layer_results[name] = {
                'singular_values': svd['S'][:20].tolist(),
                'ratio_s1_s2': float(svd['S'][0] / svd['S'][1]) if len(svd['S']) > 1 else None,
                'shape': list(svd['shape'])
            }
        
        results[layer_id] = layer_results
    
    # 保存结果
    with open(os.path.join(args.savedir, 'attention_svd.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. dense层奇异值分布
    ax1 = axes[0]
    for layer_id in layer_ids:
        if layer_id in results and 'dense' in results[layer_id]:
            sv = results[layer_id]['dense']['singular_values']
            ax1.plot(range(len(sv)), sv, marker='o', label=f'Layer {layer_id}')
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Attention Dense Layer Singular Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. σ₁/σ₂ 比值
    ax2 = axes[1]
    ratios = []
    labels = []
    for layer_id in layer_ids:
        if layer_id in results and 'dense' in results[layer_id]:
            ratios.append(results[layer_id]['dense']['ratio_s1_s2'])
            labels.append(f'Layer {layer_id}')
    
    if ratios:
        bars = ax2.bar(labels, ratios, color='steelblue')
        max_idx = np.argmax(ratios)
        bars[max_idx].set_color('red')
    ax2.set_ylabel('σ₁/σ₂ Ratio')
    ax2.set_title('Singular Value Dominance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.savedir, 'attention_svd.png'), dpi=150)
    plt.close()
    
    # 总结
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    print("\n关键发现 (dense层):")
    for layer_id in layer_ids:
        if layer_id in results and 'dense' in results[layer_id]:
            ratio = results[layer_id]['dense']['ratio_s1_s2']
            print(f"  Layer {layer_id}: σ₁/σ₂ = {ratio:.4f}")
            if ratio > 2:
                print(f"    → 存在主导奇异方向！")
    
    print(f"\n结果已保存至: {args.savedir}")

if __name__ == "__main__":
    main()
