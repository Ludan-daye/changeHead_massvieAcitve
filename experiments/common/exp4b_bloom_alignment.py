#!/usr/bin/env python3
"""
实验4b: BLOOM Attention Dense层 SVD与Massive Activation对齐分析
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
import monkey_patch as mp


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bloom_7b1')
    parser.add_argument('--layer', type=int, default=28)
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--savedir', type=str, default='results/models/bloom_7b1/exp4b')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    print("="*60)
    print(f"BLOOM Layer {args.layer} Attention SVD对齐分析")
    print("="*60)
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    
    target_layer = layers[args.layer]
    
    # 1. 获取Attention dense层的SVD
    print(f"\n[Step 1] 计算Layer {args.layer} Attention dense的SVD...")
    W = target_layer.self_attention.dense.weight.data.cpu().float().numpy()
    print(f"  Weight shape: {W.shape}")
    
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    print(f"  Top 5 Singular Values: {S[:5].round(2)}")
    print(f"  σ₁/σ₂ = {S[0]/S[1]:.4f}")
    
    u1, u2, u3 = U[:, 0], U[:, 1], U[:, 2]
    
    # 2. 收集激活向量
    print(f"\n[Step 2] 收集Layer {args.layer}的激活向量...")
    
    mp.enable_bloom_custom_block(target_layer, args.layer)
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=min(seq_len, 2048), device=device)
    
    activation_vectors = []
    with torch.no_grad():
        for testseq in tqdm(testseq_list, desc="Collecting"):
            _ = model(testseq)
            if hasattr(target_layer, 'feat') and target_layer.feat is not None:
                feat = target_layer.feat
                feat_abs = feat.abs()
                max_idx = feat_abs.sum(dim=-1).argmax()
                batch_idx = max_idx // feat.shape[1]
                seq_idx = max_idx % feat.shape[1]
                act_vec = feat[batch_idx, seq_idx, :].numpy()
                activation_vectors.append(act_vec)
    
    print(f"  收集到 {len(activation_vectors)} 个激活向量")
    
    # 3. 计算对齐
    print(f"\n[Step 3] 计算对齐...")
    
    alignments_u1 = [abs(cosine_similarity(v, u1)) for v in activation_vectors]
    alignments_u2 = [abs(cosine_similarity(v, u2)) for v in activation_vectors]
    alignments_u3 = [abs(cosine_similarity(v, u3)) for v in activation_vectors]
    
    avg_u1 = np.mean(alignments_u1)
    avg_u2 = np.mean(alignments_u2)
    avg_u3 = np.mean(alignments_u3)
    
    print(f"\n  平均余弦相似度:")
    print(f"    与 u1 (σ₁={S[0]:.2f}): {avg_u1:.4f}")
    print(f"    与 u2 (σ₂={S[1]:.2f}): {avg_u2:.4f}")
    print(f"    与 u3 (σ₃={S[2]:.2f}): {avg_u3:.4f}")
    
    # 4. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    x = range(len(alignments_u1))
    ax1.bar([i-0.2 for i in x], alignments_u1, width=0.2, label=f'u1', color='red')
    ax1.bar([i for i in x], alignments_u2, width=0.2, label=f'u2', color='blue')
    ax1.bar([i+0.2 for i in x], alignments_u3, width=0.2, label=f'u3', color='green')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('|Cosine Similarity|')
    ax1.set_title(f'BLOOM Layer {args.layer}: Activation vs SVD Vectors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    bars = ax2.bar(['u1', 'u2', 'u3'], [avg_u1, avg_u2, avg_u3], color=['red', 'blue', 'green'])
    ax2.set_ylabel('Average |Cosine Similarity|')
    ax2.set_title('Average Alignment')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [avg_u1, avg_u2, avg_u3]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.savedir, f'layer{args.layer}_alignment.png'), dpi=150)
    plt.close()
    
    # 保存结果
    results = {
        'layer': args.layer,
        'singular_values': S[:10].tolist(),
        'ratio_s1_s2': float(S[0]/S[1]),
        'avg_alignment_u1': float(avg_u1),
        'avg_alignment_u2': float(avg_u2),
        'avg_alignment_u3': float(avg_u3),
    }
    
    with open(os.path.join(args.savedir, f'layer{args.layer}_alignment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    if avg_u1 > 0.5:
        print(f"\n✅ 强对齐 (cos={avg_u1:.3f})")
    elif avg_u1 > 0.2:
        print(f"\n⚠️ 中等对齐 (cos={avg_u1:.3f})")
    else:
        print(f"\n❌ 弱对齐 (cos={avg_u1:.3f})")
        print("   → BLOOM的MA可能由其他机制产生（非SVD主导方向）")
    
    print(f"\n结果已保存至: {args.savedir}")

if __name__ == "__main__":
    main()
