#!/usr/bin/env python3
"""
实验4b: SVD向量与Massive Activation方向一致性测试
测试Layer 3的奇异向量是否与实际的massive activation对齐
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
from lib.model_utils import is_llama_model


def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2.5_7b')
    parser.add_argument('--layer', type=int, default=3, help='要分析的层')
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--savedir', type=str, default='results/models/qwen2.5_7b/exp4b')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    print("="*60)
    print(f"实验4b: Layer {args.layer} SVD与Massive Activation对齐分析")
    print(f"模型: {args.model}")
    print("="*60)
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    
    target_layer = layers[args.layer]
    
    # 1. 获取MLP down_proj的SVD
    print(f"\n[Step 1] 计算Layer {args.layer} MLP down_proj的SVD...")
    W = target_layer.mlp.down_proj.weight.data.cpu().float().numpy()
    print(f"  Weight shape: {W.shape}")
    
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    print(f"  Top 5 Singular Values: {S[:5].round(2)}")
    print(f"  σ₁/σ₂ = {S[0]/S[1]:.4f}")
    
    # 主奇异向量 (输出空间)
    u1 = U[:, 0]  # shape: [hidden_size]
    u2 = U[:, 1]
    u3 = U[:, 2]
    print(f"  u1 shape: {u1.shape}")
    
    # 2. 收集massive activation激活向量
    print(f"\n[Step 2] 收集Layer {args.layer}的激活向量...")
    
    # Enable feature capture
    mp.enable_qwen_custom_decoderlayer(target_layer, args.layer)
    
    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=min(seq_len, 2048), device=device)
    
    activation_vectors = []
    with torch.no_grad():
        for testseq in tqdm(testseq_list, desc="Collecting activations"):
            _ = model(testseq)
            
            if hasattr(target_layer, 'feat') and target_layer.feat is not None:
                feat = target_layer.feat  # [batch, seq, hidden]
                # 取绝对值最大的token的激活向量
                feat_abs = feat.abs()
                max_idx = feat_abs.sum(dim=-1).argmax()  # 找激活值总和最大的位置
                batch_idx = max_idx // feat.shape[1]
                seq_idx = max_idx % feat.shape[1]
                
                act_vec = feat[batch_idx, seq_idx, :].numpy()
                activation_vectors.append(act_vec)
    
    print(f"  收集到 {len(activation_vectors)} 个激活向量")
    
    # 3. 计算对齐
    print(f"\n[Step 3] 计算SVD向量与激活向量的对齐...")
    
    alignments_u1 = []
    alignments_u2 = []
    alignments_u3 = []
    
    for act_vec in activation_vectors:
        sim1 = abs(cosine_similarity(act_vec, u1))
        sim2 = abs(cosine_similarity(act_vec, u2))
        sim3 = abs(cosine_similarity(act_vec, u3))
        alignments_u1.append(sim1)
        alignments_u2.append(sim2)
        alignments_u3.append(sim3)
    
    avg_u1 = np.mean(alignments_u1)
    avg_u2 = np.mean(alignments_u2)
    avg_u3 = np.mean(alignments_u3)
    
    print(f"\n  平均余弦相似度:")
    print(f"    与 u1 (σ₁={S[0]:.2f}): {avg_u1:.4f}")
    print(f"    与 u2 (σ₂={S[1]:.2f}): {avg_u2:.4f}")
    print(f"    与 u3 (σ₃={S[2]:.2f}): {avg_u3:.4f}")
    
    # 4. 分析massive activation维度
    print(f"\n[Step 4] 分析Massive Activation维度...")
    
    # 找出激活最大的维度
    avg_act = np.mean([np.abs(v) for v in activation_vectors], axis=0)
    top_dims = np.argsort(avg_act)[::-1][:10]
    
    print(f"  Top 10 激活维度: {top_dims}")
    print(f"  对应平均值: {avg_act[top_dims].round(2)}")
    
    # u1在这些维度上的值
    print(f"\n  u1在Top 10维度上的值:")
    for dim in top_dims[:5]:
        print(f"    Dim {dim}: u1={u1[dim]:.4f}, avg_act={avg_act[dim]:.2f}")
    
    # 5. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 对齐分布
    ax1 = axes[0, 0]
    x = range(len(alignments_u1))
    ax1.bar([i-0.2 for i in x], alignments_u1, width=0.2, label=f'u1 (σ₁={S[0]:.1f})', color='red')
    ax1.bar([i for i in x], alignments_u2, width=0.2, label=f'u2 (σ₂={S[1]:.1f})', color='blue')
    ax1.bar([i+0.2 for i in x], alignments_u3, width=0.2, label=f'u3 (σ₃={S[2]:.1f})', color='green')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('|Cosine Similarity|')
    ax1.set_title(f'Layer {args.layer}: Activation Alignment with SVD Vectors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # (2) 平均对齐对比
    ax2 = axes[0, 1]
    bars = ax2.bar(['u1', 'u2', 'u3'], [avg_u1, avg_u2, avg_u3], color=['red', 'blue', 'green'])
    ax2.set_ylabel('Average |Cosine Similarity|')
    ax2.set_title('Average Alignment with Top-3 Singular Vectors')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [avg_u1, avg_u2, avg_u3]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')
    
    # (3) u1向量可视化
    ax3 = axes[1, 0]
    ax3.bar(range(len(u1)), np.abs(u1), width=1, color='steelblue', alpha=0.7)
    for dim in top_dims[:3]:
        ax3.axvline(x=dim, color='red', linestyle='--', alpha=0.5, label=f'Top Dim {dim}' if dim == top_dims[0] else '')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('|u1| value')
    ax3.set_title('First Singular Vector u1 Components')
    ax3.legend()
    
    # (4) 激活向量可视化
    ax4 = axes[1, 1]
    ax4.bar(range(len(avg_act)), avg_act, width=1, color='orange', alpha=0.7)
    for dim in top_dims[:3]:
        ax4.axvline(x=dim, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('Average |Activation|')
    ax4.set_title('Average Activation Vector')
    
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
        'top_activation_dims': top_dims[:10].tolist(),
        'u1_at_top_dims': [float(u1[d]) for d in top_dims[:10]],
        'alignments_u1': [float(x) for x in alignments_u1]
    }
    
    with open(os.path.join(args.savedir, f'layer{args.layer}_alignment.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    if avg_u1 > 0.5:
        print(f"\n✅ 强对齐: Layer {args.layer}的激活与u1高度对齐 (cos={avg_u1:.3f})")
        print("   → 证实MLP的主导奇异方向产生massive activation")
    elif avg_u1 > 0.2:
        print(f"\n⚠️ 中等对齐: Layer {args.layer}的激活与u1部分对齐 (cos={avg_u1:.3f})")
    else:
        print(f"\n❌ 弱对齐: Layer {args.layer}的激活与u1对齐较弱 (cos={avg_u1:.3f})")
        print("   → 可能存在其他机制")
    
    print(f"\n结果已保存至: {args.savedir}")

if __name__ == "__main__":
    main()
