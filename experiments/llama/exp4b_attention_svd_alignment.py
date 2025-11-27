#!/usr/bin/env python3
"""
Experiment 4B: Attention Head SVD Alignment Analysis
实验四B：注意力头SVD对齐分析

针对性研究：
- 专注于Layer 3（大规模激活的起始层）
- 分析attention head的权重矩阵（Q, K, V, O projection）
- 对这些矩阵的右奇异向量与激活方向进行对齐分析

研究问题：
Layer 3的attention head权重矩阵的右奇异向量，是否与该层产生的巨量激活方向一致？
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp
from lib.model_utils import is_llama_model


def compute_attention_svd(layer, layer_id, model_config):
    """
    对attention层的权重矩阵进行SVD分解
    返回：各个权重矩阵的SVD结果
    """
    print(f"\n{'='*80}")
    print(f"Computing SVD for Layer {layer_id} Attention Weights")
    print(f"{'='*80}")
    
    svd_results = {}
    
    # LLaMA的attention结构
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        
        # 1. Q projection: [hidden_dim, hidden_dim]
        if hasattr(attn, 'q_proj'):
            W_q = attn.q_proj.weight.data.cpu().float().numpy()
            print(f"Q projection shape: {W_q.shape}")
            U_q, S_q, Vh_q = np.linalg.svd(W_q, full_matrices=False)
            svd_results['q_proj'] = {
                'U': U_q,
                'S': S_q,
                'Vh': Vh_q,
                'shape': W_q.shape,
                'matrix': W_q
            }
            print(f"  Top 5 singular values: {S_q[:5]}")
            print(f"  σ₁/σ₂ ratio: {S_q[0]/S_q[1]:.4f}")
        
        # 2. K projection: [hidden_dim, hidden_dim]
        if hasattr(attn, 'k_proj'):
            W_k = attn.k_proj.weight.data.cpu().float().numpy()
            print(f"\nK projection shape: {W_k.shape}")
            U_k, S_k, Vh_k = np.linalg.svd(W_k, full_matrices=False)
            svd_results['k_proj'] = {
                'U': U_k,
                'S': S_k,
                'Vh': Vh_k,
                'shape': W_k.shape,
                'matrix': W_k
            }
            print(f"  Top 5 singular values: {S_k[:5]}")
            print(f"  σ₁/σ₂ ratio: {S_k[0]/S_k[1]:.4f}")
        
        # 3. V projection: [hidden_dim, hidden_dim]
        if hasattr(attn, 'v_proj'):
            W_v = attn.v_proj.weight.data.cpu().float().numpy()
            print(f"\nV projection shape: {W_v.shape}")
            U_v, S_v, Vh_v = np.linalg.svd(W_v, full_matrices=False)
            svd_results['v_proj'] = {
                'U': U_v,
                'S': S_v,
                'Vh': Vh_v,
                'shape': W_v.shape,
                'matrix': W_v
            }
            print(f"  Top 5 singular values: {S_v[:5]}")
            print(f"  σ₁/σ₂ ratio: {S_v[0]/S_v[1]:.4f}")
        
        # 4. O projection (output): [hidden_dim, hidden_dim]
        if hasattr(attn, 'o_proj'):
            W_o = attn.o_proj.weight.data.cpu().float().numpy()
            print(f"\nO projection shape: {W_o.shape}")
            U_o, S_o, Vh_o = np.linalg.svd(W_o, full_matrices=False)
            svd_results['o_proj'] = {
                'U': U_o,
                'S': S_o,
                'Vh': Vh_o,
                'shape': W_o.shape,
                'matrix': W_o
            }
            print(f"  Top 5 singular values: {S_o[:5]}")
            print(f"  σ₁/σ₂ ratio: {S_o[0]/S_o[1]:.4f}")
    
    return svd_results


def collect_layer3_activations(args, target_layer=3):
    """
    收集Layer 3的激活值及其方向
    """
    print("\n" + "="*80)
    print(f"COLLECTING LAYER {target_layer} ACTIVATIONS")
    print("="*80)
    
    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture for target layer
    if is_llama_model(args.model):
        mp.enable_llama_custom_decoderlayer(layers[target_layer], target_layer)
    elif "opt" in args.model:
        mp.enable_opt_custom_decoderlayer(layers[target_layer], target_layer)
    elif "gpt2" in args.model:
        mp.enable_gpt2_custom_block(layers[target_layer], target_layer)

    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Storage
    activation_data = {
        'input_vectors': [],      # 输入到Layer 3的向量
        'output_vectors': [],     # Layer 3的输出向量
        'max_activations': [],    # 最大激活值
        'max_positions': [],      # 最大激活的位置
        'max_output_vectors': []  # 产生最大激活的输出向量
    }

    print(f"\nProcessing {len(testseq_list)} samples...")

    # Hook to capture input
    input_cache = []
    def input_hook(module, input, output):
        # input[0] is the hidden states
        input_cache.append(input[0].detach().cpu().float().numpy())
        return None

    # Register input hook
    handle = layers[target_layer].register_forward_hook(input_hook)

    # Process samples
    with torch.no_grad():
        for idx, testseq in enumerate(tqdm(testseq_list, desc=f"Layer {target_layer}")):
            input_cache.clear()
            _ = model(testseq)

            layer = layers[target_layer]
            if not hasattr(layer, 'feat') or layer.feat is None:
                continue

            # Output (feat)
            feat = layer.feat.cpu().float().numpy()
            if len(feat.shape) == 3:
                feat = feat[0]  # [seq_len, hidden_dim]
            
            # Input
            if input_cache:
                input_feat = input_cache[0]
                if len(input_feat.shape) == 3:
                    input_feat = input_feat[0]  # [seq_len, hidden_dim]
            else:
                input_feat = None
            
            # 找到最大激活值及其位置
            feat_abs = np.abs(feat)
            max_val = np.max(feat_abs)
            max_pos = np.unravel_index(np.argmax(feat_abs), feat_abs.shape)
            token_idx, dim_idx = max_pos
            
            # 保存数据
            activation_data['output_vectors'].append(feat)
            activation_data['max_activations'].append(max_val)
            activation_data['max_positions'].append((token_idx, dim_idx))
            activation_data['max_output_vectors'].append(feat[token_idx, :])
            
            if input_feat is not None:
                activation_data['input_vectors'].append(input_feat)

    # Clean up
    handle.remove()

    return activation_data, model, layers


def analyze_attention_alignment(activation_data, svd_results, target_layer=3):
    """
    分析attention权重的SVD方向与激活方向的对齐程度
    """
    print("\n" + "="*80)
    print(f"ANALYZING ATTENTION SVD ALIGNMENT FOR LAYER {target_layer}")
    print("="*80)
    
    # 获取输出激活向量
    max_output_vectors = np.array(activation_data['max_output_vectors'])  # [n_samples, hidden_dim]
    
    # 计算平均激活方向
    mean_output = np.mean(max_output_vectors, axis=0)  # [hidden_dim]
    mean_output_norm = mean_output / (np.linalg.norm(mean_output) + 1e-8)
    
    print(f"\nMean output activation norm: {np.linalg.norm(mean_output):.2f}")
    print(f"Max activation value (avg): {np.mean(activation_data['max_activations']):.2f}")
    
    # 对每个权重矩阵分析
    alignment_results = {}
    
    for weight_name, svd_data in svd_results.items():
        print(f"\n{'-'*80}")
        print(f"Analyzing {weight_name}")
        print(f"{'-'*80}")
        
        Vh = svd_data['Vh']  # [k, input_dim]，每一行是一个右奇异向量
        S = svd_data['S']    # 奇异值
        
        # 计算前10个右奇异向量与平均输出方向的余弦相似度
        k = min(20, Vh.shape[0])
        alignments = []
        
        for i in range(k):
            right_singular_vec = Vh[i, :]
            
            # 确保维度匹配
            if right_singular_vec.shape[0] == mean_output_norm.shape[0]:
                # 计算余弦相似度
                cosine_sim = np.dot(right_singular_vec, mean_output_norm)
                alignments.append({
                    'component': i,
                    'singular_value': float(S[i]),
                    'cosine_similarity': float(cosine_sim),
                    'abs_cosine_similarity': float(np.abs(cosine_sim))
                })
        
        # 排序并显示top 5
        sorted_alignments = sorted(alignments, key=lambda x: x['abs_cosine_similarity'], reverse=True)
        
        print(f"\nTop 5 aligned components for {weight_name}:")
        print(f"{'Rank':<6} {'Component':<12} {'Singular Val':<15} {'Cosine Sim':<15} {'|Cosine|':<15}")
        print("-"*80)
        for rank, align in enumerate(sorted_alignments[:5], 1):
            print(f"{rank:<6} {align['component']:<12} {align['singular_value']:<15.4f} "
                  f"{align['cosine_similarity']:<15.4f} {align['abs_cosine_similarity']:<15.4f}")
        
        alignment_results[weight_name] = {
            'all_alignments': alignments,
            'top_alignment': sorted_alignments[0] if sorted_alignments else None,
            'mean_abs_alignment': np.mean([a['abs_cosine_similarity'] for a in alignments])
        }
    
    return alignment_results


def generate_visualizations(svd_results, alignment_results, savedir, target_layer=3):
    """生成可视化"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(savedir, exist_ok=True)
    
    # 1. 奇异值谱图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    weight_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    for idx, weight_type in enumerate(weight_types):
        ax = axes[idx // 2, idx % 2]
        
        if weight_type in svd_results:
            S = svd_results[weight_type]['S']
            ax.plot(range(len(S[:50])), S[:50], marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Singular Value Index', fontsize=12)
            ax.set_ylabel('Singular Value', fontsize=12)
            ax.set_title(f'Layer {target_layer} {weight_type} - Singular Value Spectrum', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'layer{target_layer}_singular_values.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: layer{target_layer}_singular_values.png")
    
    # 2. 对齐度柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weight_names = []
    max_alignments = []
    mean_alignments = []
    
    for weight_type in weight_types:
        if weight_type in alignment_results:
            weight_names.append(weight_type)
            max_alignments.append(alignment_results[weight_type]['top_alignment']['abs_cosine_similarity'])
            mean_alignments.append(alignment_results[weight_type]['mean_abs_alignment'])
    
    x = np.arange(len(weight_names))
    width = 0.35
    
    ax.bar(x - width/2, max_alignments, width, label='Max Alignment', alpha=0.8)
    ax.bar(x + width/2, mean_alignments, width, label='Mean Alignment (Top 20)', alpha=0.8)
    
    ax.set_xlabel('Weight Matrix', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(f'Layer {target_layer} Attention Weights - SVD Alignment with Activation Direction',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(weight_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'layer{target_layer}_alignment_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: layer{target_layer}_alignment_comparison.png")


def generate_summary_report(svd_results, alignment_results, savedir, target_layer=3):
    """生成总结报告"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    lines = []
    lines.append("="*80)
    lines.append(f"EXPERIMENT 4B: LAYER {target_layer} ATTENTION SVD ALIGNMENT - SUMMARY")
    lines.append("="*80)
    lines.append("\nRESEARCH QUESTION:")
    lines.append(f"  Do the right singular vectors of Layer {target_layer} attention weight matrices")
    lines.append("  align with the direction of massive activations?")
    lines.append("\n" + "="*80)
    lines.append("SINGULAR VALUE ANALYSIS")
    lines.append("="*80)
    
    for weight_name, svd_data in svd_results.items():
        S = svd_data['S']
        lines.append(f"\n{weight_name.upper()}:")
        lines.append(f"  σ₁ (largest): {S[0]:.4f}")
        lines.append(f"  σ₂: {S[1]:.4f}")
        lines.append(f"  σ₃: {S[2]:.4f}")
        lines.append(f"  σ₁/σ₂ ratio: {S[0]/S[1]:.4f}×")
        
        if S[0]/S[1] > 2.0:
            lines.append(f"  ✓ Dominant singular direction detected")
        else:
            lines.append(f"  ⚠ No dominant singular direction")
    
    lines.append("\n" + "="*80)
    lines.append("ALIGNMENT ANALYSIS")
    lines.append("="*80)
    
    for weight_name, align_data in alignment_results.items():
        top = align_data['top_alignment']
        mean_align = align_data['mean_abs_alignment']
        
        lines.append(f"\n{weight_name.upper()}:")
        lines.append(f"  Best alignment:")
        lines.append(f"    Component: {top['component']}")
        lines.append(f"    Cosine similarity: {top['cosine_similarity']:.4f}")
        lines.append(f"    |Cosine similarity|: {top['abs_cosine_similarity']:.4f}")
        lines.append(f"    Singular value: {top['singular_value']:.4f}")
        lines.append(f"  Mean alignment (top 20 components): {mean_align:.4f}")
        
        if top['abs_cosine_similarity'] > 0.5:
            lines.append(f"  ✅ STRONG ALIGNMENT")
        elif top['abs_cosine_similarity'] > 0.3:
            lines.append(f"  ⚠️ MODERATE ALIGNMENT")
        else:
            lines.append(f"  ❌ WEAK ALIGNMENT")
    
    lines.append("\n" + "="*80)
    lines.append("CONCLUSION")
    lines.append("="*80)
    
    # 判断整体对齐情况
    max_alignment = max([align_data['top_alignment']['abs_cosine_similarity'] 
                        for align_data in alignment_results.values()])
    
    if max_alignment > 0.5:
        lines.append("\n✅ STRONG ALIGNMENT DETECTED")
        lines.append(f"  Maximum alignment: {max_alignment:.4f}")
        lines.append(f"  Layer {target_layer} attention weights show strong alignment with")
        lines.append("  massive activation directions, similar to GPT-2's mechanism.")
    elif max_alignment > 0.3:
        lines.append("\n⚠️ MODERATE ALIGNMENT")
        lines.append(f"  Maximum alignment: {max_alignment:.4f}")
        lines.append("  Some alignment detected, but not as strong as GPT-2.")
    else:
        lines.append("\n❌ WEAK ALIGNMENT")
        lines.append(f"  Maximum alignment: {max_alignment:.4f}")
        lines.append(f"  Layer {target_layer} attention weights do NOT align with activation directions.")
        lines.append("  LLaMA's massive activation mechanism differs from GPT-2.")
    
    lines.append("\n" + "="*80)
    
    summary_text = "\n".join(lines)
    print(summary_text)
    
    with open(os.path.join(savedir, f'LAYER{target_layer}_ATTENTION_SVD_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n✅ Summary saved!")


def main():
    parser = argparse.ArgumentParser(description='Experiment 4B: Attention Head SVD Alignment')
    parser.add_argument('--model', type=str, default='llama2_13b')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target_layer', type=int, default=3, 
                        help='Target layer to analyze (default: 3)')
    parser.add_argument('--savedir', type=str, default='results/exp4b_llama2_13b/')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "="*80)
    print(f"EXPERIMENT 4B: LAYER {args.target_layer} ATTENTION SVD ALIGNMENT")
    print("="*80)
    print("\nResearch Question:")
    print(f"  Do the right singular vectors of Layer {args.target_layer} attention weights")
    print("  align with massive activation directions?")
    print("\n" + "="*80)

    # Step 1: Collect activations
    activation_data, model, layers = collect_layer3_activations(args, args.target_layer)
    
    # Step 2: Compute SVD for attention weights
    svd_results = compute_attention_svd(layers[args.target_layer], args.target_layer, model.config)
    
    # Step 3: Analyze alignment
    alignment_results = analyze_attention_alignment(activation_data, svd_results, args.target_layer)
    
    # Step 4: Save results
    print("\n💾 Saving results...")
    serializable_svd = {}
    for weight_name, svd_data in svd_results.items():
        serializable_svd[weight_name] = {
            'singular_values': svd_data['S'].tolist()[:50],  # 只保存前50个
            'shape': svd_data['shape']
        }
    
    with open(os.path.join(args.savedir, f'layer{args.target_layer}_svd_results.json'), 'w') as f:
        json.dump(serializable_svd, f, indent=2)
    
    with open(os.path.join(args.savedir, f'layer{args.target_layer}_alignment_results.json'), 'w') as f:
        serializable_align = {}
        for weight_name, align_data in alignment_results.items():
            serializable_align[weight_name] = {
                'top_alignment': align_data['top_alignment'],
                'mean_abs_alignment': align_data['mean_abs_alignment']
            }
        json.dump(serializable_align, f, indent=2)
    
    # Step 5: Generate visualizations
    generate_visualizations(svd_results, alignment_results, args.savedir, args.target_layer)
    
    # Step 6: Generate summary report
    generate_summary_report(svd_results, alignment_results, args.savedir, args.target_layer)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT 4B COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("="*80)


if __name__ == '__main__':
    main()
