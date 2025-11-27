#!/usr/bin/env python3
"""
Experiment 5: Multi-Model MLP SVD Alignment Analysis
实验五：多模型MLP SVD对齐分析

目标：快速测试多个模型的MLP SVD对齐机制，验证普遍性

测试模型：
1. Mistral-7B (类似LLaMA架构)
2. OPT-6.7B (不同架构)
3. LLaMA-2-7B (对比13B)
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp
from lib.model_utils import is_llama_model


def find_critical_layer(model, tokenizer, device, layers, seq_len, nsamples=5):
    """
    快速找出大规模激活开始的关键层
    """
    print("\n" + "="*80)
    print("FINDING CRITICAL LAYER")
    print("="*80)
    
    # Enable all layers
    for idx, layer in enumerate(layers):
        if is_llama_model(args.model):
            mp.enable_llama_custom_decoderlayer(layer, idx)
        elif "opt" in args.model:
            mp.enable_opt_custom_decoderlayer(layer, idx)
        elif "gpt2" in args.model:
            mp.enable_gpt2_custom_block(layer, idx)
    
    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seq_len, device=device)
    
    layer_stats = {}
    
    with torch.no_grad():
        for testseq in tqdm(testseq_list, desc="Scanning layers"):
            _ = model(testseq)
            
            for idx, layer in enumerate(layers):
                if not hasattr(layer, 'feat') or layer.feat is None:
                    continue
                
                feat = layer.feat.cpu().float().numpy()
                if len(feat.shape) == 3:
                    feat = feat[0]
                
                max_val = np.max(np.abs(feat))
                
                if idx not in layer_stats:
                    layer_stats[idx] = []
                layer_stats[idx].append(max_val)
    
    # Find jump point
    mean_max = {idx: np.mean(vals) for idx, vals in layer_stats.items()}
    
    print("\nLayer activation statistics:")
    for idx in sorted(mean_max.keys())[:10]:
        print(f"  Layer {idx}: {mean_max[idx]:.2f}")
    
    # Find first layer with significant activation (>1000)
    critical_layer = None
    for idx in sorted(mean_max.keys()):
        if mean_max[idx] > 1000:
            critical_layer = idx
            break
    
    if critical_layer is None:
        critical_layer = 2  # Default
    
    print(f"\n✅ Critical layer identified: Layer {critical_layer}")
    print(f"   Mean max activation: {mean_max[critical_layer]:.2f}")
    
    return critical_layer


def analyze_mlp_svd_quick(model, tokenizer, device, layers, target_layer, nsamples=10):
    """
    快速分析单层MLP的SVD对齐
    """
    print("\n" + "="*80)
    print(f"ANALYZING LAYER {target_layer} MLP SVD ALIGNMENT")
    print("="*80)
    
    layer = layers[target_layer]
    
    # 1. Compute SVD of down_proj
    print("\nComputing SVD of down_proj...")
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
        W_down = layer.mlp.down_proj.weight.data.cpu().float().numpy()
        print(f"  down_proj shape: {W_down.shape}")
        
        U, S, Vh = np.linalg.svd(W_down, full_matrices=False)
        
        print(f"  Top 5 singular values: {S[:5]}")
        print(f"  σ₁/σ₂ ratio: {S[0]/S[1]:.4f}")
    else:
        print("  ❌ No MLP down_proj found!")
        return None
    
    # 2. Collect activations
    print(f"\nCollecting activations from {nsamples} samples...")
    
    if is_llama_model(args.model):
        mp.enable_llama_custom_decoderlayer(layer, target_layer)
    elif "opt" in args.model:
        mp.enable_opt_custom_decoderlayer(layer, target_layer)
    elif "gpt2" in args.model:
        mp.enable_gpt2_custom_block(layer, target_layer)
    
    seq_len = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seq_len, device=device)
    
    layer_outputs = []
    
    with torch.no_grad():
        for testseq in tqdm(testseq_list, desc=f"Layer {target_layer}"):
            _ = model(testseq)
            
            if hasattr(layer, 'feat') and layer.feat is not None:
                feat = layer.feat.cpu().float().numpy()
                if len(feat.shape) == 3:
                    feat = feat[0]
                
                for i in range(feat.shape[0]):
                    layer_outputs.append(feat[i])
    
    layer_outputs = np.array(layer_outputs)
    print(f"  Collected {len(layer_outputs)} token activations")
    
    # 3. Compute alignment
    print("\nComputing SVD alignment...")
    
    mean_activation = np.mean(layer_outputs, axis=0)
    mean_activation_norm = mean_activation / (np.linalg.norm(mean_activation) + 1e-8)
    
    # Align with left singular vector u_1
    u1 = U[:, 0]
    cosine_sim = np.dot(u1, mean_activation_norm)
    
    print(f"  Cosine similarity with u₁: {cosine_sim:.4f}")
    print(f"  |Cosine similarity|: {np.abs(cosine_sim):.4f}")
    
    # 4. Linear regression
    print("\nLinear regression analysis...")
    
    projections = layer_outputs @ u1
    max_activations = np.max(np.abs(layer_outputs), axis=1)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(projections, max_activations)
    r_squared = r_value ** 2
    
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.2e}")
    
    if r_squared > 0.7:
        print(f"  ✅ STRONG ALIGNMENT (similar to GPT-2/LLaMA-2)")
    elif r_squared > 0.3:
        print(f"  ⚠️ MODERATE ALIGNMENT")
    else:
        print(f"  ❌ WEAK ALIGNMENT (different mechanism)")
    
    return {
        'layer': target_layer,
        'sigma_1': float(S[0]),
        'sigma_2': float(S[1]),
        'sigma_ratio': float(S[0]/S[1]),
        'cosine_similarity': float(cosine_sim),
        'abs_cosine_similarity': float(np.abs(cosine_sim)),
        'r_squared': float(r_squared),
        'p_value': float(p_value),
        'slope': float(slope),
        'intercept': float(intercept)
    }


def main():
    parser = argparse.ArgumentParser(description='Experiment 5: Multi-Model MLP SVD')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name (mistral_7b, opt_7b, llama2_7b, etc.)')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--auto_find_layer', action='store_true',
                       help='Automatically find critical layer')
    parser.add_argument('--target_layer', type=int, default=None,
                       help='Specific layer to analyze (if not auto-finding)')
    parser.add_argument('--savedir', type=str, default='results/exp5_multi_model/')

    global args
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "="*80)
    print(f"EXPERIMENT 5: MLP SVD ALIGNMENT - {args.model.upper()}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    
    print(f"  Model: {args.model}")
    print(f"  Layers: {len(layers)}")
    print(f"  Hidden size: {hidden_size}")
    
    # Find or use target layer
    if args.auto_find_layer:
        target_layer = find_critical_layer(model, tokenizer, device, layers, seq_len, nsamples=5)
    else:
        target_layer = args.target_layer if args.target_layer is not None else 3
        print(f"\nUsing specified layer: {target_layer}")
    
    # Analyze MLP SVD
    results = analyze_mlp_svd_quick(model, tokenizer, device, layers, target_layer, args.nsamples)
    
    if results is None:
        print("\n❌ Analysis failed!")
        return
    
    # Save results
    model_name = args.model.replace('/', '_')
    result_file = os.path.join(args.savedir, f'{model_name}_layer{target_layer}_results.json')
    
    results['model'] = args.model
    results['nsamples'] = args.nsamples
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Layer: {target_layer}")
    print(f"σ₁/σ₂ ratio: {results['sigma_ratio']:.4f}")
    print(f"|Cosine similarity|: {results['abs_cosine_similarity']:.4f}")
    print(f"R² = {results['r_squared']:.4f}")
    
    if results['r_squared'] > 0.7:
        print("\n✅ CONCLUSION: Uses SVD alignment mechanism (like GPT-2/LLaMA-2)")
    elif results['r_squared'] > 0.3:
        print("\n⚠️ CONCLUSION: Moderate alignment, mechanism unclear")
    else:
        print("\n❌ CONCLUSION: Different mechanism from GPT-2/LLaMA-2")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
