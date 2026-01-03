#!/usr/bin/env python3
"""
Experiment 4: SVD Direction Alignment Analysis
Experiment 4: Singular Value Decomposition direction alignment analysis

Research Question:
Do the right singular vectors of weight matrices that produce massive activations
align with the direction vectors of massive activations?

Methodology:
1. Identify layers and dimensions that produce massive activations
2. Perform SVD decomposition on weight matrices of these layers
3. Extract right singular vectors (principal directions in input space)
4. Calculate cosine similarity between right singular vectors and actual activation vectors
5. Analyze alignment degree
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


def compute_svd_for_layer(layer, layer_id, model_type='llama'):
    """
    Perform SVD decomposition on layer weight matrices
    Returns: U, S, Vh (right singular vectors are in Vh)
    """
    print(f"\nComputing SVD for Layer {layer_id}...")
    
    svd_results = {}
    
    if model_type == 'llama' or model_type == 'opt':
        # LLaMA/OPT: Analyze MLP weights
        # MLP has two linear layers: up_proj (or gate_proj) and down_proj
        
        # 1. MLP up_proj/gate_proj: [hidden_dim, intermediate_dim]
        if hasattr(layer.mlp, 'up_proj'):
            W_up = layer.mlp.up_proj.weight.data.cpu().float().numpy()  # [intermediate, hidden]
            print(f"  up_proj shape: {W_up.shape}")
            U_up, S_up, Vh_up = np.linalg.svd(W_up, full_matrices=False)
            svd_results['up_proj'] = {
                'U': U_up,  # [intermediate, min(intermediate, hidden)]
                'S': S_up,  # [min(intermediate, hidden)]
                'Vh': Vh_up,  # [min(intermediate, hidden), hidden]
                'shape': W_up.shape
            }
        
        # 2. MLP gate_proj (LLaMA-specific)
        if hasattr(layer.mlp, 'gate_proj'):
            W_gate = layer.mlp.gate_proj.weight.data.cpu().float().numpy()
            print(f"  gate_proj shape: {W_gate.shape}")
            U_gate, S_gate, Vh_gate = np.linalg.svd(W_gate, full_matrices=False)
            svd_results['gate_proj'] = {
                'U': U_gate,
                'S': S_gate,
                'Vh': Vh_gate,
                'shape': W_gate.shape
            }
        
        # 3. MLP down_proj: [intermediate_dim, hidden_dim]
        if hasattr(layer.mlp, 'down_proj'):
            W_down = layer.mlp.down_proj.weight.data.cpu().float().numpy()  # [hidden, intermediate]
            print(f"  down_proj shape: {W_down.shape}")
            U_down, S_down, Vh_down = np.linalg.svd(W_down, full_matrices=False)
            svd_results['down_proj'] = {
                'U': U_down,  # [hidden, min(hidden, intermediate)]
                'S': S_down,
                'Vh': Vh_down,  # [min(hidden, intermediate), intermediate]
                'shape': W_down.shape
            }
        
        # 4. Self-attention output projection: [hidden_dim, hidden_dim]
        if hasattr(layer.self_attn, 'o_proj'):
            W_o = layer.self_attn.o_proj.weight.data.cpu().float().numpy()
            print(f"  o_proj shape: {W_o.shape}")
            U_o, S_o, Vh_o = np.linalg.svd(W_o, full_matrices=False)
            svd_results['o_proj'] = {
                'U': U_o,
                'S': S_o,
                'Vh': Vh_o,
                'shape': W_o.shape
            }
    
    elif model_type == 'gpt2':
        # GPT-2: Analyze MLP weights
        W_fc = layer.mlp.c_fc.weight.data.cpu().float().numpy().T  # GPT-2 weights are transposed
        U_fc, S_fc, Vh_fc = np.linalg.svd(W_fc, full_matrices=False)
        svd_results['c_fc'] = {
            'U': U_fc,
            'S': S_fc,
            'Vh': Vh_fc,
            'shape': W_fc.shape
        }
        
        W_proj = layer.mlp.c_proj.weight.data.cpu().float().numpy().T
        U_proj, S_proj, Vh_proj = np.linalg.svd(W_proj, full_matrices=False)
        svd_results['c_proj'] = {
            'U': U_proj,
            'S': S_proj,
            'Vh': Vh_proj,
            'shape': W_proj.shape
        }
    
    return svd_results


def collect_activations_with_directions(args):
    """
    Collect activation values and their direction vectors
    """
    print("\n" + "="*80)
    print("COLLECTING ACTIVATIONS AND DIRECTIONS")
    print("="*80)
    
    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    # Enable feature capture
    for layer_id in range(len(layers)):
        if is_llama_model(args.model):
            mp.enable_llama_custom_decoderlayer(layers[layer_id], layer_id)
        elif "opt" in args.model:
            mp.enable_opt_custom_decoderlayer(layers[layer_id], layer_id)
        elif "gpt2" in args.model:
            mp.enable_gpt2_custom_block(layers[layer_id], layer_id)

    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Storage
    n_layers = len(layers)
    activation_data = {}
    
    for layer_id in range(n_layers):
        activation_data[layer_id] = {
            'max_activations': [],  # Maximum activation value for each sample
            'max_positions': [],    # Position of maximum activation (token_idx, dim_idx)
            'activation_vectors': [],  # Complete activation vectors
            'max_activation_vectors': []  # Vector of the token that produces maximum activation
        }

    print(f"\nProcessing {len(testseq_list)} samples...")

    # Process samples
    with torch.no_grad():
        for idx, testseq in enumerate(tqdm(testseq_list, desc="Collecting activations")):
            _ = model(testseq)

            for layer_id in range(n_layers):
                layer = layers[layer_id]
                if not hasattr(layer, 'feat') or layer.feat is None:
                    continue

                # feat shape: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
                feat = layer.feat.cpu().float().numpy()
                
                if len(feat.shape) == 3:
                    feat = feat[0]  # [seq_len, hidden_dim]
                
                # Find maximum activation value and its position
                feat_abs = np.abs(feat)
                max_val = np.max(feat_abs)
                max_pos = np.unravel_index(np.argmax(feat_abs), feat_abs.shape)
                token_idx, dim_idx = max_pos
                
                # Save data
                activation_data[layer_id]['max_activations'].append(max_val)
                activation_data[layer_id]['max_positions'].append((token_idx, dim_idx))
                activation_data[layer_id]['activation_vectors'].append(feat)
                activation_data[layer_id]['max_activation_vectors'].append(feat[token_idx, :])

    return activation_data, model, layers


def analyze_svd_alignment(activation_data, layers, args):
    """
    Analyze alignment degree between SVD directions and activation directions
    """
    print("\n" + "="*80)
    print("ANALYZING SVD-ACTIVATION ALIGNMENT")
    print("="*80)
    
    model_type = 'llama' if is_llama_model(args.model) else 'gpt2' if 'gpt2' in args.model else 'opt'
    
    alignment_results = {}
    
    # Focus on analyzing layers that produce massive activations (from Experiment 1 results)
    critical_layers = list(range(3, 38))  # Layer 3-37
    
    for layer_id in tqdm(critical_layers, desc="Analyzing layers"):
        print(f"\n{'='*60}")
        print(f"Layer {layer_id}")
        print(f"{'='*60}")
        
        # 1. Perform SVD on this layer
        svd_results = compute_svd_for_layer(layers[layer_id], layer_id, model_type)

        # 2. Get activation vectors for this layer
        max_act_vectors = np.array(activation_data[layer_id]['max_activation_vectors'])  # [n_samples, hidden_dim]

        # 3. Calculate mean activation direction
        mean_activation = np.mean(max_act_vectors, axis=0)  # [hidden_dim]
        mean_activation = mean_activation / (np.linalg.norm(mean_activation) + 1e-8)  # Normalize

        # 4. For each weight matrix, calculate alignment between right singular vectors and activation direction
        layer_alignments = {}
        
        for weight_name, svd_data in svd_results.items():
            Vh = svd_data['Vh']  # [k, input_dim], each row is a right singular vector
            S = svd_data['S']    # Singular values

            # Calculate cosine similarity between top k right singular vectors and mean activation direction
            k = min(10, Vh.shape[0])  # Take top 10 principal components
            alignments = []
            
            for i in range(k):
                right_singular_vec = Vh[i, :]  # i-th right singular vector

                # Ensure dimension matching
                if right_singular_vec.shape[0] == mean_activation.shape[0]:
                    # Calculate cosine similarity
                    cosine_sim = np.dot(right_singular_vec, mean_activation) / (
                        np.linalg.norm(right_singular_vec) * np.linalg.norm(mean_activation) + 1e-8
                    )
                    alignments.append({
                        'component': i,
                        'singular_value': float(S[i]),
                        'cosine_similarity': float(cosine_sim),
                        'abs_cosine_similarity': float(np.abs(cosine_sim))
                    })
            
            layer_alignments[weight_name] = alignments
            
            # Print best aligned component
            if alignments:
                best_alignment = max(alignments, key=lambda x: x['abs_cosine_similarity'])
                print(f"  {weight_name}:")
                print(f"    Best alignment: Component {best_alignment['component']}")
                print(f"    Cosine similarity: {best_alignment['cosine_similarity']:.4f}")
                print(f"    Singular value: {best_alignment['singular_value']:.2f}")
        
        alignment_results[layer_id] = {
            'svd_results': svd_results,
            'alignments': layer_alignments,
            'mean_activation': mean_activation.tolist(),
            'max_activation_value': float(np.mean(activation_data[layer_id]['max_activations']))
        }
    
    return alignment_results


def generate_visualizations(alignment_results, savedir):
    """Generate visualizations"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(savedir, exist_ok=True)
    
    # 1. Plot maximum alignment for each layer
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    layers = sorted(alignment_results.keys())

    # Plot alignment for each weight type
    weight_types = ['up_proj', 'gate_proj', 'down_proj', 'o_proj']
    
    for idx, weight_type in enumerate(weight_types):
        ax = axes[idx // 2, idx % 2]
        
        max_alignments = []
        for layer_id in layers:
            if weight_type in alignment_results[layer_id]['alignments']:
                alignments = alignment_results[layer_id]['alignments'][weight_type]
                if alignments:
                    max_align = max([a['abs_cosine_similarity'] for a in alignments])
                    max_alignments.append(max_align)
                else:
                    max_alignments.append(0)
            else:
                max_alignments.append(0)
        
        ax.plot(layers, max_alignments, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Max Cosine Similarity', fontsize=12)
        ax.set_title(f'{weight_type} - SVD Alignment', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 threshold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'exp4_svd_alignment_by_weight.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: exp4_svd_alignment_by_weight.png")
    
    # 2. Plot heatmap: layer vs singular value component
    fig, ax = plt.subplots(figsize=(14, 10))

    # Use down_proj as example
    heatmap_data = []
    for layer_id in layers:
        if 'down_proj' in alignment_results[layer_id]['alignments']:
            alignments = alignment_results[layer_id]['alignments']['down_proj']
            row = [a['abs_cosine_similarity'] for a in alignments[:10]]
            heatmap_data.append(row)
        else:
            heatmap_data.append([0] * 10)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'PC{i}' for i in range(10)],
                yticklabels=[f'L{i}' for i in layers],
                cbar_kws={'label': 'Cosine Similarity'},
                ax=ax)
    ax.set_title('SVD Right Singular Vectors Alignment with Activation Direction\n(down_proj)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'exp4_alignment_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: exp4_alignment_heatmap.png")


def generate_summary_report(alignment_results, savedir):
    """Generate summary report"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    lines = []
    lines.append("="*80)
    lines.append("EXPERIMENT 4: SVD DIRECTION ALIGNMENT ANALYSIS - SUMMARY")
    lines.append("="*80)
    lines.append("\nRESEARCH QUESTION:")
    lines.append("  Do the right singular vectors of weight matrices align with")
    lines.append("  the direction of massive activations?")
    lines.append("\n" + "="*80)
    lines.append("KEY FINDINGS")
    lines.append("="*80)
    
    # Count layers with high alignment
    high_alignment_layers = {}
    weight_types = ['up_proj', 'gate_proj', 'down_proj', 'o_proj']
    
    for weight_type in weight_types:
        high_layers = []
        for layer_id in sorted(alignment_results.keys()):
            if weight_type in alignment_results[layer_id]['alignments']:
                alignments = alignment_results[layer_id]['alignments'][weight_type]
                if alignments:
                    max_align = max([a['abs_cosine_similarity'] for a in alignments])
                    if max_align > 0.5:  # Threshold
                        high_layers.append((layer_id, max_align))
        high_alignment_layers[weight_type] = high_layers
    
    for weight_type in weight_types:
        lines.append(f"\n{weight_type.upper()}:")
        lines.append("-"*80)
        if high_alignment_layers[weight_type]:
            lines.append(f"  Found {len(high_alignment_layers[weight_type])} layers with high alignment (>0.5)")
            lines.append("  Top 5 layers:")
            sorted_layers = sorted(high_alignment_layers[weight_type], key=lambda x: x[1], reverse=True)[:5]
            for rank, (layer_id, align) in enumerate(sorted_layers, 1):
                lines.append(f"    {rank}. Layer {layer_id}: {align:.4f}")
        else:
            lines.append("  No layers with high alignment (>0.5)")
    
    lines.append("\n" + "="*80)
    lines.append("CONCLUSION")
    lines.append("="*80)
    
    # Judge overall alignment situation
    total_high_alignment = sum(len(v) for v in high_alignment_layers.values())
    total_tested = len(alignment_results) * len(weight_types)
    alignment_ratio = total_high_alignment / total_tested if total_tested > 0 else 0
    
    if alignment_ratio > 0.3:
        lines.append("\n✅ STRONG ALIGNMENT DETECTED")
        lines.append(f"  {alignment_ratio*100:.1f}% of weight matrices show high alignment with")
        lines.append("  massive activation directions. This suggests that:")
        lines.append("  1. Massive activations are driven by specific weight directions")
        lines.append("  2. These directions correspond to principal components of the weights")
        lines.append("  3. Pruning or modifying these components may reduce massive activations")
    elif alignment_ratio > 0.1:
        lines.append("\n⚠️ MODERATE ALIGNMENT DETECTED")
        lines.append(f"  {alignment_ratio*100:.1f}% of weight matrices show alignment.")
        lines.append("  Further investigation needed.")
    else:
        lines.append("\n❌ WEAK ALIGNMENT")
        lines.append(f"  Only {alignment_ratio*100:.1f}% alignment detected.")
        lines.append("  Massive activations may be caused by other mechanisms.")
    
    lines.append("\n" + "="*80)
    
    summary_text = "\n".join(lines)
    print(summary_text)
    
    with open(os.path.join(savedir, 'EXPERIMENT_4_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n✅ Summary saved to: {os.path.join(savedir, 'EXPERIMENT_4_SUMMARY.txt')}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 4: SVD Direction Alignment')
    parser.add_argument('--model', type=str, default='llama2_13b')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='results/exp4_llama2_13b/')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENT 4: SVD DIRECTION ALIGNMENT ANALYSIS")
    print("="*80)
    print("\nResearch Question:")
    print("  Do the right singular vectors of weight matrices align with")
    print("  the direction of massive activations?")
    print("\n" + "="*80)

    # Step 1: Collect activations
    activation_data, model, layers = collect_activations_with_directions(args)
    
    # Step 2: Analyze SVD alignment
    alignment_results = analyze_svd_alignment(activation_data, layers, args)
    
    # Step 3: Save results
    print("\n Saving results...")
    with open(os.path.join(args.savedir, 'alignment_results.json'), 'w') as f:
        # Only save serializable parts
        serializable_results = {}
        for layer_id, data in alignment_results.items():
            serializable_results[layer_id] = {
                'alignments': data['alignments'],
                'max_activation_value': data['max_activation_value']
            }
        json.dump(serializable_results, f, indent=2)
    
    # Step 4: Generate visualizations
    generate_visualizations(alignment_results, args.savedir)
    
    # Step 5: Generate summary report
    generate_summary_report(alignment_results, args.savedir)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT 4 COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("="*80)


if __name__ == '__main__':
    main()
