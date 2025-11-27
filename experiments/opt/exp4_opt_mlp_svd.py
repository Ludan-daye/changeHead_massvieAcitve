#!/usr/bin/env python3
"""
Experiment 4: OPT MLP SVD Alignment Analysis
实验四：OPT MLP层 SVD 对齐分析

重点分析：
1. Layer 0 的 fc2 (输出层) 是否有主导奇异方向？
2. 该奇异方向是否与 MLP 的输出激活向量对齐？
3. Layer 0 的 fc1 (输入层) 是否对特定的 Embedding 输入敏感？
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import lib

def compute_opt_mlp_svd(layer, layer_id):
    """Compute SVD for OPT MLP weights (fc1, fc2)"""
    print(f"\nComputing SVD for Layer {layer_id} MLP Weights...")
    svd_results = {}
    
    # OPT MLP Structure: fc1 -> ReLU -> fc2
    
    # 1. FC1 (Input Projection): [intermediate_dim, hidden_dim]
    if hasattr(layer, 'fc1'):
        W_fc1 = layer.fc1.weight.data.cpu().float().numpy()
        # bias_fc1 = layer.fc1.bias.data.cpu().float().numpy()
        print(f"  fc1 shape: {W_fc1.shape}")
        
        U_fc1, S_fc1, Vh_fc1 = np.linalg.svd(W_fc1, full_matrices=False)
        svd_results['fc1'] = {
            'U': U_fc1,
            'S': S_fc1,
            'Vh': Vh_fc1,
            'shape': W_fc1.shape
        }
        print(f"  fc1 Top 5 Singular Values: {S_fc1[:5]}")
        print(f"  fc1 Ratio σ₁/σ₂: {S_fc1[0]/S_fc1[1]:.4f}")

    # 2. FC2 (Output Projection): [hidden_dim, intermediate_dim]
    # Note: In PyTorch Linear layer, weight is [out_features, in_features]
    if hasattr(layer, 'fc2'):
        W_fc2 = layer.fc2.weight.data.cpu().float().numpy()
        print(f"  fc2 shape: {W_fc2.shape}")
        
        U_fc2, S_fc2, Vh_fc2 = np.linalg.svd(W_fc2, full_matrices=False)
        svd_results['fc2'] = {
            'U': U_fc2,     # Left Singular Vectors (Output Space) -> [hidden, min_dim]
            'S': S_fc2,
            'Vh': Vh_fc2,   # Right Singular Vectors (Input Space)
            'shape': W_fc2.shape
        }
        print(f"  fc2 Top 5 Singular Values: {S_fc2[:5]}")
        print(f"  fc2 Ratio σ₁/σ₂: {S_fc2[0]/S_fc2[1]:.4f}")
        print(f"  -> We expect fc2's U[:, 0] to align with massive activations.")

    return svd_results

def collect_activations_and_align(model, tokenizer, device, layer_ids, svd_data, nsamples=30):
    """Collect activations and compute alignment with SVD vectors"""
    
    print(f"\nCollecting activations for Layers {layer_ids}...")
    
    # Storage for activations
    # we need: 
    # 1. MLP Input (x) -> align with fc1 Vh
    # 2. MLP Output (y) -> align with fc2 U
    
    acts = {lid: {'mlp_in': [], 'mlp_out': []} for lid in layer_ids}
    
    # Hooks
    hooks = []
    
    def get_hook(lid, key):
        def hook(module, input, output):
            # input is a tuple
            if key == 'mlp_in':
                val = input[0].detach().cpu().float()
            else:
                val = output.detach().cpu().float()
            
            # Store only flattened or max vectors to save memory?
            # Let's store the max magnitude vectors (where massive activation happens)
            # Shape: [batch, seq, hidden]
            
            # Ensure val is [batch, seq, hidden] or handle 2D
            if val.dim() == 2:
                # [tokens, hidden] - treat as batch=1, seq=tokens
                val = val.unsqueeze(0)
            
            # Find token with max norm
            norms = val.norm(dim=-1) # [batch, seq]
            max_val, max_idx = norms.max(dim=-1) # [batch]
            
            # Extract that vector
            batch_size = val.shape[0]
            max_indices = max_idx.tolist()
            if not isinstance(max_indices, list):
                max_indices = [max_indices]
                
            for b in range(batch_size):
                idx = max_indices[b]
                vec = val[b, idx, :]
                acts[lid][key].append(vec.numpy())
                
        return hook

    for lid in layer_ids:
        layer = model.model.decoder.layers[lid]
        # Hook FC1 Input
        hooks.append(layer.fc1.register_forward_hook(get_hook(lid, 'mlp_in')))
        # Hook FC2 Output
        hooks.append(layer.fc2.register_forward_hook(get_hook(lid, 'mlp_out')))
        
    # Run Inference
    dataloader = lib.get_data(tokenizer, nsamples=nsamples, seqlen=2048, device=device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, tuple):
                batch = batch[0]
            batch = batch.to(device)
            model(batch)
            
    for h in hooks:
        h.remove()
        
    # Compute Alignment
    print("\nComputing Alignments...")
    
    alignment_results = {}
    
    for lid in layer_ids:
        print(f"\n--- Layer {lid} ---")
        
        # 1. FC2 Output Alignment (Massive Activation vs FC2 U1)
        if 'fc2' in svd_data[lid]:
            U = svd_data[lid]['fc2']['U'] # [hidden, k]
            u1 = U[:, 0] # First singular vector
            
            mlp_out_vecs = np.array(acts[lid]['mlp_out']) # [N, hidden]
            
            # Compute cosine similarity
            # Normalize activations
            mlp_out_norms = np.linalg.norm(mlp_out_vecs, axis=1, keepdims=True)
            mlp_out_normalized = mlp_out_vecs / (mlp_out_norms + 1e-8)
            
            # Cosine sim with u1
            # u1 is already normalized
            cos_sims = np.abs(np.dot(mlp_out_normalized, u1))
            
            avg_sim = np.mean(cos_sims)
            print(f"FC2 Output Alignment with U1: {avg_sim:.4f}")
            
            alignment_results[lid] = {
                'fc2_u1_alignment': avg_sim,
                'fc2_sigma_ratio': svd_data[lid]['fc2']['S'][0] / svd_data[lid]['fc2']['S'][1]
            }
            
            # Optional: Check R^2 like in GPT-2 paper
            # Projection on U1 vs Norm
            projections = np.abs(np.dot(mlp_out_vecs, u1))
            norms = mlp_out_norms.flatten()
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(projections, norms)
            print(f"FC2 Projection vs Norm R²: {r_value**2:.4f}")
            alignment_results[lid]['fc2_r2'] = r_value**2

    return alignment_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt_7b')
    parser.add_argument('--layers', type=str, default='0,1,2,3,29,30,31', help='Comma separated layer ids')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--savedir', type=str, default='results/exp4_opt_svd/')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    layer_ids = [int(x) for x in args.layers.split(',')]
    
    print("="*80)
    print(f"EXPERIMENT 4: OPT MLP SVD ALIGNMENT")
    print(f"Target Layers: {layer_ids}")
    print("="*80)
    
    # Load Model
    class Args:
        def __init__(self):
            self.model = args.model
            self.dataset = 'wikitext'
            self.nsamples = args.nsamples
            self.seed = 0
            self.access_token = 'type in your access token here'
    
    model_args = Args()
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(model_args)
    
    # Compute SVD
    svd_data = {}
    for lid in layer_ids:
        svd_data[lid] = compute_opt_mlp_svd(layers[lid], lid)
        
    # Collect Activations & Align
    alignments = collect_activations_and_align(model, tokenizer, device, layer_ids, svd_data, args.nsamples)
    
    # Save Results
    res_path = os.path.join(args.savedir, 'alignment_results.json')
    with open(res_path, 'w') as f:
        json.dump(alignments, f, indent=2)
        
    print(f"\nResults saved to {res_path}")

if __name__ == "__main__":
    main()
