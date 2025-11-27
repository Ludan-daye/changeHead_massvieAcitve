#!/usr/bin/env python3
"""
Experiment 3: Single-Head Restoration
实验三：单头恢复 - 在关键层中找出哪些注意力头对大规模激活贡献最大

基于实验二的结果，我们已经知道哪些层是关键的。
现在在这些关键层中，逐个恢复单个注意力头，找出最关键的头。
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp
from lib.model_utils import is_llama_model


class SingleHeadRestoreHook:
    """Hook to disable all heads EXCEPT one specific head in one specific layer"""
    def __init__(self, layer_id, num_heads, target_layer_id, target_head_id):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.target_layer_id = target_layer_id
        self.target_head_id = target_head_id

    def __call__(self, module, input, output):
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
        
        # 如果是目标层
        if self.layer_id == self.target_layer_id:
            # 禁用所有头
            attn_output_reshaped[:, :, :, :] = 0
            # 只恢复目标头（不修改它，相当于保持原值）
            # 实际上我们需要保存原始值
            pass
        else:
            # 其他层：禁用所有头
            attn_output_reshaped[:, :, :, :] = 0
        
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


class SingleHeadRestoreHookV2:
    """改进版：先保存原始输出，再选择性恢复"""
    def __init__(self, layer_id, num_heads, target_layer_id, target_head_id):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.target_layer_id = target_layer_id
        self.target_head_id = target_head_id

    def __call__(self, module, input, output):
        attn_output = output[0].clone()  # 保存原始输出
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads
        
        # Reshape to separate heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
        
        if self.layer_id == self.target_layer_id:
            # 在目标层：保存原始值，然后禁用所有头
            original_head = attn_output_reshaped[:, :, self.target_head_id, :].clone()
            attn_output_reshaped[:, :, :, :] = 0
            # 恢复目标头
            attn_output_reshaped[:, :, self.target_head_id, :] = original_head
        else:
            # 其他层：禁用所有头
            attn_output_reshaped[:, :, :, :] = 0
        
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


def run_single_head_experiment(args, target_layer_id, target_head_id):
    """运行单头恢复实验：禁用所有头，只恢复指定层的指定头"""
    
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

    # Register hooks
    hooks = []
    for layer_id in range(len(layers)):
        layer = layers[layer_id]
        
        if is_llama_model(args.model) or "opt" in args.model:
            num_heads = model.config.num_attention_heads
            target_module = layer.self_attn
        elif "gpt2" in args.model:
            num_heads = model.config.n_head
            target_module = layer.attn
        else:
            raise ValueError(f"Model {args.model} not supported")

        hook = SingleHeadRestoreHookV2(layer_id, num_heads, target_layer_id, target_head_id)
        handle = target_module.register_forward_hook(hook)
        hooks.append(handle)

    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Storage
    n_layers = len(layers)
    layer_stats = {lid: {'top1': [], 'median': []} for lid in range(n_layers)}

    # Process samples
    with torch.no_grad():
        for testseq in tqdm(testseq_list, desc=f"L{target_layer_id}H{target_head_id}", leave=False):
            _ = model(testseq)

            for layer_id in range(n_layers):
                layer = layers[layer_id]
                if not hasattr(layer, 'feat') or layer.feat is None:
                    continue

                feat_abs = layer.feat.abs()
                if len(feat_abs.shape) == 3:
                    feat_abs = feat_abs.view(-1, feat_abs.shape[-1])

                sorted_vals, _ = torch.sort(feat_abs.flatten(), descending=True)
                layer_stats[layer_id]['top1'].append(sorted_vals[0].item())
                layer_stats[layer_id]['median'].append(torch.median(feat_abs).item())

    # Clean up
    for handle in hooks:
        handle.remove()

    # Compute statistics
    results = {}
    for layer_id in range(n_layers):
        results[layer_id] = {
            'top1_mean': np.mean(layer_stats[layer_id]['top1']),
            'top1_std': np.std(layer_stats[layer_id]['top1']),
            'median_mean': np.mean(layer_stats[layer_id]['median']),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Single-Head Restoration')
    parser.add_argument('--model', type=str, default='llama2_13b')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='results/exp3_llama2_13b/')
    parser.add_argument('--target_layers', type=str, default='3,10,20,30', 
                        help='Comma-separated list of layers to test (from Exp2 results)')
    parser.add_argument('--num_heads', type=int, default=40, 
                        help='Number of attention heads per layer')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    # Parse target layers
    target_layers = [int(x) for x in args.target_layers.split(',')]

    print("\n" + "="*80)
    print("EXPERIMENT 3: SINGLE-HEAD RESTORATION")
    print("="*80)
    print("\nGoal: Identify which specific heads contribute most to massive activations")
    print(f"Testing layers: {target_layers}")
    print(f"Heads per layer: {args.num_heads}")
    print(f"Total tests: {len(target_layers) * args.num_heads}")
    print("\n" + "="*80)

    # Load baseline and all_disabled results
    baseline_path = 'results/exp1_llama2_13b/baseline/results.json'
    all_disabled_path = 'results/exp1_llama2_13b/all_heads_disabled/results.json'
    
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    with open(all_disabled_path, 'r') as f:
        all_disabled_results = json.load(f)

    # Run experiments for each layer and each head
    all_results = {}
    
    for target_layer in target_layers:
        print(f"\n{'='*80}")
        print(f"Testing Layer {target_layer} - All {args.num_heads} heads")
        print(f"{'='*80}")
        
        layer_results = {}
        
        for head_id in range(args.num_heads):
            print(f"\rTesting Layer {target_layer}, Head {head_id}/{args.num_heads}...", end='', flush=True)
            
            results = run_single_head_experiment(args, target_layer, head_id)
            layer_results[head_id] = results
            
            # Save intermediate results
            os.makedirs(os.path.join(args.savedir, f'layer_{target_layer}'), exist_ok=True)
            with open(os.path.join(args.savedir, f'layer_{target_layer}', f'head_{head_id}_results.json'), 'w') as f:
                json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
                           for k, v in results.items()}, f, indent=2)
        
        print()  # New line after progress
        all_results[target_layer] = layer_results
        
        # Analyze this layer's heads
        print(f"\nLayer {target_layer} - Top 5 heads by recovery:")
        head_recoveries = {}
        for head_id in range(args.num_heads):
            results = layer_results[head_id]
            top1 = results[target_layer]['top1_mean']
            baseline_top1 = float(baseline_results[str(target_layer)]['top1_mean'])
            disabled_top1 = float(all_disabled_results[str(target_layer)]['top1_mean'])
            
            if baseline_top1 - disabled_top1 > 0:
                recovery = ((top1 - disabled_top1) / (baseline_top1 - disabled_top1)) * 100
            else:
                recovery = 0
            
            head_recoveries[head_id] = recovery
        
        sorted_heads = sorted(head_recoveries.items(), key=lambda x: x[1], reverse=True)[:5]
        for rank, (head_id, recovery) in enumerate(sorted_heads, 1):
            print(f"  {rank}. Head {head_id}: {recovery:.1f}% recovery")

    # Generate summary
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("EXPERIMENT 3: SINGLE-HEAD RESTORATION - SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"\nTested layers: {target_layers}")
    summary_lines.append(f"Heads per layer: {args.num_heads}")
    summary_lines.append("\n" + "="*80)
    summary_lines.append("TOP CRITICAL HEADS (Across All Tested Layers):")
    summary_lines.append("="*80)
    summary_lines.append(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Recovery %':<12} {'Impact':<10}")
    summary_lines.append("-"*80)
    
    # Collect all head recoveries
    all_head_recoveries = []
    for target_layer in target_layers:
        layer_results = all_results[target_layer]
        baseline_top1 = float(baseline_results[str(target_layer)]['top1_mean'])
        disabled_top1 = float(all_disabled_results[str(target_layer)]['top1_mean'])
        
        for head_id in range(args.num_heads):
            results = layer_results[head_id]
            top1 = results[target_layer]['top1_mean']
            
            if baseline_top1 - disabled_top1 > 0:
                recovery = ((top1 - disabled_top1) / (baseline_top1 - disabled_top1)) * 100
            else:
                recovery = 0
            
            all_head_recoveries.append((target_layer, head_id, recovery, top1))
    
    # Sort and get top 20
    sorted_all = sorted(all_head_recoveries, key=lambda x: x[2], reverse=True)[:20]
    for rank, (layer_id, head_id, recovery, top1) in enumerate(sorted_all, 1):
        impact = "HIGH" if recovery > 50 else "MEDIUM" if recovery > 20 else "LOW"
        summary_lines.append(f"{rank:<6} {layer_id:<8} {head_id:<8} {recovery:<12.1f} {impact:<10}")
    
    summary_lines.append("\n" + "="*80)
    summary_lines.append("CONCLUSION:")
    summary_lines.append("="*80)
    summary_lines.append("\nThe above heads are the most critical for generating massive activations.")
    summary_lines.append("These heads should be the focus of further analysis and potential pruning.")
    summary_lines.append("\n" + "="*80)
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(os.path.join(args.savedir, 'EXPERIMENT_3_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n✅ Experiment 3 complete! Results saved to: {args.savedir}")
    print("="*80)


if __name__ == '__main__':
    main()
