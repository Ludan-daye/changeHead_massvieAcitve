#!/usr/bin/env python3
"""
Experiment 2: Single-Layer Restoration
实验二：单层恢复 - 找出哪些层的注意力头对大规模激活贡献最大

基于实验一的结果，我们知道禁用所有注意力头会显著降低激活值。
现在我们逐层恢复注意力头，找出关键层。
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


class SelectiveHeadDisableHook:
    """Hook to disable all heads EXCEPT those in a specific layer"""
    def __init__(self, layer_id, num_heads, target_layer_id):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.target_layer_id = target_layer_id  # 只有这一层的头会被启用

    def __call__(self, module, input, output):
        # 如果是目标层，不做任何修改（保持头启用）
        if self.layer_id == self.target_layer_id:
            return output
        
        # 否则禁用所有头
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
        attn_output_reshaped[:, :, :, :] = 0
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


def run_single_layer_experiment(args, restore_layer_id):
    """运行单层恢复实验：禁用所有层的头，只恢复指定层"""
    
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

    # Register hooks: disable all except restore_layer_id
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

        hook = SelectiveHeadDisableHook(layer_id, num_heads, restore_layer_id)
        handle = target_module.register_forward_hook(hook)
        hooks.append(handle)

    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Storage
    n_layers = len(layers)
    layer_stats = {lid: {'top1': [], 'median': []} for lid in range(n_layers)}

    # Process samples
    with torch.no_grad():
        for testseq in tqdm(testseq_list, desc=f"Layer {restore_layer_id} restored"):
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
    parser = argparse.ArgumentParser(description='Experiment 2: Single-Layer Restoration')
    parser.add_argument('--model', type=str, default='llama2_13b')
    parser.add_argument('--access_token', type=str, default='type in your access token here')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='results/exp2_llama2_13b/')
    parser.add_argument('--start_layer', type=int, default=3, help='Start layer to test')
    parser.add_argument('--end_layer', type=int, default=37, help='End layer to test')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENT 2: SINGLE-LAYER RESTORATION")
    print("="*80)
    print("\nGoal: Identify which layers contribute most to massive activations")
    print(f"Testing layers: {args.start_layer} to {args.end_layer}")
    print("\n" + "="*80)

    # Load baseline results from Exp1
    baseline_path = 'results/exp1_llama2_13b/baseline/results.json'
    all_disabled_path = 'results/exp1_llama2_13b/all_heads_disabled/results.json'
    
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    with open(all_disabled_path, 'r') as f:
        all_disabled_results = json.load(f)

    # Run experiments for each layer
    all_results = {}
    
    for restore_layer in range(args.start_layer, args.end_layer + 1):
        print(f"\n{'='*80}")
        print(f"Testing Layer {restore_layer} (restoring its attention heads)")
        print(f"{'='*80}")
        
        results = run_single_layer_experiment(args, restore_layer)
        all_results[restore_layer] = results
        
        # Save intermediate results
        with open(os.path.join(args.savedir, f'layer_{restore_layer}_results.json'), 'w') as f:
            json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                           for kk, vv in v.items()}
                       for k, v in results.items()}, f, indent=2)
        
        # Print key metrics
        print(f"\nLayer {restore_layer} restored - Key metrics:")
        for key_layer in [3, 10, 20, 30, 37]:
            if key_layer < len(results):
                top1 = results[key_layer]['top1_mean']
                baseline_top1 = float(baseline_results[str(key_layer)]['top1_mean'])
                disabled_top1 = float(all_disabled_results[str(key_layer)]['top1_mean'])
                recovery = ((top1 - disabled_top1) / (baseline_top1 - disabled_top1)) * 100
                print(f"  Layer {key_layer}: Top1={top1:.2f} (Recovery: {recovery:.1f}%)")

    # Generate summary
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("EXPERIMENT 2: SINGLE-LAYER RESTORATION - SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"\nTested layers: {args.start_layer} to {args.end_layer}")
    summary_lines.append("\nRecovery Rate for Each Restored Layer:")
    summary_lines.append("-"*80)
    summary_lines.append(f"{'Layer':<8} {'Baseline':<12} {'All Disabled':<12} {'Restored':<12} {'Recovery %':<12}")
    summary_lines.append("-"*80)
    
    recovery_rates = {}
    for restore_layer in range(args.start_layer, args.end_layer + 1):
        # 计算该层自身的恢复率
        results = all_results[restore_layer]
        top1 = results[restore_layer]['top1_mean']
        baseline_top1 = float(baseline_results[str(restore_layer)]['top1_mean'])
        disabled_top1 = float(all_disabled_results[str(restore_layer)]['top1_mean'])
        
        if baseline_top1 - disabled_top1 > 0:
            recovery = ((top1 - disabled_top1) / (baseline_top1 - disabled_top1)) * 100
        else:
            recovery = 0
        
        recovery_rates[restore_layer] = recovery
        summary_lines.append(f"{restore_layer:<8} {baseline_top1:<12.2f} {disabled_top1:<12.2f} {top1:<12.2f} {recovery:<12.1f}")
    
    summary_lines.append("\n" + "="*80)
    summary_lines.append("TOP 5 MOST CRITICAL LAYERS (Highest Recovery):")
    summary_lines.append("="*80)
    
    sorted_layers = sorted(recovery_rates.items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (layer_id, recovery) in enumerate(sorted_layers, 1):
        summary_lines.append(f"{rank}. Layer {layer_id}: {recovery:.1f}% recovery")
    
    summary_lines.append("\n" + "="*80)
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(os.path.join(args.savedir, 'EXPERIMENT_2_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n✅ Experiment 2 complete! Results saved to: {args.savedir}")
    print("="*80)


if __name__ == '__main__':
    main()
