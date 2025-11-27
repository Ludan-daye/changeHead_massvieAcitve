#!/usr/bin/env python3
"""
Analyze Experiment 2 Results
分析实验二结果：找出 OPT-6.7B 中的关键抑制层
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    results_dir = 'results/exp2_opt_6.7b'
    
    # 1. Load Baseline and All-Disabled stats (from Exp 1)
    # 注意：这里我们需要 OPT 的 Exp 1 结果
    # 假设 Baseline ~ 391, All Disabled ~ 1370 (根据之前的日志)
    # 为了准确，我们需要读取 Exp 1 的结果文件
    
    exp1_dir = 'results/exp1_opt_6.7b'
    baseline_file = os.path.join(exp1_dir, 'baseline', 'results.json')
    disabled_file = os.path.join(exp1_dir, 'all_heads_disabled', 'results.json')
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
        
    with open(disabled_file, 'r') as f:
        disabled_data = json.load(f)
        
    # 2. Load Exp 2 Results (Single Layer Restored)
    layer_results = {}
    result_files = [f for f in os.listdir(results_dir) if f.startswith('layer_') and f.endswith('.json')]
    
    for file in result_files:
        layer_id = int(file.split('_')[1])
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
            # Extract Top1 Mean for the target layer itself
            # The file contains stats for all layers, but we only care about the restored layer's own activation
            if str(layer_id) in data:
                layer_results[layer_id] = data[str(layer_id)]['top1_mean']
            else:
                # fallback if key is int
                # JSON keys are always strings, but let's be safe
                try:
                    layer_results[layer_id] = data[str(layer_id)]['top1_mean']
                except KeyError:
                     # Try to find the key in data keys (sometimes saved as "3" sometimes 3)
                     found = False
                     for k in data.keys():
                         if int(k) == layer_id:
                             layer_results[layer_id] = data[k]['top1_mean']
                             found = True
                             break
                     if not found:
                         print(f"Warning: Could not find data for layer {layer_id} in {file}")
                         layer_results[layer_id] = 0

    # 3. Calculate Recovery Rates
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("OPT-6.7B EXPERIMENT 2 ANALYSIS: CRITICAL INHIBITION LAYERS")
    summary_lines.append("="*80)
    summary_lines.append(f"{'Layer':<8} {'Baseline':<12} {'Disabled':<12} {'Restored':<12} {'Recovery %':<12}")
    summary_lines.append("-"*80)
    
    recovery_scores = []
    
    sorted_layers = sorted(layer_results.keys())
    
    # Global Baseline/Disabled averages for reference (or per-layer if available)
    # It's better to use per-layer baseline/disabled values
    
    for layer in sorted_layers:
        layer_str = str(layer)
        
        # Get Baseline for this layer
        base_val = baseline_data.get(layer_str, {}).get('top1_mean', 0)
        
        # Get Disabled for this layer
        dis_val = disabled_data.get(layer_str, {}).get('top1_mean', 0)
        
        # Get Restored Value
        res_val = layer_results[layer]
        
        # Calculate Recovery
        # Logic: We want Res_val to go DOWN from Dis_val towards Base_val
        # Total Drop Range = Dis_val - Base_val
        # Actual Drop = Dis_val - Res_val
        
        denom = dis_val - base_val
        if abs(denom) < 1e-5:
            recovery = 0.0
        else:
            recovery = (dis_val - res_val) / denom * 100
            
        recovery_scores.append((layer, recovery, res_val, base_val, dis_val))
        
        summary_lines.append(f"{layer:<8} {base_val:<12.2f} {dis_val:<12.2f} {res_val:<12.2f} {recovery:<12.1f}")

    # 4. Sort by Efficiency
    summary_lines.append("\n" + "="*80)
    summary_lines.append("TOP 10 MOST CRITICAL LAYERS (Highest Recovery Rate)")
    summary_lines.append("="*80)
    
    # Sort descending by recovery
    ranked = sorted(recovery_scores, key=lambda x: x[1], reverse=True)
    
    for rank, (layer, score, res, base, dis) in enumerate(ranked[:10], 1):
        summary_lines.append(f"{rank}. Layer {layer}: {score:.1f}% Recovery (Val: {res:.2f} vs Base: {base:.2f})")
        
    summary_lines.append("\n" + "="*80)
    summary_lines.append("CONCLUSION")
    summary_lines.append("="*80)
    
    best_layer = ranked[0]
    if best_layer[1] > 50:
        summary_lines.append(f"Found critical layer: Layer {best_layer[0]} with {best_layer[1]:.1f}% recovery.")
        summary_lines.append("Restoring this single layer significantly suppresses massive activations.")
    else:
        summary_lines.append("No single layer showed strong recovery (>50%).")
        summary_lines.append(f"Best was Layer {best_layer[0]} with only {best_layer[1]:.1f}%.")
        summary_lines.append("This suggests inhibition requires multi-layer coordination.")

    # Save Report
    output_path = os.path.join(results_dir, 'EXPERIMENT_2_SUMMARY.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
        
    print('\n'.join(summary_lines))
    print(f"\nAnalysis saved to: {output_path}")

if __name__ == "__main__":
    main()
