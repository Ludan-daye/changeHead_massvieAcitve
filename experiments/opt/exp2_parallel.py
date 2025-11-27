#!/usr/bin/env python3
"""
Experiment 2: Parallel Single-Layer Restoration
并行版本 - 同时测试多个层以加速实验
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp_patch
from lib.model_utils import is_llama_model


class SelectiveHeadDisableHook:
    """Hook to disable all heads EXCEPT those in a specific layer"""
    def __init__(self, layer_id, num_heads, target_layer_id):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.target_layer_id = target_layer_id

    def __call__(self, module, input, output):
        if self.layer_id == self.target_layer_id:
            return output
        
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
        attn_output_reshaped[:, :, :, :] = 0
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


def run_single_layer_experiment(model_name, dataset, nsamples, restore_layer_id, savedir, gpu_id):
    """在指定GPU上运行单层恢复实验"""
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 创建临时args对象
    class Args:
        def __init__(self):
            self.model = model_name
            self.dataset = dataset
            self.nsamples = nsamples
            self.seed = 0
            self.access_token = 'type in your access token here'
    
    args = Args()
    
    print(f"[GPU {gpu_id}] Loading model for Layer {restore_layer_id}...")
    
    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    
    # Enable feature capture
    if is_llama_model(args.model):
        for i, layer in enumerate(layers):
            mp_patch.enable_llama_custom_decoderlayer(layer, i)
    elif "opt" in args.model:
        for i, layer in enumerate(layers):
            mp_patch.enable_opt_custom_decoderlayer(layer, i)
    
    # Get number of heads
    if hasattr(layers[0].self_attn, 'num_heads'):
        num_heads = layers[0].self_attn.num_heads
    else:
        num_heads = layers[0].self_attn.num_attention_heads
    
    # Register hooks to disable all heads except target layer
    hooks = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            hook = layer.self_attn.register_forward_hook(
                SelectiveHeadDisableHook(i, num_heads, restore_layer_id)
            )
            hooks.append(hook)
    
    # Load data
    print(f"[GPU {gpu_id}] Layer {restore_layer_id}: Loading dataset...")
    np.random.seed(0)
    torch.manual_seed(0)
    dataloader = lib.get_data(tokenizer, nsamples=nsamples, seqlen=seq_len, device=device)
    
    # Collect activations
    layer_stats = {i: {'top1_values': [], 'median_values': [], 
                       'dim_138_values': [], 'dim_447_values': []} 
                   for i in range(len(layers))}
    
    print(f"[GPU {gpu_id}] Layer {restore_layer_id}: Processing {nsamples} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, 
                                               desc=f"[GPU {gpu_id}] Layer {restore_layer_id}",
                                               position=gpu_id)):
            if isinstance(batch, tuple):
                testseq = batch[0]
            else:
                testseq = batch
            
            testseq = testseq.to(device)
            _ = model(testseq)
            
            for i, layer in enumerate(layers):
                if hasattr(layer, 'feat'):
                    feat = layer.feat
                    top1 = torch.topk(feat.abs().flatten(), k=1)[0].mean().item()
                    median = torch.median(feat.abs()).item()
                    
                    layer_stats[i]['top1_values'].append(top1)
                    layer_stats[i]['median_values'].append(median)
                    
                    if feat.shape[-1] > 447:
                        layer_stats[i]['dim_138_values'].append(feat[:, :, 138].abs().mean().item())
                        layer_stats[i]['dim_447_values'].append(feat[:, :, 447].abs().mean().item())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute statistics
    results = {}
    for i in range(len(layers)):
        results[i] = {
            'top1_mean': np.mean(layer_stats[i]['top1_values']),
            'top1_std': np.std(layer_stats[i]['top1_values']),
            'median_mean': np.mean(layer_stats[i]['median_values']),
            'dim_138_mean': np.mean(layer_stats[i]['dim_138_values']) if layer_stats[i]['dim_138_values'] else 0,
            'dim_447_mean': np.mean(layer_stats[i]['dim_447_values']) if layer_stats[i]['dim_447_values'] else 0,
        }
    
    # Save results
    output_file = os.path.join(savedir, f'layer_{restore_layer_id}_results.json')
    with open(output_file, 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    
    print(f"[GPU {gpu_id}] Layer {restore_layer_id}: Complete! Saved to {output_file}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return restore_layer_id, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt_7b')
    parser.add_argument('--dataset', type=str, default='wikitext', choices=['wikitext', 'c4', 'RedPajama'])
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--savedir', type=str, default='results/exp2_opt_6.7b/')
    parser.add_argument('--start_layer', type=int, default=9, help='Start layer to test')
    parser.add_argument('--end_layer', type=int, default=37, help='End layer to test')
    parser.add_argument('--parallel', type=int, default=3, help='Number of parallel processes (max 5 for 79GB GPU)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENT 2: PARALLEL SINGLE-LAYER RESTORATION")
    print("="*80)
    print(f"\nTesting layers: {args.start_layer} to {args.end_layer}")
    print(f"Parallel processes: {args.parallel}")
    print(f"GPU: {args.gpu}")
    print(f"Estimated GPU memory per process: ~13GB")
    print(f"Total estimated: ~{13 * args.parallel}GB / 79GB")
    print("\n" + "="*80)

    layers_to_test = list(range(args.start_layer, args.end_layer + 1))
    completed_layers = []
    
    # Check for already completed layers
    for layer in layers_to_test[:]:
        result_file = os.path.join(args.savedir, f'layer_{layer}_results.json')
        if os.path.exists(result_file):
            print(f"✓ Layer {layer} already completed, skipping...")
            layers_to_test.remove(layer)
            completed_layers.append(layer)
    
    print(f"\nLayers to process: {len(layers_to_test)}")
    print(f"Already completed: {len(completed_layers)}")
    
    if not layers_to_test:
        print("\n✅ All layers already completed!")
        return
    
    # Run experiments in parallel
    all_results = {}
    
    # Use ProcessPoolExecutor for true parallelism
    # Note: Each process will use the same GPU but PyTorch will handle memory
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        future_to_layer = {}
        for i, layer in enumerate(layers_to_test):
            # Stagger GPU assignment if needed (for multi-GPU)
            gpu_id = args.gpu
            future = executor.submit(
                run_single_layer_experiment,
                args.model, args.dataset, args.nsamples, 
                layer, args.savedir, gpu_id
            )
            future_to_layer[future] = layer
        
        # Collect results as they complete
        for future in as_completed(future_to_layer):
            layer = future_to_layer[future]
            try:
                layer_id, results = future.result()
                all_results[layer_id] = results
                print(f"\n✅ Layer {layer_id} completed ({len(all_results)}/{len(layers_to_test)})")
            except Exception as e:
                print(f"\n❌ Layer {layer} failed with error: {e}")
    
    print("\n" + "="*80)
    print("✅ ALL LAYERS COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print(f"Total layers processed: {len(all_results)}")


if __name__ == '__main__':
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
