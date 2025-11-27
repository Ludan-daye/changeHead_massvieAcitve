#!/usr/bin/env python3
"""
Experiment 3: MLP Fire Intensity Test
实验三：在全Attention禁止的情况下，测试每一层MLP的“放火”强度
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import lib
import monkey_patch as mp_patch

# Hook to disable attention completely
class AllHeadDisableHook:
    def __init__(self, num_heads):
        self.num_heads = num_heads

    def __call__(self, module, input, output):
        # output[0] is attn_output
        # We zero it out completely
        return (torch.zeros_like(output[0]),) + output[1:]

# Hook to capture MLP output intensity
class MLPCaptureHook:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.activations = []

    def __call__(self, module, input, output):
        # OPT MLP output is just the output tensor
        # output shape: (batch, seq_len, hidden_dim)
        # We want to measure the intensity of what MLP adds to the stream
        
        # Measure Top-1 magnitude (max absolute value)
        top1 = output.abs().max().item()
        
        # Measure L2 Norm (energy)
        l2 = output.norm(dim=-1).mean().item()
        
        self.activations.append({'top1': top1, 'l2': l2})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt_7b')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--savedir', type=str, default='results/exp3_opt_fire_test/')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: MLP FIRE INTENSITY TEST (All Heads Disabled)")
    print("="*80)
    
    # 1. Load Model
    class Args:
        def __init__(self):
            self.model = args.model
            self.dataset = args.dataset
            self.nsamples = args.nsamples
            self.seed = 0
            self.access_token = 'type in your access token here'
    
    model_args = Args()
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(model_args)
    
    # 2. Register Hooks
    # A. Disable ALL Attention Heads
    attn_hooks = []
    if hasattr(layers[0].self_attn, 'num_heads'):
        num_heads = layers[0].self_attn.num_heads
    else:
        num_heads = layers[0].self_attn.num_attention_heads
        
    print("1. Disabling ALL Attention Heads...")
    for layer in layers:
        h = layer.self_attn.register_forward_hook(AllHeadDisableHook(num_heads))
        attn_hooks.append(h)
        
    # B. Capture MLP Output
    print("2. Hooking into MLP outputs...")
    mlp_hooks = {}
    for i, layer in enumerate(layers):
        # In OPT, the final linear layer of MLP is 'fc2'
        # We want to capture what comes OUT of fc2
        mlp_capture = MLPCaptureHook(i)
        h = layer.fc2.register_forward_hook(mlp_capture)
        mlp_hooks[i] = {'hook': h, 'capture': mlp_capture}
        
    # 3. Run Inference
    print(f"3. Running Inference on {args.nsamples} samples...")
    torch.manual_seed(0)
    np.random.seed(0)
    dataloader = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, tuple):
                batch = batch[0]
            batch = batch.to(device)
            model(batch)
            
    # 4. Analyze Results
    print("\n" + "="*80)
    print("ANALYSIS: WHICH LAYER'S MLP IS THE BIGGEST ARSONIST?")
    print("="*80)
    print(f"{'Layer':<8} {'MLP Top-1 Output':<20} {'MLP L2 Energy':<20}")
    print("-" * 60)
    
    layer_scores = []
    
    for i in range(len(layers)):
        capture = mlp_hooks[i]['capture']
        # Calculate averages
        avg_top1 = np.mean([x['top1'] for x in capture.activations])
        avg_l2 = np.mean([x['l2'] for x in capture.activations])
        
        layer_scores.append({
            'layer': i,
            'top1': avg_top1,
            'l2': avg_l2
        })
        
        print(f"{i:<8} {avg_top1:<20.4f} {avg_l2:<20.4f}")
        
    # 5. Save and Rank
    with open(os.path.join(args.savedir, 'mlp_fire_stats.json'), 'w') as f:
        json.dump(layer_scores, f, indent=2)
        
    print("\n" + "="*80)
    print("TOP 5 'FIRE STARTERS' (Highest MLP Output)")
    print("="*80)
    
    ranked = sorted(layer_scores, key=lambda x: x['top1'], reverse=True)
    for rank, item in enumerate(ranked[:5], 1):
        print(f"{rank}. Layer {item['layer']}: MLP Output = {item['top1']:.4f}")
        
    # 6. Generate Plot
    layers_x = [x['layer'] for x in layer_scores]
    top1_y = [x['top1'] for x in layer_scores]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers_x, top1_y, marker='o', color='red', linewidth=2)
    plt.title('OPT-6.7B MLP Output Intensity (All Attention Disabled)')
    plt.xlabel('Layer Index')
    plt.ylabel('MLP Output Top-1 Magnitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.savedir, 'mlp_fire_intensity.png'))
    print(f"\nPlot saved to {os.path.join(args.savedir, 'mlp_fire_intensity.png')}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
