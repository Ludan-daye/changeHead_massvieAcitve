#!/usr/bin/env python3
"""
Worker script for single layer restoration
Only responsible for testing a single layer, disposable after use to ensure no memory leaks
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm

# Add path to import lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import lib
import monkey_patch as mp_patch
from lib.model_utils import is_llama_model

class SelectiveHeadDisableHook:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt_7b')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--savedir', type=str, default='results/exp2_opt_6.7b/')
    args = parser.parse_args()

    print(f"Worker started for Layer {args.layer}")
    
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # Load model
    class ModelArgs:
        def __init__(self):
            self.model = args.model
            self.dataset = args.dataset
            self.nsamples = args.nsamples
            self.access_token = 'type in your access token here'
            
    model_args = ModelArgs()
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(model_args)

    # Enable Feature Capture
    if "opt" in args.model:
        for i, layer in enumerate(layers):
            mp_patch.enable_opt_custom_decoderlayer(layer, i)
    elif is_llama_model(args.model):
        for i, layer in enumerate(layers):
            mp_patch.enable_llama_custom_decoderlayer(layer, i)

    # Register Hook
    if hasattr(layers[0].self_attn, 'num_heads'):
        num_heads = layers[0].self_attn.num_heads
    else:
        num_heads = layers[0].self_attn.num_attention_heads

    hooks = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'self_attn'):
            hook = layer.self_attn.register_forward_hook(
                SelectiveHeadDisableHook(i, num_heads, args.layer)
            )
            hooks.append(hook)

    # Load data
    print(f"Layer {args.layer}: Loading data...")
    dataloader = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=seq_len, device=device)

    # Collect data
    layer_stats = {i: {'top1_values': [], 'median_values': [], 
                       'dim_138_values': [], 'dim_447_values': []} 
                   for i in range(len(layers))}

    print(f"Layer {args.layer}: Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, tuple):
                testseq = batch[0]
            else:
                testseq = batch
            testseq = testseq.to(device)
            _ = model(testseq)

            for i, layer in enumerate(layers):
                if hasattr(layer, 'feat'):
                    feat = layer.feat.float() # Convert to float to save memory
                    top1 = torch.topk(feat.abs().flatten(), k=1)[0].mean().item()
                    median = torch.median(feat.abs()).item()
                    
                    layer_stats[i]['top1_values'].append(top1)
                    layer_stats[i]['median_values'].append(median)

    # Save results
    results = {}
    for i in range(len(layers)):
        results[i] = {
            'top1_mean': np.mean(layer_stats[i]['top1_values']),
            'top1_std': np.std(layer_stats[i]['top1_values']),
            'median_mean': np.mean(layer_stats[i]['median_values']),
        }

    os.makedirs(args.savedir, exist_ok=True)
    output_file = os.path.join(args.savedir, f'layer_{args.layer}_results.json')
    
    # Convert numpy float
    def convert(o):
        if isinstance(o, (np.float32, np.float64)): return float(o)
        raise TypeError
        
    with open(output_file, 'w') as f:
        json.dump(results, f, default=convert, indent=2)

    print(f"✅ Layer {args.layer} Done!")
    
    # Explicitly delete
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
