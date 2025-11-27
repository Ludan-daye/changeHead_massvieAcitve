#!/usr/bin/env python3
"""
实验2: 多线程层分析 - 找出哪个层产生massive activation
并行测试每一层恢复后的激活值变化
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import lib
import monkey_patch as mp
from lib.model_utils import is_llama_model


class SelectiveHeadDisableHook:
    """禁用所有头，除了指定层"""
    def __init__(self, layer_id, num_heads, restore_layer_id):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.restore_layer_id = restore_layer_id

    def __call__(self, module, input, output):
        if self.layer_id == self.restore_layer_id:
            return output
        
        attn_output = output[0]
        batch_size, seq_len, hidden_dim = attn_output.shape
        head_dim = hidden_dim // self.num_heads
        attn_output_reshaped = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
        attn_output_reshaped[:, :, :, :] = 0
        modified_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
        return (modified_output,) + output[1:]


def get_model_config(model_name):
    """获取模型配置"""
    if "qwen" in model_name:
        return {"patch_func": "enable_qwen_custom_decoderlayer", "attn_attr": "self_attn", "heads_attr": "num_attention_heads"}
    elif is_llama_model(model_name) or "deepseek" in model_name:
        return {"patch_func": "enable_llama_custom_decoderlayer", "attn_attr": "self_attn", "heads_attr": "num_attention_heads"}
    elif "opt" in model_name:
        return {"patch_func": "enable_opt_custom_decoderlayer", "attn_attr": "self_attn", "heads_attr": "num_attention_heads"}
    elif "gpt2" in model_name or "gptj" in model_name:
        return {"patch_func": "enable_gpt2_custom_block", "attn_attr": "attn", "heads_attr": "n_head"}
    elif "bloom" in model_name:
        return {"patch_func": "enable_bloom_custom_block", "attn_attr": "self_attention", "heads_attr": "n_head"}
    elif "falcon" in model_name:
        return {"patch_func": "enable_gpt2_custom_block", "attn_attr": "self_attention", "heads_attr": "num_attention_heads"}
    else:
        raise ValueError(f"Model {model_name} not supported")


def run_layer_test(args, restore_layer_id):
    """测试恢复单层注意力后的激活值"""
    print(f"  Testing Layer {restore_layer_id}...")
    
    # 禁用代理
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ.pop(key, None)
    
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    
    config = get_model_config(args.model)
    patch_func = getattr(mp, config["patch_func"])
    
    # Enable feature capture
    for layer_id in range(len(layers)):
        patch_func(layers[layer_id], layer_id)
    
    # Register hooks
    hooks = []
    num_heads = getattr(model.config, config["heads_attr"])
    
    for layer_id in range(len(layers)):
        layer = layers[layer_id]
        target_module = getattr(layer, config["attn_attr"])
        hook = SelectiveHeadDisableHook(layer_id, num_heads, restore_layer_id)
        handle = target_module.register_forward_hook(hook)
        hooks.append(handle)
    
    # Load data
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=min(seq_len, 2048), device=device)
    
    # Process
    top1_values = []
    with torch.no_grad():
        for testseq in testseq_list:
            _ = model(testseq)
            
            # 取最后一层的激活
            last_layer = layers[-2] if len(layers) > 1 else layers[-1]
            if hasattr(last_layer, 'feat') and last_layer.feat is not None:
                feat_abs = last_layer.feat.abs()
                if len(feat_abs.shape) == 3:
                    feat_abs = feat_abs.view(-1, feat_abs.shape[-1])
                top1 = feat_abs.max().item()
                top1_values.append(top1)
    
    # Cleanup
    for h in hooks:
        h.remove()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    avg_top1 = np.mean(top1_values) if top1_values else 0
    return restore_layer_id, avg_top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2.5_7b')
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1, help='并行进程数(建议1，因为GPU共享)')
    parser.add_argument('--savedir', type=str, default='results/models/qwen2.5_7b/exp2')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    print("="*60)
    print("实验2: 层分析 - 找出massive activation来源层")
    print(f"模型: {args.model}")
    print(f"样本数: {args.nsamples}")
    print("="*60)
    
    # 先获取层数
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ.pop(key, None)
    
    print("\n加载模型获取层数...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    n_layers = len(layers)
    print(f"模型共 {n_layers} 层")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # 测试每一层
    results = {}
    
    print(f"\n开始测试 {n_layers} 层...")
    for layer_id in tqdm(range(n_layers), desc="Layer Progress"):
        lid, top1 = run_layer_test(args, layer_id)
        results[lid] = top1
        print(f"  Layer {lid}: Top1 = {top1:.2f}")
    
    # 保存结果
    results_file = os.path.join(args.savedir, "layer_contribution.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    layers_list = sorted(results.keys())
    values = [results[l] for l in layers_list]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(layers_list, values, color='steelblue')
    
    # 标记最大值
    max_idx = np.argmax(values)
    bars[max_idx].set_color('red')
    
    plt.xlabel('Layer ID')
    plt.ylabel('Average Top1 Activation')
    plt.title(f'{args.model} - Layer Contribution to Massive Activation\n(Red = Max Contribution Layer)')
    plt.xticks(layers_list)
    plt.tight_layout()
    
    fig_file = os.path.join(args.savedir, "layer_contribution.png")
    plt.savefig(fig_file, dpi=150)
    plt.close()
    
    # 打印总结
    print("\n" + "="*60)
    print("实验2完成!")
    print("="*60)
    print(f"\n关键发现:")
    sorted_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"  最大贡献层: Layer {sorted_layers[0][0]} (Top1 = {sorted_layers[0][1]:.2f})")
    print(f"  Top 3层:")
    for lid, val in sorted_layers[:3]:
        print(f"    - Layer {lid}: {val:.2f}")
    
    print(f"\n结果已保存至: {args.savedir}")

if __name__ == "__main__":
    main()
