#!/usr/bin/env python3
"""
实验2: 快速层分析 - 找出哪个层产生massive activation
只加载一次模型，动态切换hook
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import gc
from datetime import datetime
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# 禁用代理
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib
import monkey_patch as mp
from monkey_patch import modify_falcon as falcon_mp
from monkey_patch import modify_gptj as gptj_mp
from lib.model_utils import is_llama_model


class DynamicHeadDisableHook:
    """动态禁用头的Hook，可以运行时切换恢复的层"""
    def __init__(self, layer_id, num_heads):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.restore_layer_id = -1  # -1表示禁用所有层

    def set_restore_layer(self, restore_layer_id):
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
    elif is_llama_model(model_name) or "deepseek" in model_name or "mistral" in model_name:
        return {"patch_func": "enable_llama_custom_decoderlayer", "attn_attr": "self_attn", "heads_attr": "num_attention_heads"}
    elif "opt" in model_name:
        return {"patch_func": "enable_opt_custom_decoderlayer", "attn_attr": "self_attn", "heads_attr": "num_attention_heads"}
    elif "gpt2" in model_name:
        return {"patch_func": "enable_gpt2_custom_block", "attn_attr": "attn", "heads_attr": "n_head"}
    elif "gptj" in model_name:
        return {"patch_func": "enable_gptj_custom_block", "attn_attr": "attn", "heads_attr": "n_head"}
    elif "bloom" in model_name:
        return {"patch_func": "enable_bloom_custom_block", "attn_attr": "self_attention", "heads_attr": "n_head"}
    elif "falcon" in model_name:
        return {"patch_func": "enable_falcon_custom_decoderlayer", "attn_attr": "self_attention", "heads_attr": "num_attention_heads"}
    else:
        raise ValueError(f"Model {model_name} not supported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2.5_7b')
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--savedir', type=str, default='results/models/qwen2.5_7b/exp2')
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    print("="*60)
    print("实验2: 快速层分析 - 找出massive activation来源层")
    print(f"模型: {args.model}")
    print(f"样本数: {args.nsamples}")
    print("="*60)
    
    # 加载模型（只加载一次）
    print("\n加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    n_layers = len(layers)
    print(f"模型共 {n_layers} 层")
    
    config = get_model_config(args.model)
    if "falcon" in args.model:
        patch_func = falcon_mp.enable_falcon_custom_decoderlayer
    elif "gptj" in args.model:
        patch_func = gptj_mp.enable_gptj_custom_block
    else:
        patch_func = getattr(mp, config["patch_func"])
    num_heads = getattr(model.config, config["heads_attr"])
    
    # Enable feature capture
    for layer_id in range(n_layers):
        patch_func(layers[layer_id], layer_id)
    
    # 创建动态Hook
    hooks_obj = []
    handles = []
    for layer_id in range(n_layers):
        layer = layers[layer_id]
        target_module = getattr(layer, config["attn_attr"])
        hook = DynamicHeadDisableHook(layer_id, num_heads)
        handle = target_module.register_forward_hook(hook)
        hooks_obj.append(hook)
        handles.append(handle)
    
    # 加载数据
    print("\n加载数据...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=min(seq_len, 2048), device=device)
    print(f"数据加载完成: {len(testseq_list)} 样本")
    
    # 测试每一层
    results = {}
    
    print(f"\n开始测试 {n_layers} 层...")
    for restore_layer_id in tqdm(range(n_layers), desc="Testing layers"):
        # 设置当前恢复的层
        for hook in hooks_obj:
            hook.set_restore_layer(restore_layer_id)
        
        top1_values = []
        with torch.no_grad():
            for testseq in testseq_list:
                _ = model(testseq)
                
                # 取倒数第二层的激活（最后一层通常有特殊处理）
                target_layer = layers[-2] if n_layers > 1 else layers[-1]
                if hasattr(target_layer, 'feat') and target_layer.feat is not None:
                    feat_abs = target_layer.feat.abs()
                    if len(feat_abs.shape) == 3:
                        feat_abs = feat_abs.view(-1, feat_abs.shape[-1])
                    top1 = feat_abs.max().item()
                    top1_values.append(top1)
        
        avg_top1 = np.mean(top1_values) if top1_values else 0
        results[restore_layer_id] = avg_top1
        
        if restore_layer_id % 5 == 0 or restore_layer_id == n_layers - 1:
            print(f"  Layer {restore_layer_id}: Top1 = {avg_top1:.2f}")
    
    # 清理
    for h in handles:
        h.remove()
    
    # 保存结果
    results_file = os.path.join(args.savedir, "layer_contribution.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    layers_list = sorted(results.keys())
    values = [results[l] for l in layers_list]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(layers_list, values, color='steelblue')
    
    # 标记最大值和最小值
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    bars[max_idx].set_color('red')
    bars[min_idx].set_color('green')
    
    plt.xlabel('Layer ID (Restored Layer)')
    plt.ylabel('Average Top1 Activation')
    plt.title(f'{args.model} - Layer Contribution Analysis\n(Red=Max, Green=Min)')
    plt.xticks(layers_list)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    fig_file = os.path.join(args.savedir, "layer_contribution.png")
    plt.savefig(fig_file, dpi=150)
    plt.close()
    
    # 打印总结
    print("\n" + "="*60)
    print("实验2完成!")
    print("="*60)
    
    sorted_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔥 最大贡献层 (恢复后激活最高):")
    for lid, val in sorted_layers[:5]:
        print(f"    Layer {lid}: {val:.2f}")
    
    print(f"\n🧊 最小贡献层 (恢复后激活最低):")
    for lid, val in sorted_layers[-5:]:
        print(f"    Layer {lid}: {val:.2f}")
    
    print(f"\n📊 结果已保存至: {args.savedir}")
    print(f"   - layer_contribution.json")
    print(f"   - layer_contribution.png")

if __name__ == "__main__":
    main()
