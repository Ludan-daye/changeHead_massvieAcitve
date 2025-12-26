#!/usr/bin/env python3
"""
为GPT-J-6B运行完整的Exp4 SVD分析
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib

def get_mlp_weights(layer):
    """获取GPT-J MLP输出层权重"""
    if hasattr(layer.mlp, 'fc_out'):
        # GPT-J使用fc_out作为输出层
        return layer.mlp.fc_out.weight.data.cpu().float().numpy()
    raise ValueError("Cannot find fc_out")

# GPT-J-6B: 28层，MA层是Layer 0，选择代表性层
layer_ids = [0, 1, 2, 7, 13, 20, 26, 27]

print("="*80)
print("EXPERIMENT 4: GPT-J-6B MLP SVD Analysis")
print(f"Target Layers: {layer_ids}")
print("="*80)

# 加载模型
class Args:
    model = 'gptj_6b'
    dataset = 'wikitext'
    nsamples = 10
    seed = 0

args = Args()
print("\n加载模型...")
model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
print(f"✓ 模型加载完成 (层数: {len(layers)}, Hidden Size: {hidden_size})")

# 计算SVD
svd_analysis = {}

for lid in layer_ids:
    print(f"\n{'='*60}")
    print(f"Layer {lid}")
    print(f"{'='*60}")

    try:
        layer = layers[lid]
        W = get_mlp_weights(layer)
        print(f"  MLP output weight shape: {W.shape}")

        # SVD分解
        U, S, Vh = np.linalg.svd(W, full_matrices=False)

        # 保存前20个奇异值
        top_20_sv = S[:20].tolist()
        ratio = float(S[0] / S[1])

        print(f"  Top 5 Singular Values: {S[:5]}")
        print(f"  σ₁/σ₂ Ratio: {ratio:.4f}")

        svd_analysis[str(lid)] = {
            'singular_values': top_20_sv,
            'ratio_s1_s2': ratio,
            'shape': list(W.shape)
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# 保存结果
savedir = os.path.join(PROJECT_ROOT, 'results/experiments/exp4/gptj_6b')
os.makedirs(savedir, exist_ok=True)

output_data = {
    'experiment': 'exp4_mlp_svd',
    'model': 'gptj_6b',
    'date': datetime.now().isoformat(),
    'layers_analyzed': layer_ids,
    'svd_analysis': svd_analysis
}

output_path = os.path.join(savedir, 'svd_analysis.json')
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n{'='*80}")
print(f"✅ GPT-J-6B完成！结果已保存至: {output_path}")
print(f"{'='*80}")

# 打印摘要
print("\nGPT-J-6B 摘要:")
print(f"{'Layer':<8} {'σ1/σ2 Ratio':<15} {'Has Dominant':<15}")
print("-" * 40)
for lid in layer_ids:
    if str(lid) in svd_analysis and 'ratio_s1_s2' in svd_analysis[str(lid)]:
        ratio = svd_analysis[str(lid)]['ratio_s1_s2']
        dominant = 'Yes' if ratio > 2.0 else 'No'
        print(f"{lid:<8} {ratio:<15.2f} {dominant:<15}")
