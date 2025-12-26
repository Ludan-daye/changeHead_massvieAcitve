#!/usr/bin/env python3
"""
Experiment 4: OPT MLP SVD Analysis - Complete Version
保存完整的SVD数据用于图表生成
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

def compute_opt_mlp_svd(layer, layer_id):
    """Compute SVD for OPT MLP fc2 weights"""
    print(f"\nComputing SVD for Layer {layer_id} fc2...")

    # FC2 (Output Projection): [hidden_dim, intermediate_dim]
    if hasattr(layer, 'fc2'):
        W_fc2 = layer.fc2.weight.data.cpu().float().numpy()
        print(f"  fc2 shape: {W_fc2.shape}")

        U_fc2, S_fc2, Vh_fc2 = np.linalg.svd(W_fc2, full_matrices=False)

        # 保存前20个奇异值
        top_20_sv = S_fc2[:20].tolist()
        ratio = float(S_fc2[0] / S_fc2[1])

        print(f"  fc2 Top 5 Singular Values: {S_fc2[:5]}")
        print(f"  fc2 Ratio σ₁/σ₂: {ratio:.4f}")

        result = {
            'singular_values': top_20_sv,
            'sigma1_sigma2_ratio': ratio,
            'shape': list(W_fc2.shape)
        }

        return result

    return None

def main():
    # 配置
    model_name = 'opt_7b'  # 注意：模型id是opt_7b，实际是6.7B参数
    output_name = 'opt_6.7b'  # 输出目录使用6.7b
    # 选择关键层：前几层、中间层、MA层（Layer 25）、最后几层
    layer_ids = [0, 1, 2, 12, 24, 25, 26, 30, 31]

    savedir = os.path.join(PROJECT_ROOT, f'results/experiments/exp4/{output_name}')
    os.makedirs(savedir, exist_ok=True)

    print("="*80)
    print(f"EXPERIMENT 4: OPT MLP SVD Analysis")
    print(f"Model: {model_name}")
    print(f"Target Layers: {layer_ids}")
    print("="*80)

    # Load Model
    class Args:
        model = model_name
        dataset = 'wikitext'
        nsamples = 10
        seed = 0

    args = Args()
    print("\n加载模型...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print("✓ 模型加载完成")

    # Compute SVD for each layer
    svd_data = {}
    svd_analysis = {}

    for lid in layer_ids:
        print(f"\n{'='*60}")
        print(f"Layer {lid}")
        print(f"{'='*60}")

        result = compute_opt_mlp_svd(layers[lid], lid)
        if result:
            svd_analysis[str(lid)] = result

    # 保存结果
    output_data = {
        'experiment': 'exp4_opt_mlp_svd',
        'model': output_name,
        'date': datetime.now().isoformat(),
        'layers_analyzed': layer_ids,
        'svd_analysis': svd_analysis
    }

    output_path = os.path.join(savedir, 'svd_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ 完成！结果已保存至: {output_path}")
    print(f"{'='*80}")

    # 打印摘要
    print("\n摘要:")
    print(f"{'Layer':<8} {'σ1/σ2 Ratio':<15} {'Has Dominant':<15}")
    print("-" * 40)
    for lid in layer_ids:
        if str(lid) in svd_analysis:
            ratio = svd_analysis[str(lid)]['sigma1_sigma2_ratio']
            dominant = 'Yes' if ratio > 2.0 else 'No'
            print(f"{lid:<8} {ratio:<15.2f} {dominant:<15}")

if __name__ == "__main__":
    main()
