#!/usr/bin/env python3
"""
为GPT-2和LLaMA2-13B运行Exp4 SVD分析
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
from lib.model_utils import is_llama_model

def get_mlp_weights(layer, model_type):
    """获取MLP输出层权重"""
    if is_llama_model(model_type):
        # LLaMA使用 down_proj
        if hasattr(layer.mlp, 'down_proj'):
            return layer.mlp.down_proj.weight.data.cpu().float().numpy()
    elif "gpt2" in model_type:
        # GPT-2使用 c_proj (注意：GPT-2的weight需要转置)
        if hasattr(layer.mlp, 'c_proj'):
            W = layer.mlp.c_proj.weight.data.cpu().float().numpy()
            # GPT-2的Conv1D层，权重是[in_features, out_features]，需要转置
            return W.T

    raise ValueError(f"Cannot find MLP output weights for model type: {model_type}")

def compute_svd_analysis(model_name, layer_ids):
    """为指定模型运行SVD分析"""

    print("="*80)
    print(f"EXPERIMENT 4: MLP SVD Analysis")
    print(f"Model: {model_name}")
    print(f"Target Layers: {layer_ids}")
    print("="*80)

    # 加载模型
    class Args:
        model = model_name
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
        if lid >= len(layers):
            print(f"Layer {lid} 超出范围，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"Layer {lid}")
        print(f"{'='*60}")

        try:
            layer = layers[lid]
            W = get_mlp_weights(layer, model_name)

            print(f"  MLP output weight shape: {W.shape}")

            # SVD分解
            U, S, Vh = np.linalg.svd(W, full_matrices=False)

            # 保存前20个奇异值
            top_20_sv = S[:20].tolist()
            ratio = float(S[0] / S[1]) if len(S) > 1 else None

            print(f"  Top 5 Singular Values: {S[:5]}")
            print(f"  σ₁/σ₂ Ratio: {ratio:.4f}" if ratio else "  (insufficient singular values)")

            svd_analysis[str(lid)] = {
                'singular_values': top_20_sv,
                'ratio_s1_s2': ratio,
                'shape': list(W.shape)
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            svd_analysis[str(lid)] = {'error': str(e)}

    # 保存结果
    output_name = model_name.replace('_', '')
    if 'llama' in model_name:
        output_name = 'llama2_13b'
    elif 'gpt2' in model_name:
        output_name = 'gpt2'

    savedir = os.path.join(PROJECT_ROOT, f'results/experiments/exp4/{output_name}')
    os.makedirs(savedir, exist_ok=True)

    output_data = {
        'experiment': 'exp4_mlp_svd',
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
        if str(lid) in svd_analysis and 'ratio_s1_s2' in svd_analysis[str(lid)]:
            ratio = svd_analysis[str(lid)]['ratio_s1_s2']
            if ratio:
                dominant = 'Yes' if ratio > 2.0 else 'No'
                print(f"{lid:<8} {ratio:<15.2f} {dominant:<15}")

    # 清理
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    # GPT-2: MA层是Layer 16
    print("\n" + "="*80)
    print("运行 GPT-2 Exp4...")
    print("="*80)
    compute_svd_analysis('gpt2', [0, 1, 2, 8, 15, 16, 17, 23])

    print("\n\n" + "="*80)
    print("运行 LLaMA2-13B Exp4...")
    print("="*80)
    compute_svd_analysis('llama2_13b', [0, 1, 2, 3, 10, 20, 22, 38, 39])
