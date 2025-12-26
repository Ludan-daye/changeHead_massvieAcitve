#!/usr/bin/env python3
"""
调试qwen2.5_7b的MA计算问题
对比不同hook方法的结果
"""

import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib

class Args:
    model = 'qwen2.5_7b'

args = Args()

print("="*80)
print("调试 Qwen2.5-7B MA 计算")
print("="*80)

# 加载模型
model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
model.eval()

# 获取测试数据
testseq_list = lib.get_data(tokenizer, nsamples=1, seqlen=2048, device=device)
testseq = testseq_list[0]

print(f"\n模型层数: {len(layers)}")
print(f"测试序列形状: {testseq.shape}")

# 测试不同层
test_layers = [3, 14, 27]  # 前、中、后

for layer_id in test_layers:
    print(f"\n{'='*80}")
    print(f"测试 Layer {layer_id}")
    print(f"{'='*80}")

    layer = layers[layer_id]

    # 方法1：Hook整个layer输出（Exp5当前方法）
    print("\n方法1: Hook整个layer输出")
    activations = {}

    def hook_layer(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        activations['layer'] = out.detach().cpu().float()

    handle = layer.register_forward_hook(hook_layer)

    with torch.no_grad():
        _ = model(testseq)

    handle.remove()

    if 'layer' in activations:
        feat = activations['layer'].numpy()
        max_val = float(np.abs(feat).max())
        print(f"  最大激活值: {max_val:.2f}")
    else:
        print(f"  ❌ 未捕获到激活值")

    # 方法2：Hook MLP输出（Exp1方法）
    print("\n方法2: Hook MLP输出模块")
    capture = {'mlp': None}

    mlp_mod = None
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
        mlp_mod = layer.mlp.down_proj
    elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_proj'):
        mlp_mod = layer.mlp.c_proj

    if mlp_mod:
        def hook_mlp(m, inp, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            capture['mlp'] = out0.detach().float().abs().max().item()

        h = mlp_mod.register_forward_hook(hook_mlp)

        with torch.no_grad():
            _ = model(testseq)

        h.remove()

        if capture['mlp'] is not None:
            print(f"  最大激活值: {capture['mlp']:.2f}")
        else:
            print(f"  ❌ 未捕获到激活值")
    else:
        print(f"  ❌ 无法找到MLP模块")

    # 方法3：Hook MLP中间激活（after activation function）
    print("\n方法3: Hook MLP中间激活 (after act_fn)")
    capture_mid = {'act': None}

    # Qwen使用SwiGLU，需要hook gate_proj/up_proj组合后
    if hasattr(layer.mlp, 'up_proj'):
        print(f"  检测到MLP结构: gate_proj + up_proj + down_proj (SwiGLU)")

        def hook_before_down(m, inp, out):
            # inp是down_proj的输入，即activation后的值
            in0 = inp[0] if isinstance(inp, (tuple, list)) else inp
            capture_mid['act'] = in0.detach().float().abs().max().item()

        h_down = layer.mlp.down_proj.register_forward_hook(hook_before_down)

        with torch.no_grad():
            _ = model(testseq)

        h_down.remove()

        if capture_mid['act'] is not None:
            print(f"  中间激活最大值: {capture_mid['act']:.2f}")
        else:
            print(f"  ❌ 未捕获到中间激活")

print(f"\n{'='*80}")
print("调试完成")
print(f"{'='*80}")
