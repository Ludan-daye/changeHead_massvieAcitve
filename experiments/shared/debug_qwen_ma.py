#!/usr/bin/env python3
"""
Debug Qwen2.5_7b MA calculation issues
Compare results from different hook methods
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
print("Debug Qwen2.5-7B MA Calculation")
print("="*80)

# Load model
model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
model.eval()

# Get test data
testseq_list = lib.get_data(tokenizer, nsamples=1, seqlen=2048, device=device)
testseq = testseq_list[0]

print(f"\nModel layer count: {len(layers)}")
print(f"Test sequence shape: {testseq.shape}")

# Test different layers
test_layers = [3, 14, 27]  # Early, middle, late

for layer_id in test_layers:
    print(f"\n{'='*80}")
    print(f"Testing Layer {layer_id}")
    print(f"{'='*80}")

    layer = layers[layer_id]

    # Method 1: Hook entire layer output (current Exp5 method)
    print("\nMethod 1: Hook entire layer output")
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
        print(f"  Maximum activation value: {max_val:.2f}")
    else:
        print(f"  Failed to capture activation")

    # Method 2: Hook MLP output (Exp1 method)
    print("\nMethod 2: Hook MLP output module")
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
            print(f"  Maximum activation value: {capture['mlp']:.2f}")
        else:
            print(f"  Failed to capture activation")
    else:
        print(f"  Cannot find MLP module")

    # Method 3: Hook MLP intermediate activation (after activation function)
    print("\nMethod 3: Hook MLP intermediate activation (after act_fn)")
    capture_mid = {'act': None}

    # Qwen uses SwiGLU, need to hook after gate_proj/up_proj combination
    if hasattr(layer.mlp, 'up_proj'):
        print(f"  Detected MLP structure: gate_proj + up_proj + down_proj (SwiGLU)")

        def hook_before_down(m, inp, out):
            # inp is down_proj input, i.e., value after activation
            in0 = inp[0] if isinstance(inp, (tuple, list)) else inp
            capture_mid['act'] = in0.detach().float().abs().max().item()

        h_down = layer.mlp.down_proj.register_forward_hook(hook_before_down)

        with torch.no_grad():
            _ = model(testseq)

        h_down.remove()

        if capture_mid['act'] is not None:
            print(f"  Intermediate activation maximum value: {capture_mid['act']:.2f}")
        else:
            print(f"  Failed to capture intermediate activation")

print(f"\n{'='*80}")
print("Debug Complete")
print(f"{'='*80}")
