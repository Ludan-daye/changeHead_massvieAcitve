#!/usr/bin/env python3
"""
Experiment RQ2: MA Source Verification for a specific model
Compare Attention output vs MLP output at a critical layer.
Saves results to results/models/{model}/RQ2_mlp_source/verification.json
"""
import os
import sys
import json
from datetime import datetime
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import lib


def get_attention_module(layer):
    # Try common attribute names across architectures
    for name in ['self_attn', 'self_attention', 'attn', 'attention']:
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def get_mlp_output_module(layer, model_name: str):
    mn = model_name.lower()
    if 'opt' in mn:
        return getattr(layer, 'fc2', None)
    if 'gptj' in mn:
        return getattr(layer.mlp, 'fc_out', None)
    if 'falcon' in mn or 'bloom' in mn:
        return getattr(layer.mlp, 'dense_4h_to_h', None)
    if 'qwen' in mn or 'mistral' in mn:
        return getattr(layer.mlp, 'down_proj', None)
    if 'gpt2' in mn:
        return getattr(layer.mlp, 'c_proj', None)
    # Fallbacks
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
        return layer.mlp.down_proj
    return None


def run_rq2(args):
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()

    layer_id = args.critical_layer
    layer = layers[layer_id]

    attn_mod = get_attention_module(layer)
    mlp_mod = get_mlp_output_module(layer, args.model)

    if attn_mod is None or mlp_mod is None:
        raise RuntimeError(f"Cannot locate modules (attn={attn_mod is not None}, mlp={mlp_mod is not None}) for model {args.model}")

    capture = {'attn': None, 'mlp': None}

    def hook_attn(m, inp, out):
        out0 = out[0] if isinstance(out, (tuple, list)) else out
        capture['attn'] = out0.detach().float().abs().max().item()

    def hook_mlp(m, inp, out):
        out0 = out[0] if isinstance(out, (tuple, list)) else out
        capture['mlp'] = out0.detach().float().abs().max().item()

    h1 = attn_mod.register_forward_hook(hook_attn)
    h2 = mlp_mod.register_forward_hook(hook_mlp)

    attn_maxes = []
    mlp_maxes = []

    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, seqlen=min(seq_len, 2048), device=device)

    with torch.no_grad():
        for testseq in testseq_list:
            capture['attn'] = None
            capture['mlp'] = None
            _ = model(testseq)
            if capture['attn'] is not None:
                attn_maxes.append(capture['attn'])
            if capture['mlp'] is not None:
                mlp_maxes.append(capture['mlp'])

    h1.remove()
    h2.remove()

    attn_max = float(max(attn_maxes)) if attn_maxes else 0.0
    mlp_max = float(max(mlp_maxes)) if mlp_maxes else 0.0
    ratio = float(mlp_max / attn_max) if attn_max > 0 else float('inf' if mlp_max > 0 else 0)

    os.makedirs(args.savedir, exist_ok=True)
    out_path = os.path.join(args.savedir, 'verification.json')
    with open(out_path, 'w') as f:
        json.dump({
            'model': args.model,
            'critical_layer': layer_id,
            'nsamples': args.nsamples,
            'attention_output_max': attn_max,
            'mlp_output_max': mlp_max,
            'ratio': ratio,
            'method': 'hook attn and mlp outputs at critical layer',
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"✓ RQ2 verification saved to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--critical_layer', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()

    run_rq2(args)
