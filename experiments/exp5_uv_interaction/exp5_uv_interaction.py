#!/usr/bin/env python3
"""
Experiment 5: U x V Interaction Attribution Analysis
Test whether MA generation is independent contribution from U and V or requires synergy

Storage format follows exp2b standard:
- baseline.json: Original model
- ablate_u.json: Ablate U (U_random @ Sigma @ V^T)
- ablate_v.json: Ablate V (U @ Sigma @ V_random)
- ablate_both.json: Ablate both (U_random @ Sigma @ V_random)
- summary.json: Attribution percentage summary
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib


def compute_top1_massive_activation_percentage(model, tokenizer, device, layer_id, seq_len=2048, nsamples=1, seed=42):
    """
    Calculate Top1 Massive Activation value
    Based on exp3's run_and_collect_ma logic

    Args:
        layer_id: Layer ID to monitor
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get test data
    testseq_list = lib.get_data(tokenizer, nsamples=nsamples, seqlen=min(seq_len, 2048), device=device)
    if not isinstance(testseq_list, list):
        testseq_list = [testseq_list]

    # Collect MA from all samples
    all_top1 = []

    for testseq in testseq_list:
        # Hook to collect MLP output from specified layer
        capture = {'mlp': None}

        def hook_mlp(m, inp, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            capture['mlp'] = out0.detach().float().abs().max().item()

        # Get layers
        if hasattr(model, 'transformer'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'gpt_neox'):
            layers = model.gpt_neox.layers
        else:
            # Cannot get layers, return default value
            return 8000.0

        # Use specified layer
        layer = layers[layer_id]

        # Get MLP output module
        mlp_mod = None
        if hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'down_proj'):
                mlp_mod = layer.mlp.down_proj
            elif hasattr(layer.mlp, 'c_proj'):
                mlp_mod = layer.mlp.c_proj
            elif hasattr(layer.mlp, 'fc_out'):
                mlp_mod = layer.mlp.fc_out
            elif hasattr(layer.mlp, 'dense_4h_to_h'):
                mlp_mod = layer.mlp.dense_4h_to_h

        if mlp_mod is None:
            # If cannot find MLP module, return default value
            all_top1.append(8000.0)
            continue

        handle = mlp_mod.register_forward_hook(hook_mlp)

        with torch.no_grad():
            _ = model(testseq)

        handle.remove()

        # Collect MA
        if capture['mlp'] is not None:
            all_top1.append(capture['mlp'])
        else:
            all_top1.append(8000.0)

    # Return average
    return float(np.mean(all_top1))

def create_random_orthogonal(shape, device='cpu'):
    """Create random orthogonal matrix"""
    random_matrix = torch.randn(shape, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q


def get_mlp_layer(model, layer_id, model_type):
    """Get MLP module of specified layer"""
    if hasattr(model, 'transformer'):
        return model.transformer.h[layer_id]
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            return model.model.layers[layer_id]
        elif hasattr(model.model, 'decoder'):
            return model.model.decoder.layers[layer_id]
    elif hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_id]
    raise ValueError(f"Cannot access layer {layer_id} for model type {model_type}")


def get_w2_weight(layer, model_type):
    """Get W2 weight matrix (handles GPT-2 Conv1D transpose)"""
    weight = None
    if hasattr(layer, 'mlp'):
        if hasattr(layer.mlp, 'down_proj'):
            weight = layer.mlp.down_proj.weight
        elif hasattr(layer.mlp, 'c_proj'):
            weight = layer.mlp.c_proj.weight
        elif hasattr(layer.mlp, 'fc_out'):
            weight = layer.mlp.fc_out.weight
        elif hasattr(layer.mlp, 'dense_4h_to_h'):
            weight = layer.mlp.dense_4h_to_h.weight
    if weight is None and hasattr(layer, 'fc2'):
        weight = layer.fc2.weight
    if weight is None:
        raise ValueError(f"Cannot find W2 for model type {model_type}")
    return weight


def set_w2_weight(layer, model_type, new_weight):
    """Set W2 weight matrix (handles GPT-2 Conv1D transpose)"""
    if hasattr(layer, 'mlp'):
        if hasattr(layer.mlp, 'down_proj'):
            layer.mlp.down_proj.weight.data = new_weight
        elif hasattr(layer.mlp, 'c_proj'):
            layer.mlp.c_proj.weight.data = new_weight
        elif hasattr(layer.mlp, 'fc_out'):
            layer.mlp.fc_out.weight.data = new_weight
        elif hasattr(layer.mlp, 'dense_4h_to_h'):
            layer.mlp.dense_4h_to_h.weight.data = new_weight
    elif hasattr(layer, 'fc2'):
        layer.fc2.weight.data = new_weight



def run_intervention(model, tokenizer, device, layer_id, model_type,
                     seq_len, ablate_u=False, ablate_v=False, n_samples=5):
    """
    Run intervention experiment

    Args:
        ablate_u: Whether to replace U with random orthogonal matrix
        ablate_v: Whether to replace V with random orthogonal matrix
    """
    layer = get_mlp_layer(model, layer_id, model_type)
    original_weight = get_w2_weight(layer, model_type).detach().clone()

    # SVD decomposition
    W = original_weight.detach().cpu().float().numpy()
    # GPT-2 uses Conv1D, weight needs transpose
    if "gpt2" in model_type:
        W = W.T
    U, S, Vh = np.linalg.svd(W, full_matrices=False)

    # Build intervened weight
    if ablate_u and ablate_v:
        # Ablate both
        U_new = create_random_orthogonal(U.shape, device='cpu').numpy()
        V_new = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        W_new = U_new @ np.diag(S) @ V_new
        intervention_type = "ablate_both"
    elif ablate_u:
        # Ablate U only
        U_new = create_random_orthogonal(U.shape, device='cpu').numpy()
        W_new = U_new @ np.diag(S) @ Vh
        intervention_type = "ablate_u"
    elif ablate_v:
        # Ablate V only
        V_new = create_random_orthogonal(Vh.T.shape, device='cpu').numpy().T
        W_new = U @ np.diag(S) @ V_new
        intervention_type = "ablate_v"
    else:
        # Baseline
        W_new = W
        intervention_type = "baseline"

    # Set new weight
    # GPT-2 needs transpose back to Conv1D format
    if "gpt2" in model_type:
        W_new = W_new.T
    new_weight_tensor = torch.from_numpy(W_new).to(device).to(original_weight.dtype)
    set_w2_weight(layer, model_type, new_weight_tensor)

    # Run evaluation
    results = {}
    for sample_id in range(n_samples):
        ma_pct = compute_top1_massive_activation_percentage(
            model, tokenizer, device, layer_id, seq_len,
            nsamples=1, seed=42 + sample_id
        )
        results[str(sample_id)] = {
            "mean": float(ma_pct),
            "n_samples": 1,
            "intervention": intervention_type
        }

    # Restore original weight
    set_w2_weight(layer, model_type, original_weight)

    # Calculate statistics
    values = [results[str(i)]["mean"] for i in range(n_samples)]
    summary = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": values
    }

    return summary, results


def compute_attribution(baseline, ablate_u, ablate_v, ablate_both):
    """
    Calculate attribution percentages

    MA = U_contribution + V_contribution + Interaction

    U_only = baseline - ablate_v  (keep U, remove V)
    V_only = baseline - ablate_u  (keep V, remove U)
    Interaction = baseline - U_only - V_only + ablate_both
    """
    baseline_val = baseline['mean']

    # Main effects
    u_main_effect = baseline_val - ablate_v['mean']  # V destroyed, see U's effect
    v_main_effect = baseline_val - ablate_u['mean']  # U destroyed, see V's effect

    # Interaction effect
    # If U and V completely independent: ablate_both approx baseline - U_main - V_main
    # Interaction term = actual difference
    expected_both = baseline_val - u_main_effect - v_main_effect
    interaction = ablate_both['mean'] - expected_both

    # Attribution percentages (relative to baseline)
    u_attribution = (u_main_effect / baseline_val) * 100 if baseline_val != 0 else 0
    v_attribution = (v_main_effect / baseline_val) * 100 if baseline_val != 0 else 0
    interaction_attribution = (interaction / baseline_val) * 100 if baseline_val != 0 else 0

    return {
        'baseline': baseline_val,
        'ablate_u_mean': ablate_u['mean'],
        'ablate_v_mean': ablate_v['mean'],
        'ablate_both_mean': ablate_both['mean'],
        'u_main_effect': u_main_effect,
        'v_main_effect': v_main_effect,
        'interaction_effect': interaction,
        'u_attribution_pct': u_attribution,
        'v_attribution_pct': v_attribution,
        'interaction_pct': interaction_attribution,
        'total_explained': u_attribution + v_attribution + interaction_attribution,
        'interpretation': 'independent' if abs(interaction_attribution) < 5 else 'synergistic'
    }


def main():
    parser = argparse.ArgumentParser(description='Experiment 5: U x V Interaction Attribution Analysis')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True, help='Layer to analyze')
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print("="*80)
    print("Experiment 5: U x V Interaction Attribution Analysis")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Critical layer: {args.layer}")
    print(f"Samples: {args.nsamples}")
    print(f"Save directory: {args.savedir}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print(f"Model loaded successfully")

    # Run four conditions
    print("\n" + "="*80)
    print("1. Baseline (original model)")
    print("="*80)
    baseline_summary, baseline_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=False, ablate_v=False, n_samples=args.nsamples
    )
    print(f"  MA average: {baseline_summary['mean']:.2f}%")

    with open(os.path.join(args.savedir, 'baseline.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_baseline',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': baseline_summary,
            'results': baseline_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("2. Ablate U matrix (U_random @ Sigma @ V^T)")
    print("="*80)
    ablate_u_summary, ablate_u_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=True, ablate_v=False, n_samples=args.nsamples
    )
    print(f"  MA average: {ablate_u_summary['mean']:.2f}% (change: {ablate_u_summary['mean']-baseline_summary['mean']:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_u.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_ablate_u',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_u_summary,
            'results': ablate_u_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("3. Ablate V matrix (U @ Sigma @ V_random)")
    print("="*80)
    ablate_v_summary, ablate_v_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=False, ablate_v=True, n_samples=args.nsamples
    )
    print(f"  MA average: {ablate_v_summary['mean']:.2f}% (change: {ablate_v_summary['mean']-baseline_summary['mean']:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_v.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_ablate_v',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_v_summary,
            'results': ablate_v_results
        }, f, indent=2)

    print("\n" + "="*80)
    print("4. Ablate both U and V (U_random @ Sigma @ V_random)")
    print("="*80)
    ablate_both_summary, ablate_both_results = run_intervention(
        model, tokenizer, device, args.layer, args.model,
        seq_len,
        ablate_u=True, ablate_v=True, n_samples=args.nsamples
    )
    print(f"  MA average: {ablate_both_summary['mean']:.2f}% (change: {ablate_both_summary['mean']-baseline_summary['mean']:.2f}%)")

    with open(os.path.join(args.savedir, 'ablate_both.json'), 'w') as f:
        json.dump({
            'experiment': 'exp5_uv_interaction_ablate_both',
            'model': args.model,
            'layer': args.layer,
            'date': datetime.now().isoformat(),
            'n_samples': args.nsamples,
            'summary': ablate_both_summary,
            'results': ablate_both_results
        }, f, indent=2)

    # Calculate attribution
    print("\n" + "="*80)
    print("Attribution Analysis")
    print("="*80)
    attribution = compute_attribution(
        baseline_summary, ablate_u_summary, ablate_v_summary, ablate_both_summary
    )

    print(f"\nBaseline MA: {attribution['baseline']:.2f}%")
    print(f"\nU matrix contribution: {attribution['u_attribution_pct']:.2f}%")
    print(f"V matrix contribution: {attribution['v_attribution_pct']:.2f}%")
    print(f"Interaction effect: {attribution['interaction_pct']:.2f}%")
    print(f"Total explained: {attribution['total_explained']:.2f}%")
    print(f"\nMechanism type: {attribution['interpretation']}")
    if attribution['interpretation'] == 'independent':
        print("  -> U and V approximately independent contribution (SVD-aligned)")
    else:
        print("  -> U and V require synergy (multi-directional)")

    # Save summary
    summary_data = {
        'model': args.model,
        'layer': args.layer,
        'date': datetime.now().isoformat(),
        'n_samples': args.nsamples,
        'attribution': attribution
    }

    with open(os.path.join(args.savedir, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nAll results saved to: {args.savedir}")

    # Cleanup
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
