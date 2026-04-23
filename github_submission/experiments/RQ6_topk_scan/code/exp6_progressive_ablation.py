#!/usr/bin/env python3
"""
RQ6: Progressive Greedy Layer Ablation - How many layers must be disabled to kill MA?

Greedy strategy:
    S = {}                      # set of disabled layers
    while top1 > threshold:
        for each L not in S:
            test ablating S ∪ {L}, measure top1
        add to S the L that causes biggest top1 drop
    return S, the trajectory of top1 vs |S|

This distinguishes:
  - Concentrated single-source: |S| = 1 already kills MA (drops to floor)
  - Few-source backup: |S| = 2-3 needed; each removal causes huge drop
  - Truly dispersed: many layers needed; each removal causes small drop
"""

import os, sys, argparse, json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib
import monkey_patch as mp


class MLPZeroHook:
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


def measure_top1(args, layers, model, tokenizer, device, seq_len, disabled_set):
    """Run with given set of MLP layers disabled, return mean top1 across sequences."""
    hooks = []
    for lid in disabled_set:
        mlp = getattr(layers[lid], 'mlp', None)
        if mlp is not None:
            h = mlp.register_forward_hook(MLPZeroHook())
            hooks.append(h)

    # Use the same data each call (cache outside if possible — simpler: re-load)
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=seq_len, device=device)
    top1_list = []
    with torch.no_grad():
        for seq in testseq_list:
            model(seq.to(device))
            peak = 0.0
            for lid in range(len(layers)):
                feat = getattr(layers[lid], 'feat', None)
                if feat is not None:
                    val = feat.detach().abs().max().item()
                    if val > peak:
                        peak = val
            top1_list.append(peak)

    for h in hooks:
        h.remove()

    return float(np.mean(top1_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nsamples', type=int, default=10,
                        help='Few samples to keep greedy fast')
    parser.add_argument('--seqlen', type=int, default=1024)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--access_token', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--threshold_pct', type=float, default=10.0,
                        help='Stop when top1 drops below this fraction of baseline (e.g., 10%)')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Hard cap on greedy steps')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"RQ6: PROGRESSIVE GREEDY ABLATION - {args.model}")
    print(f"{'='*80}")

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    for lid in range(len(layers)):
        lib.enable_custom_block(args.model, layers[lid], lid)
    n_layers = len(layers)
    print(f"\nModel has {n_layers} layers")

    # Baseline
    print("\n[1] Baseline (no MLP disabled)...")
    baseline_top1 = measure_top1(args, layers, model, tokenizer, device, seq_len, [])
    print(f"  baseline top1 = {baseline_top1:.1f}")

    # Floor
    print("\n[2] All-MLP-disabled (floor)...")
    floor_top1 = measure_top1(args, layers, model, tokenizer, device, seq_len, list(range(n_layers)))
    print(f"  floor top1 = {floor_top1:.1f}")
    print(f"  total room = {baseline_top1 - floor_top1:.1f}")

    threshold = floor_top1 + (baseline_top1 - floor_top1) * (args.threshold_pct / 100)
    print(f"  stop threshold = {threshold:.1f} ({args.threshold_pct}% above floor)")

    # Greedy
    print(f"\n[3] Greedy progressive ablation (max {args.max_steps} steps)...")
    disabled = []
    trajectory = [{'step': 0, 'disabled': [], 'top1': baseline_top1, 'remaining_pct': 100.0}]
    current_top1 = baseline_top1
    candidates = set(range(n_layers))

    for step in range(1, args.max_steps + 1):
        if not candidates:
            print(f"\n  All {n_layers} layers disabled — stopping.")
            break
        if current_top1 <= threshold:
            print(f"\n  Top1 ({current_top1:.1f}) below threshold ({threshold:.1f}) — stopping.")
            break

        print(f"\n  Step {step}: testing {len(candidates)} candidates "
              f"(currently disabled: {disabled})")

        best_lid = None
        best_top1 = float('inf')
        per_candidate = {}

        for lid in tqdm(sorted(candidates), desc=f'Step {step}'):
            t = measure_top1(args, layers, model, tokenizer, device, seq_len,
                            disabled + [lid])
            per_candidate[lid] = t
            if t < best_top1:
                best_top1 = t
                best_lid = lid

        disabled.append(best_lid)
        candidates.remove(best_lid)
        current_top1 = best_top1
        remaining_pct = ((current_top1 - floor_top1) /
                         (baseline_top1 - floor_top1) * 100) if baseline_top1 > floor_top1 else 0
        delta_pct = (1 - current_top1 / baseline_top1) * 100

        print(f"\n  >>> Step {step}: add L{best_lid}, "
              f"top1={current_top1:.1f}, remaining={remaining_pct:.1f}%, "
              f"total drop={delta_pct:.1f}%")

        trajectory.append({
            'step': step,
            'disabled': list(disabled),
            'added_layer': int(best_lid),
            'top1': current_top1,
            'remaining_pct': remaining_pct,
            'total_drop_pct': delta_pct,
            'per_candidate': {int(k): float(v) for k, v in per_candidate.items()},
        })

    # Save
    out = {
        'model': args.model,
        'n_layers': n_layers,
        'timestamp': datetime.now().isoformat(),
        'baseline_top1': baseline_top1,
        'floor_top1': floor_top1,
        'threshold': threshold,
        'final_disabled_set': list(disabled),
        'final_top1': current_top1,
        'steps_to_kill': len(disabled),
        'trajectory': trajectory,
    }

    out_path = os.path.join(args.savedir, f'{args.model}_rq6_greedy.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*80}")
    print(f"  Steps to drop MA below {args.threshold_pct}%: {len(disabled)}")
    print(f"  Disabled layers (in order): {disabled}")
    print(f"  Final top1: {current_top1:.1f} (baseline {baseline_top1:.1f})")
    if len(disabled) == 1:
        print("  >>> CONCENTRATED: single layer suffices")
    elif len(disabled) <= 3:
        print(f"  >>> FEW-SOURCE: {len(disabled)} layers needed")
    else:
        print(f"  >>> DISPERSED: {len(disabled)} layers needed (multi-source)")
    print(f"\n  Saved to: {out_path}")


if __name__ == '__main__':
    main()
