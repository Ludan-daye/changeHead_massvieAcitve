#!/usr/bin/env python3
"""
RQ6 Exhaustive: try all 2^N MLP-disable subsets for small models (GPT-2: N=12).

For each subset S ⊆ {0,...,N-1}, disable MLPs in S, measure mean top1 MA.
Output:
  - per-subset top1
  - per-k: min top1 achievable, optimal subset of size k
  - trajectory: how MA drops as we move from min for k to min for k+1

This gives the ground-truth answer to "minimum k layers to kill MA".
"""

import os, sys, argparse, json, itertools, time
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import lib


class MLPZeroHook:
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


def measure_top1(layers, model, testseq_list, device, disabled_set):
    hooks = []
    for lid in disabled_set:
        mlp = getattr(layers[lid], 'mlp', None)
        if mlp is not None:
            h = mlp.register_forward_hook(MLPZeroHook())
            hooks.append(h)

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
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--nsamples', type=int, default=10,
                        help='Few samples for speed; final config can use more')
    parser.add_argument('--seqlen', type=int, default=1024)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--access_token', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--max_k', type=int, default=-1,
                        help='Stop enumeration at subsets of size max_k (-1 = all)')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"RQ6 EXHAUSTIVE: all-subsets MLP ablation - {args.model}")
    print(f"{'='*80}")

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    for lid in range(len(layers)):
        lib.enable_custom_block(args.model, layers[lid], lid)
    n_layers = len(layers)
    print(f"Model has {n_layers} layers")

    # IMPORTANT: cache one fixed dataset to make subsets comparable
    print(f"\nLoading {args.nsamples} test sequences (cached, used for all subsets)...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples,
                                seqlen=seq_len, device=device)

    # Determine k range
    max_k = n_layers if args.max_k < 0 else min(args.max_k, n_layers)
    total_subsets = sum(1 for k in range(max_k + 1)
                        for _ in itertools.combinations(range(n_layers), k))
    print(f"Will enumerate {total_subsets} subsets (k=0..{max_k}, N={n_layers})")

    # Run all subsets
    results = {}            # frozenset -> top1
    per_k_min = {}          # k -> {top1, subset}
    start = time.time()
    pbar = tqdm(total=total_subsets, desc='subsets')
    for k in range(max_k + 1):
        best_for_k = None
        for subset in itertools.combinations(range(n_layers), k):
            t = measure_top1(layers, model, testseq_list, device, list(subset))
            results[subset] = t
            if best_for_k is None or t < best_for_k['top1']:
                best_for_k = {'top1': t, 'subset': list(subset)}
            pbar.update(1)
        per_k_min[k] = best_for_k
        elapsed = time.time() - start
        print(f"\n  k={k:2d}: min top1 = {best_for_k['top1']:.1f} via {best_for_k['subset']}  "
              f"(elapsed {elapsed/60:.1f} min)")
    pbar.close()

    # Build trajectory: minimum top1 reachable as k grows
    trajectory = []
    baseline_top1 = per_k_min[0]['top1']
    for k in sorted(per_k_min.keys()):
        info = per_k_min[k]
        drop_pct = (1 - info['top1'] / baseline_top1) * 100 if baseline_top1 > 0 else 0
        trajectory.append({
            'k': k,
            'min_top1': info['top1'],
            'optimal_subset': info['subset'],
            'drop_pct': drop_pct,
        })

    # Save (frozenset is not JSON serializable; use sorted-tuple → list)
    full_results = {tuple(sorted(s)): t for s, t in results.items()}
    full_results_str = {','.join(str(x) for x in s): t for s, t in full_results.items()}

    out = {
        'model': args.model,
        'n_layers': n_layers,
        'max_k_enumerated': max_k,
        'nsamples': args.nsamples,
        'timestamp': datetime.now().isoformat(),
        'baseline_top1': baseline_top1,
        'floor_top1': per_k_min[n_layers]['top1'] if n_layers in per_k_min else None,
        'trajectory': trajectory,
        'all_subsets_top1': full_results_str,
    }
    out_path = os.path.join(args.savedir, f'{args.model}_rq6_exhaustive.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*80}")
    print(f"  Baseline top1: {baseline_top1:.1f}")
    print(f"  Trajectory (min top1 as k grows):")
    for t in trajectory:
        print(f"    k={t['k']:2d}: top1={t['min_top1']:>8.1f}  drop={t['drop_pct']:5.1f}%  "
              f"subset={t['optimal_subset']}")

    # Identify "kill point": smallest k with drop > 90%
    kill_k = next((t['k'] for t in trajectory if t['drop_pct'] > 90), None)
    if kill_k is not None:
        print(f"\n  >>> Smallest k to drop MA by >90%: k={kill_k}")
        print(f"  >>> Optimal subset: {[x for x in trajectory[kill_k]['optimal_subset']]}")
    else:
        print(f"\n  >>> No k achieves 90% drop within k=0..{max_k}")
    print(f"\n  Saved to: {out_path}")


if __name__ == '__main__':
    main()
