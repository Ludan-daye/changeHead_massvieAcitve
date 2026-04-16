#!/usr/bin/env python3
"""
RQ6 Exhaustive (parallel): all 2^N MLP-disable subsets with multi-process GPU parallelism.

Each worker process:
  - loads its own copy of the model on GPU
  - pulls a chunk of subsets to evaluate
  - writes results to its own shard file

Driver then merges shards and computes the trajectory.
"""

import os, sys, argparse, json, itertools, time
import torch
import numpy as np
from datetime import datetime
import multiprocessing as mp_py

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


def worker(worker_id, subsets_chunk, args_dict, shard_path):
    """Worker process: handle its subsets and write to shard file."""
    # Rebuild argparse.Namespace
    class A: pass
    a = A()
    for k, v in args_dict.items():
        setattr(a, k, v)

    # Each worker on same GPU 0 (A100 has plenty of room for small models)
    torch.cuda.set_device(0)

    model, tokenizer, device, layers, _, seq_len = lib.load_llm(a)
    model.eval()
    for lid in range(len(layers)):
        lib.enable_custom_block(a.model, layers[lid], lid)

    # Fixed test sequences
    torch.manual_seed(0)
    np.random.seed(0)
    testseq_list = lib.get_data(tokenizer, nsamples=a.nsamples,
                                seqlen=seq_len, device=device)

    shard_results = {}
    t0 = time.time()
    for i, subset in enumerate(subsets_chunk):
        t = measure_top1(layers, model, testseq_list, device, list(subset))
        shard_results[','.join(str(x) for x in sorted(subset))] = t
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(subsets_chunk) - i - 1) / rate
            print(f"  [worker {worker_id}] {i+1}/{len(subsets_chunk)}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}min")

    with open(shard_path, 'w') as f:
        json.dump(shard_results, f)
    print(f"  [worker {worker_id}] done, saved to {shard_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--seqlen', type=int, default=1024)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--access_token', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--max_k', type=int, default=-1)
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    # Quick model load to get n_layers
    print(f"\n{'='*80}")
    print(f"RQ6 EXHAUSTIVE-PARALLEL - {args.model}")
    print(f"{'='*80}")
    print(f"Probing model to get n_layers...")
    model, tokenizer, device, layers, _, seq_len = lib.load_llm(args)
    n_layers = len(layers)
    print(f"n_layers = {n_layers}")
    del model
    torch.cuda.empty_cache()

    max_k = n_layers if args.max_k < 0 else min(args.max_k, n_layers)

    # Enumerate all subsets with k=0..max_k
    all_subsets = []
    for k in range(max_k + 1):
        for subset in itertools.combinations(range(n_layers), k):
            all_subsets.append(subset)
    total = len(all_subsets)
    print(f"Total subsets to evaluate: {total}")

    # Split into n_workers chunks
    chunks = [all_subsets[i::args.n_workers] for i in range(args.n_workers)]
    print(f"Split into {args.n_workers} workers, "
          f"chunk sizes: {[len(c) for c in chunks]}")

    # Spawn workers
    args_dict = vars(args)
    shards = []
    procs = []
    mp_py.set_start_method('spawn', force=True)
    for wid in range(args.n_workers):
        shard_path = os.path.join(args.savedir, f'shard_{wid}.json')
        shards.append(shard_path)
        p = mp_py.Process(target=worker, args=(wid, chunks[wid], args_dict, shard_path))
        p.start()
        procs.append(p)

    start = time.time()
    for p in procs:
        p.join()
    print(f"\nAll workers finished in {(time.time() - start)/60:.1f} min")

    # Merge shards
    merged = {}
    for sp in shards:
        with open(sp) as f:
            merged.update(json.load(f))
    print(f"Merged {len(merged)} results")

    # Build trajectory per k
    per_k_min = {}
    for subset_str, t in merged.items():
        subset = [] if not subset_str else [int(x) for x in subset_str.split(',')]
        k = len(subset)
        if k not in per_k_min or t < per_k_min[k]['top1']:
            per_k_min[k] = {'top1': t, 'subset': subset}

    baseline_top1 = per_k_min[0]['top1']
    trajectory = []
    for k in sorted(per_k_min.keys()):
        info = per_k_min[k]
        drop_pct = (1 - info['top1'] / baseline_top1) * 100 if baseline_top1 > 0 else 0
        trajectory.append({
            'k': k,
            'min_top1': info['top1'],
            'optimal_subset': info['subset'],
            'drop_pct': drop_pct,
        })

    out = {
        'model': args.model,
        'n_layers': n_layers,
        'max_k_enumerated': max_k,
        'nsamples': args.nsamples,
        'n_workers': args.n_workers,
        'timestamp': datetime.now().isoformat(),
        'baseline_top1': baseline_top1,
        'floor_top1': per_k_min.get(n_layers, {}).get('top1'),
        'trajectory': trajectory,
        'all_subsets_top1': merged,
    }
    out_path = os.path.join(args.savedir, f'{args.model}_rq6_exhaustive.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # Cleanup shards
    for sp in shards:
        os.remove(sp)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*80}")
    print(f"  Baseline top1: {baseline_top1:.1f}")
    for t in trajectory:
        print(f"    k={t['k']:2d}: top1={t['min_top1']:>8.1f}  "
              f"drop={t['drop_pct']:5.1f}%  subset={t['optimal_subset']}")

    kill_k = next((t['k'] for t in trajectory if t['drop_pct'] > 90), None)
    if kill_k is not None:
        sub = trajectory[kill_k]['optimal_subset']
        print(f"\n  >>> Smallest k for >90% drop: k={kill_k}, subset={sub}")
    print(f"\n  Saved to: {out_path}")


if __name__ == '__main__':
    main()
