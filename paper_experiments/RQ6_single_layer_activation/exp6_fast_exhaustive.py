#!/usr/bin/env python3
"""
RQ6 FAST Exhaustive: all 2^N MLP-disable subsets, optimized for speed.

Optimizations vs v1:
  - No monkey_patch (avoids .cpu().double() per layer per forward)
  - Capture residual-stream peak via single forward hook on last layer
  - Short seqlen (128) to cut GPU time 8x
  - Pre-cache inputs as a single batched tensor
  - Use torch.inference_mode() instead of no_grad()

Target: < 0.1 s per subset on GPT-2 small, ~7 min total for 4096 subsets.
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


def get_mlp_modules(model, model_name):
    """Collect the N per-layer MLP modules. GPT-2 specific for now."""
    if 'gpt2' in model_name.lower():
        return [blk.mlp for blk in model.transformer.h]
    if 'llama' in model_name.lower() or 'mistral' in model_name.lower() or 'qwen' in model_name.lower():
        return [blk.mlp for blk in model.model.layers]
    raise ValueError(f"Unknown model family: {model_name}")


def get_layer_blocks(model, model_name):
    if 'gpt2' in model_name.lower():
        return list(model.transformer.h)
    return list(model.model.layers)


def measure_top1_fast(blocks, mlp_modules, batch_input, device, disabled_set):
    """
    Run forward with MLP hooks zeroing `disabled_set`, capture peak residual activation.
    Uses a single residual-capture via forward hook on each block (records max abs).
    """
    peaks = [0.0] * len(blocks)

    def mk_block_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Only compute the max once, on-device
            v = h.detach().abs().max().item()
            if v > peaks[idx]:
                peaks[idx] = v
        return hook

    hooks = []
    # Zero MLPs in disabled_set
    for lid in disabled_set:
        hooks.append(mlp_modules[lid].register_forward_hook(MLPZeroHook()))
    # Capture peak per block
    for i, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(mk_block_hook(i)))

    with torch.inference_mode():
        model_fn = blocks[0].__self__ if hasattr(blocks[0], '__self__') else None
        # We don't have model here; rely on caller-provided forward

    # Cleanup caller does, not here (we return without model call)
    # Instead, caller does the forward — but we want a single function.
    # Simpler: receive model+input, do the forward here.
    raise RuntimeError("unreachable")


def worker(worker_id, subsets_chunk, args_dict, shard_path):
    """Each worker: load model once, run its chunk of subsets, write shard."""
    class A: pass
    a = A()
    for k, v in args_dict.items():
        setattr(a, k, v)

    torch.cuda.set_device(0)
    print(f"[worker {worker_id}] loading model...", flush=True)
    model, tokenizer, device, layers, _, seq_len = lib.load_llm(a)
    model.eval()

    # Use short seqlen for speed (override default)
    eff_seqlen = min(a.seqlen, seq_len)

    blocks = get_layer_blocks(model, a.model)
    mlp_modules = get_mlp_modules(model, a.model)
    n_layers = len(blocks)

    # Prepare a single batched tensor of inputs (same for all subsets → fair comparison)
    torch.manual_seed(0)
    np.random.seed(0)
    testseq_list = lib.get_data(tokenizer, nsamples=a.nsamples,
                                seqlen=eff_seqlen, device=device)
    # Stack into one batch [B, T]
    batch = torch.cat([s.view(1, -1)[:, :eff_seqlen] for s in testseq_list], dim=0).to(device)
    print(f"[worker {worker_id}] batch shape: {batch.shape}", flush=True)

    # Pre-register per-block max-capture hooks (always active)
    peaks = [0.0] * n_layers

    def mk_block_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = h.detach().abs().max().item()
            if v > peaks[idx]:
                peaks[idx] = v
        return hook

    block_hooks = []
    for i, blk in enumerate(blocks):
        block_hooks.append(blk.register_forward_hook(mk_block_hook(i)))

    shard_results = {}
    t0 = time.time()
    for i, subset in enumerate(subsets_chunk):
        # Reset peaks
        for j in range(n_layers):
            peaks[j] = 0.0
        # Install MLP-zero hooks for this subset
        zero_hooks = []
        for lid in subset:
            zero_hooks.append(mlp_modules[lid].register_forward_hook(MLPZeroHook()))

        with torch.inference_mode():
            model(batch)

        for h in zero_hooks:
            h.remove()

        top1 = max(peaks)
        shard_results[','.join(str(x) for x in sorted(subset))] = float(top1)

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(subsets_chunk) - i - 1) / rate
            print(f"[worker {worker_id}] {i+1}/{len(subsets_chunk)}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    for h in block_hooks:
        h.remove()

    with open(shard_path, 'w') as f:
        json.dump(shard_results, f)
    print(f"[worker {worker_id}] done → {shard_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--seqlen', type=int, default=128)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--access_token', type=str, default='')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--max_k', type=int, default=-1)
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"RQ6 FAST-EXHAUSTIVE - {args.model}")
    print(f"  nsamples={args.nsamples}, seqlen={args.seqlen}, workers={args.n_workers}")
    print(f"{'='*80}", flush=True)

    # Probe n_layers via a quick load
    print("Probing n_layers...", flush=True)
    model, _, _, layers, _, _ = lib.load_llm(args)
    n_layers = len(layers)
    del model
    torch.cuda.empty_cache()
    print(f"n_layers = {n_layers}", flush=True)

    max_k = n_layers if args.max_k < 0 else min(args.max_k, n_layers)
    all_subsets = []
    for k in range(max_k + 1):
        for s in itertools.combinations(range(n_layers), k):
            all_subsets.append(s)
    total = len(all_subsets)
    print(f"Total subsets: {total}", flush=True)

    chunks = [all_subsets[i::args.n_workers] for i in range(args.n_workers)]
    args_dict = vars(args)

    mp_py.set_start_method('spawn', force=True)
    shards = []
    procs = []
    for wid in range(args.n_workers):
        shard_path = os.path.join(args.savedir, f'shard_{wid}.json')
        shards.append(shard_path)
        p = mp_py.Process(target=worker, args=(wid, chunks[wid], args_dict, shard_path))
        p.start()
        procs.append(p)

    start = time.time()
    for p in procs:
        p.join()
    print(f"\nAll workers finished in {(time.time() - start)/60:.1f} min", flush=True)

    # Merge
    merged = {}
    for sp in shards:
        if os.path.exists(sp):
            with open(sp) as f:
                merged.update(json.load(f))
    print(f"Merged {len(merged)} results", flush=True)

    per_k_min = {}
    for ss, t in merged.items():
        subset = [] if not ss else [int(x) for x in ss.split(',')]
        k = len(subset)
        if k not in per_k_min or t < per_k_min[k]['top1']:
            per_k_min[k] = {'top1': t, 'subset': subset}

    baseline_top1 = per_k_min[0]['top1']
    trajectory = []
    for k in sorted(per_k_min):
        info = per_k_min[k]
        drop = (1 - info['top1'] / baseline_top1) * 100 if baseline_top1 > 0 else 0
        trajectory.append({
            'k': k, 'min_top1': info['top1'],
            'optimal_subset': info['subset'], 'drop_pct': drop,
        })

    out = {
        'model': args.model, 'n_layers': n_layers,
        'nsamples': args.nsamples, 'seqlen': args.seqlen,
        'timestamp': datetime.now().isoformat(),
        'baseline_top1': baseline_top1,
        'floor_top1': per_k_min.get(n_layers, {}).get('top1'),
        'trajectory': trajectory,
        'all_subsets_top1': merged,
    }
    out_path = os.path.join(args.savedir, f'{args.model}_rq6_exhaustive.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    for sp in shards:
        if os.path.exists(sp):
            os.remove(sp)

    print(f"\n{'='*80}\nSUMMARY: {args.model}\n{'='*80}")
    print(f"  Baseline top1: {baseline_top1:.1f}")
    for t in trajectory:
        print(f"    k={t['k']:2d}: top1={t['min_top1']:>8.1f}  "
              f"drop={t['drop_pct']:5.1f}%  subset={t['optimal_subset']}")
    kill_k = next((t['k'] for t in trajectory if t['drop_pct'] > 90), None)
    if kill_k is not None:
        print(f"\n  >>> Smallest k for >90% drop: k={kill_k}")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
