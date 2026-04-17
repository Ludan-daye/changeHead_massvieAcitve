#!/usr/bin/env python3
"""
RQ7: MA evolution during training - using Pythia checkpoints.

Measure MA magnitude at different training steps to verify:
  - MA grows during training (not present at init)
  - MA stabilizes at some point

Uses Pythia-1.4B with public checkpoints at various training steps.
"""

import os, sys, argparse, json, time
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def get_blocks(model):
    """Get transformer blocks for Pythia/GPT-NeoX."""
    if hasattr(model, 'gpt_neox'):
        return list(model.gpt_neox.layers)
    if hasattr(model, 'transformer'):
        return list(model.transformer.h)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    raise ValueError("Unknown model architecture")


def measure_ma(model, tokenizer, device, nsamples=5, seqlen=512):
    """Run forward pass, capture max activation per layer."""
    blocks = get_blocks(model)
    n_layers = len(blocks)

    # Capture peak activation per block
    peaks = [0.0] * n_layers

    def mk_hook(idx):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            v = o.detach().abs().max().item()
            if v > peaks[idx]:
                peaks[idx] = v
        return hook

    hooks = [blk.register_forward_hook(mk_hook(i)) for i, blk in enumerate(blocks)]

    # Prepare data - use simple repeated text for consistency across checkpoints
    test_text = "The cat sat on the mat. A dog ran in the park. She went to the store and bought some food for her family. The weather was nice today so they decided to go outside."
    tokens = tokenizer(test_text, return_tensors='pt', truncation=True,
                       max_length=seqlen)['input_ids'].to(device)

    # Also try with longer diverse text
    long_text = (test_text + " ") * 20
    long_tokens = tokenizer(long_text, return_tensors='pt', truncation=True,
                            max_length=seqlen)['input_ids'].to(device)

    with torch.inference_mode():
        # Run on short text
        model(tokens)
        peaks_short = list(peaks)

        # Reset and run on long text
        for i in range(n_layers):
            peaks[i] = 0.0
        model(long_tokens)
        peaks_long = list(peaks)

    for h in hooks:
        h.remove()

    return {
        'top1_short': float(max(peaks_short)),
        'top1_long': float(max(peaks_long)),
        'top1_max': float(max(max(peaks_short), max(peaks_long))),
        'per_layer_short': [float(p) for p in peaks_short],
        'per_layer_long': [float(p) for p in peaks_long],
        'peak_layer_short': int(np.argmax(peaks_short)),
        'peak_layer_long': int(np.argmax(peaks_long)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-1.4b',
                        help='HuggingFace model name')
    parser.add_argument('--steps', type=str,
                        default='1,1000,8000,32000,64000,143000',
                        help='Comma-separated training steps to test')
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    steps = [int(s.strip()) for s in args.steps.split(',')]

    print(f"\n{'='*80}")
    print(f"RQ7: MA EVOLUTION DURING TRAINING")
    print(f"Model: {args.model_name}")
    print(f"Steps: {steps}")
    print(f"{'='*80}\n", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}
    for step in steps:
        revision = f"step{step}"
        print(f"\n--- Step {step} (revision={revision}) ---", flush=True)

        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, revision=revision, cache_dir=args.cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, revision=revision, cache_dir=args.cache_dir,
                torch_dtype=torch.float16, device_map='auto')
            model.eval()
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            results[step] = {'error': str(e)}
            continue

        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s", flush=True)

        device = next(model.parameters()).device
        ma = measure_ma(model, tokenizer, device, seqlen=args.seqlen)
        results[step] = ma

        print(f"  Top1 MA (short): {ma['top1_short']:.1f}", flush=True)
        print(f"  Top1 MA (long):  {ma['top1_long']:.1f}", flush=True)
        print(f"  Top1 MA (max):   {ma['top1_max']:.1f}", flush=True)
        print(f"  Peak layer:      L{ma['peak_layer_long']}", flush=True)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("MA EVOLUTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Step':>10} | {'Top1 MA':>10} | {'Peak Layer':>10}")
    print('-' * 40)
    for step in steps:
        r = results.get(step, {})
        if 'error' in r:
            print(f"{step:>10} | {'ERROR':>10} |")
        else:
            print(f"{step:>10} | {r['top1_max']:>10.1f} | L{r['peak_layer_long']}")

    # Save
    out = {
        'model': args.model_name,
        'steps': steps,
        'timestamp': datetime.now().isoformat(),
        'results': {str(k): v for k, v in results.items()},
    }
    out_path = os.path.join(args.savedir, 'pythia_ma_evolution.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
