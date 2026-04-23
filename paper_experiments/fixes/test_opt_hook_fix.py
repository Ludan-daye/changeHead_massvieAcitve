#!/usr/bin/env python3
"""
Smoke test for OPT MLPDisableHook fix (2026-04-23).

Verifies that:
  1. `get_mlp_module_for_hook` returns a non-None hookable module for OPT
     (previously `layer.mlp` raised AttributeError).
  2. `get_mlp_down_proj` returns the correct W_down for OPT
     (`layer.fc2.weight`, shape [hidden_size, ffn_dim]).
  3. Registering `MLPDisableHook` on that module via a forward hook
     correctly zeros the MLP residual contribution during a forward pass
     (baseline output - mlp-disabled output != 0 at non-MLP layers? no
     actually: mlp-disabled output must DIFFER from baseline, proving the
     hook fired and mutated output).
  4. For comparison, a LLaMA-style dummy layer with `.mlp` still works the
     same way.

Runs on CPU with a tiny randomly-initialised OPT config — no pretrained
weights needed. Completes in <5 seconds.
"""
import os
import sys
import torch

# Add parent `paper_experiments/` to path so `lib` imports work
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, '..')))

import lib
from lib.model_utils import get_mlp_module_for_hook, get_mlp_down_proj


# ---------------------------------------------------------------------------
# Re-define MLPDisableHook locally so we don't depend on sys.path mangling
# into RQ2_mlp_source (keeps this test self-contained).
# ---------------------------------------------------------------------------
class MLPDisableHook:
    def __init__(self, layer_id, mode='disable_all'):
        self.layer_id = layer_id
        self.mode = mode
        self.call_count = 0  # smoke-test counter

    def __call__(self, module, inputs, output):
        self.call_count += 1
        if self.mode != 'disable_all':
            return output
        if isinstance(output, tuple):
            if len(output) == 0:
                return output
            zeroed = torch.zeros_like(output[0]) if torch.is_tensor(output[0]) else output[0]
            return (zeroed,) + tuple(output[1:])
        return torch.zeros_like(output)


def test_opt():
    print("\n" + "="*70)
    print("TEST 1: OPTDecoderLayer (hidden=32, ffn=64, 2 layers)")
    print("="*70)

    from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig
    cfg = OPTConfig(
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
    )
    layer = OPTDecoderLayer(cfg)
    layer.eval()

    # (1) get_mlp_module_for_hook should NOT raise, should return fc2.
    mod = get_mlp_module_for_hook('opt_6.7b', layer)
    print(f"  [1] get_mlp_module_for_hook -> {type(mod).__name__}")
    assert mod is not None, "FAIL: got None"
    assert mod is layer.fc2, f"FAIL: expected layer.fc2, got {mod}"

    # (2) get_mlp_down_proj shape
    W_down = get_mlp_down_proj('opt_6.7b', layer)
    print(f"  [2] W_down shape: {tuple(W_down.shape)} (expect (32, 64))")
    assert W_down.shape == (32, 64), f"FAIL: unexpected shape {W_down.shape}"
    assert torch.equal(W_down, layer.fc2.weight.data), "FAIL: W_down != fc2.weight"

    # (3) Forward pass with hook ACTIVE — hook must fire, output must differ
    bsz, seq = 1, 8
    x = torch.randn(bsz, seq, cfg.hidden_size)
    attn_mask = torch.zeros(bsz, 1, seq, seq)

    # Baseline forward (no hook)
    with torch.no_grad():
        out_baseline = layer(x, attention_mask=attn_mask)[0].clone()

    hook = MLPDisableHook(layer_id=0)
    handle = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out_disabled = layer(x, attention_mask=attn_mask)[0].clone()
    finally:
        handle.remove()

    print(f"  [3] hook.call_count = {hook.call_count} (expect >= 1)")
    assert hook.call_count >= 1, f"FAIL: hook was never called"

    diff = (out_baseline - out_disabled).abs().max().item()
    print(f"  [4] |baseline - disabled|_inf = {diff:.4f} (expect > 0)")
    assert diff > 1e-6, f"FAIL: outputs identical, hook did not affect forward pass"

    print("  PASS: OPT hook fires and mutates output as expected.")


def test_llama():
    """Regression: llama-style layer with `.mlp` must still work."""
    print("\n" + "="*70)
    print("TEST 2: LlamaDecoderLayer (hidden=32, inter=64, regression)")
    print("="*70)

    try:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaConfig,
        )
    except ImportError:
        print("  SKIP: transformers version lacks LlamaDecoderLayer")
        return

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        vocab_size=100,
    )
    # LlamaDecoderLayer requires layer_idx in some versions
    try:
        layer = LlamaDecoderLayer(cfg, layer_idx=0)
    except TypeError:
        layer = LlamaDecoderLayer(cfg)

    mod = get_mlp_module_for_hook('llama2_7b', layer)
    print(f"  [1] get_mlp_module_for_hook -> {type(mod).__name__}")
    assert mod is layer.mlp, "FAIL: expected layer.mlp"

    hook = MLPDisableHook(0)
    handle = mod.register_forward_hook(hook)
    try:
        x = torch.randn(1, 8, cfg.hidden_size)
        with torch.no_grad():
            _ = layer.mlp(x)
    finally:
        handle.remove()
    print(f"  [2] hook.call_count = {hook.call_count} (expect 1)")
    assert hook.call_count == 1
    print("  PASS: LLaMA-style .mlp hook path still works.")


def test_moe_tuple_safety():
    """Sanity: MLPDisableHook handles tuple output (MoE returns tuple)."""
    print("\n" + "="*70)
    print("TEST 3: MLPDisableHook tuple-output safety")
    print("="*70)

    hook = MLPDisableHook(0)
    # Simulate a SparseMoeBlock output: (hidden_states, router_logits)
    h = torch.randn(1, 4, 16)
    r = torch.randn(1, 4, 8)
    result = hook(None, None, (h, r))
    assert isinstance(result, tuple) and len(result) == 2
    assert torch.all(result[0] == 0), "FAIL: hidden not zeroed"
    assert torch.equal(result[1], r), "FAIL: router logits mutated"
    print("  PASS: tuple output correctly zeros [0] and preserves [1].")


if __name__ == '__main__':
    test_opt()
    test_llama()
    test_moe_tuple_safety()
    print("\n" + "="*70)
    print("ALL SMOKE TESTS PASSED")
    print("="*70)
