"""
Model-agnostic utilities for experiment scripts.
Handles monkey_patch selection and MA dimension detection.
"""

import torch
import numpy as np


# -------------------------------------------------------------------
# Bug B2 fix (2026-04-22): MoE detection helper.
# Qwen3MoE / Qwen3.5MoE store all experts as STACKED 3D Parameter tensors
# on `layer.mlp.experts.gate_up_proj` / `down_proj` (shape [E, ...]);
# the SparseMoeBlock itself has NO `.up_proj` / `.down_proj` attributes,
# so scripts accessing those on MoE layers crash with AttributeError.
# Older-style MoE (e.g. Mixtral) stores experts as an nn.ModuleList where
# each expert is its own nn.Module with `.up_proj / .down_proj`.
# We detect both shapes below.
# -------------------------------------------------------------------
def _is_moe_layer(layer):
    """Return True if this transformer layer's MLP block is a MoE block.

    Covers:
      - Qwen3MoeSparseMoeBlock / Qwen3_5MoeSparseMoeBlock
        (experts stored as 3D `nn.Parameter` on `mlp.experts.{gate_up_proj,down_proj}`)
      - Mixtral-style (experts as `nn.ModuleList`, each with `.up_proj/.down_proj`)
    """
    mlp = getattr(layer, 'mlp', None)
    if mlp is None:
        return False
    # Both shapes expose `.experts`:
    experts = getattr(mlp, 'experts', None)
    if experts is None:
        return False
    # Stacked-tensor shape (Qwen3MoE, Qwen3.5MoE in current transformers)
    if hasattr(experts, 'gate_up_proj') or hasattr(experts, 'down_proj'):
        return True
    # ModuleList shape (Mixtral, older Qwen2MoE): has len() and indexed access
    try:
        if len(experts) > 0:
            e0 = experts[0]
            if hasattr(e0, 'up_proj') or hasattr(e0, 'down_proj') \
                    or hasattr(e0, 'w1') or hasattr(e0, 'w2'):
                return True
    except (TypeError, AttributeError):
        pass
    return False


def _moe_effective_down_proj(layer):
    """Return an 'effective' W_down = uniform mean over experts.

    Shape returned: [hidden_size, intermediate_size]
    Used by RQ3/RQ4/RQ5 as a best-effort dense approximation of the MoE layer's
    down projection for SVD analysis. NOT accurate for per-token routing; MoE
    should ideally use per-expert analysis (see exp5_v_ablation_moe.py).
    """
    mlp = layer.mlp
    experts = mlp.experts
    # Stacked-tensor shape
    if hasattr(experts, 'down_proj') and isinstance(experts.down_proj, torch.nn.Parameter):
        # [E, hidden, intermediate]  →  mean over E  →  [hidden, intermediate]
        return experts.down_proj.data.mean(dim=0)
    # ModuleList shape — mean the .weight tensors
    stacks = []
    for e in experts:
        w = None
        for name in ('down_proj', 'w2'):
            sub = getattr(e, name, None)
            if sub is not None and hasattr(sub, 'weight'):
                w = sub.weight.data
                break
        if w is not None:
            stacks.append(w)
    if not stacks:
        raise RuntimeError("MoE layer has no extractable expert down-projections")
    return torch.stack(stacks, dim=0).mean(dim=0)


def _moe_effective_up_proj(layer):
    """Return an 'effective' W_up = uniform mean over experts.

    Handles both fused gate_up (Qwen3MoE: chunk last dim into gate/up) and
    separate up_proj/w1 (Mixtral-style).
    Shape: [intermediate_size, hidden_size]
    """
    mlp = layer.mlp
    experts = mlp.experts
    if hasattr(experts, 'gate_up_proj') and isinstance(experts.gate_up_proj, torch.nn.Parameter):
        # [E, 2*intermediate, hidden] — take the 'up' half (second chunk)
        gu = experts.gate_up_proj.data
        _gate, up = gu.chunk(2, dim=1)  # up: [E, intermediate, hidden]
        return up.mean(dim=0)
    if hasattr(experts, 'up_proj') and isinstance(experts.up_proj, torch.nn.Parameter):
        return experts.up_proj.data.mean(dim=0)
    stacks = []
    for e in experts:
        w = None
        for name in ('up_proj', 'w3', 'w1'):
            sub = getattr(e, name, None)
            if sub is not None and hasattr(sub, 'weight'):
                w = sub.weight.data
                break
        if w is not None:
            stacks.append(w)
    if not stacks:
        raise RuntimeError("MoE layer has no extractable expert up-projections")
    return torch.stack(stacks, dim=0).mean(dim=0)


def enable_custom_block(model_name, layer, layer_id):
    """
    Apply the correct monkey_patch based on model name.

    Args:
        model_name: str, the --model argument value
        layer: the layer module to patch
        layer_id: int, layer index
    """
    import monkey_patch as mp

    # Bug B11 fix (2026-04-21): glm4 uses ChatGLM's GLMBlock (not llama-compatible).
    # Attention attribute is `self_attention` (underscore), forward signature is
    # different. Instead of rewriting the forward, use an architecture-agnostic
    # forward HOOK that just captures the layer's final output → self.feat.
    # This preserves the original GLMBlock logic untouched.
    # Bug B11 (glm4) + Bug B13 (opt) + Bug B14 (gptj, bloom, falcon):
    # Use arch-agnostic forward hook. Avoids fragile monkey-patching of forward
    # signatures that break across transformers versions (4.50+).
    if ("glm4" in model_name
            or "opt" in model_name
            or "gptj" in model_name or "gpt-j" in model_name
            or "bloom" in model_name
            or "falcon" in model_name
            or "llama2" in model_name):
        import torch as _torch
        def _feat_capture_hook(module, _input, output):
            hidden = output if not isinstance(output, tuple) else output[0]
            try:
                module.feat = hidden.clone().detach().cpu().double()
            except Exception:
                module.feat = hidden
        layer.layer_id = layer_id
        layer._feat_hook_handle = layer.register_forward_hook(_feat_capture_hook)
        return

    # Bug B9 fix (2026-04-21): yi uses llama-compatible decoder layer.
    if ("llama" in model_name or "qwen" in model_name
            or "yi" in model_name):
        mp.enable_llama_custom_decoderlayer(layer, layer_id)
    elif "mistral" in model_name:
        mp.enable_mistral_custom_decoderlayer(layer, layer_id)
    elif "gpt2" in model_name:
        mp.enable_gpt2_custom_block(layer, layer_id)
    elif "gptj" in model_name or "gpt-j" in model_name or "bloom" in model_name or "falcon" in model_name:
        # GPT-J, BLOOM, Falcon use GPT-2 style transformer.h blocks
        mp.enable_gpt2_custom_block(layer, layer_id)
    elif "opt" in model_name:
        # OPT uses decoder layers similar to LLaMA
        mp.enable_llama_custom_decoderlayer(layer, layer_id)
    elif "phi" in model_name:
        mp.enable_phi2_custom_decoderlayer(layer, layer_id)
    else:
        # Try GPT-2 style as default fallback
        try:
            mp.enable_gpt2_custom_block(layer, layer_id)
        except Exception:
            raise ValueError(
                f"No monkey_patch available for model '{model_name}'. "
                f"Add support in lib/model_utils.py:enable_custom_block()"
            )


def enable_custom_attention(model_name, layer, layer_id):
    """
    Apply the correct attention monkey_patch based on model name.
    """
    import monkey_patch as mp

    # Bug B9 fix: glm4/yi use llama-compatible attention
    if ("llama" in model_name or "qwen" in model_name
            or "glm4" in model_name or "yi" in model_name):
        mp.enable_llama_custom_attention(layer, layer_id)
    elif "mistral" in model_name:
        mp.enable_mistral_custom_attention(layer, layer_id)
    elif "gpt2" in model_name:
        mp.enable_gpt2_custom_attention(layer, layer_id)
    elif "gptj" in model_name or "gpt-j" in model_name or "bloom" in model_name or "falcon" in model_name:
        mp.enable_gpt2_custom_attention(layer, layer_id)
    elif "phi" in model_name:
        mp.enable_phi2_custom_attention(layer, layer_id)
    else:
        try:
            mp.enable_gpt2_custom_attention(layer, layer_id)
        except Exception:
            raise ValueError(
                f"No attention monkey_patch for model '{model_name}'. "
                f"Add support in lib/model_utils.py:enable_custom_attention()"
            )


def detect_ma_dimensions(feat, top_k=2):
    """
    Dynamically detect the top-K MA dimensions from activation features,
    instead of hardcoding dim 138/447 (GPT-2 specific).

    Args:
        feat: torch.Tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        top_k: number of top dimensions to return

    Returns:
        list of (dim_index, max_value) tuples, sorted by max_value descending
    """
    if feat.dim() == 3:
        feat_abs = feat.abs().max(dim=0).values.max(dim=0).values  # [hidden_dim]
    elif feat.dim() == 2:
        feat_abs = feat.abs().max(dim=0).values  # [hidden_dim]
    else:
        feat_abs = feat.abs()

    top_values, top_indices = torch.topk(feat_abs, top_k)
    return [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]


def get_mlp_down_proj(model_name, layer):
    """
    Extract the MLP down-projection weight matrix (W_down) from a layer.
    Different models have different attribute names.

    Args:
        model_name: str
        layer: the layer module

    Returns:
        torch.Tensor: the W_down weight matrix
    """
    # Bug B2 fix (2026-04-22): MoE models (Qwen3MoE, Qwen3.5MoE, Mixtral) have
    # no single `layer.mlp.down_proj` — experts are stored either as a 3D
    # Parameter stack (`mlp.experts.down_proj`) or an nn.ModuleList of experts.
    # Return the uniform-mean "effective" W_down so RQ3/RQ4 SVD analyses do not
    # crash. Per-expert analysis should use exp5_v_ablation_moe.py directly.
    if _is_moe_layer(layer):
        return _moe_effective_down_proj(layer)
    # Bug B18 fix (2026-04-22): glm4 is polymorphic —
    #   - HF-native `Glm4MLP` (transformers 4.50+): `mlp.down_proj` (SwiGLU, fused gate_up)
    #   - ChatGLM shim (glm4_9b trust_remote_code): `mlp.dense_4h_to_h`
    # glm4_32b loads as HF-native; glm4_9b loads via ChatGLM shim. Detect at runtime.
    if "glm4" in model_name:
        mlp = layer.mlp
        if hasattr(mlp, 'down_proj'):
            return mlp.down_proj.weight.data
        if hasattr(mlp, 'dense_4h_to_h'):
            return mlp.dense_4h_to_h.weight.data
        raise AttributeError(
            f"glm4 layer.mlp has neither 'down_proj' nor 'dense_4h_to_h' "
            f"(got attrs: {[a for a in dir(mlp) if not a.startswith('_')][:10]})"
        )
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
        return layer.mlp.down_proj.weight.data
    elif "yi" in model_name:
        # yi_9b uses llama-style SwiGLU
        return layer.mlp.down_proj.weight.data
    elif "gpt2" in model_name:
        return layer.mlp.c_proj.weight.data.T  # GPT-2 uses Conv1D, need transpose
    elif "gptj" in model_name or "gpt-j" in model_name:
        return layer.mlp.fc_out.weight.data
    elif "bloom" in model_name:
        return layer.mlp.dense_4h_to_h.weight.data
    elif "falcon" in model_name:
        return layer.mlp.dense_4h_to_h.weight.data
    elif "opt" in model_name:
        return layer.fc2.weight.data
    elif "phi" in model_name:
        return layer.mlp.fc2.weight.data
    elif "mpt" in model_name:
        return layer.ffn.down_proj.weight.data
    elif "pythia" in model_name:
        return layer.mlp.dense_4h_to_h.weight.data
    else:
        # Try common patterns
        for attr_path in ['mlp.down_proj.weight', 'mlp.c_proj.weight', 'mlp.fc_out.weight', 'mlp.dense_4h_to_h.weight']:
            parts = attr_path.split('.')
            obj = layer
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj.data
            except AttributeError:
                continue
        raise ValueError(f"Cannot find MLP down-projection weight for model '{model_name}'")


def set_mlp_down_proj(model_name, layer, new_weight):
    """
    Set the MLP down-projection weight matrix. Inverse of get_mlp_down_proj.
    Handles Conv1D transpose for GPT-2.

    Args:
        model_name: str
        layer: the layer module
        new_weight: torch.Tensor, shape [hidden_size, intermediate_size]
    """
    # Bug B2 fix (2026-04-22): MoE is not supported for single-V ablation.
    # The caller should use exp5_v_ablation_moe.py which handles per-expert
    # W_down. We cannot write back a single averaged matrix because each expert
    # has its own W_down_e.
    if _is_moe_layer(layer):
        raise NotImplementedError(
            f"set_mlp_down_proj: model '{model_name}' uses MoE (layer.mlp is "
            f"SparseMoeBlock). Single-W ablation is not defined for MoE. "
            f"Use exp5_v_ablation_moe.py (per-expert ablation) instead."
        )
    # Bug B18 fix (2026-04-22): glm4 dual-path (HF-native vs ChatGLM shim).
    if "glm4" in model_name:
        mlp = layer.mlp
        if hasattr(mlp, 'down_proj'):
            mlp.down_proj.weight.data = new_weight
            return
        if hasattr(mlp, 'dense_4h_to_h'):
            mlp.dense_4h_to_h.weight.data = new_weight
            return
        raise AttributeError(
            f"glm4 layer.mlp has neither 'down_proj' nor 'dense_4h_to_h'"
        )
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
        layer.mlp.down_proj.weight.data = new_weight
    elif "yi" in model_name:
        # yi_9b uses llama-style SwiGLU
        layer.mlp.down_proj.weight.data = new_weight
    elif "gpt2" in model_name:
        layer.mlp.c_proj.weight.data = new_weight.T  # Conv1D: transpose back
    elif "gptj" in model_name or "gpt-j" in model_name:
        layer.mlp.fc_out.weight.data = new_weight
    elif "bloom" in model_name:
        layer.mlp.dense_4h_to_h.weight.data = new_weight
    elif "falcon" in model_name:
        layer.mlp.dense_4h_to_h.weight.data = new_weight
    elif "opt" in model_name:
        layer.fc2.weight.data = new_weight
    elif "phi" in model_name:
        layer.mlp.fc2.weight.data = new_weight
    elif "mpt" in model_name:
        layer.ffn.down_proj.weight.data = new_weight
    elif "pythia" in model_name:
        layer.mlp.dense_4h_to_h.weight.data = new_weight
    else:
        raise ValueError(f"Cannot set MLP down-projection weight for model '{model_name}'")


def get_mlp_up_proj(model_name, layer):
    """
    Extract the MLP up-projection weight matrix (W_up) from a layer.

    Args:
        model_name: str
        layer: the layer module

    Returns:
        torch.Tensor: the W_up weight matrix
    """
    # Bug B2 fix: MoE — return uniform-mean effective W_up.
    if _is_moe_layer(layer):
        return _moe_effective_up_proj(layer)
    # Bug B18 fix: glm4 dual-path (HF-native has fused `gate_up_proj`,
    # ChatGLM shim has fused `dense_h_to_4h`). For both, the 'up' half is the
    # second chunk of the fused tensor.
    if "glm4" in model_name:
        mlp = layer.mlp
        fused = None
        if hasattr(mlp, 'gate_up_proj'):
            fused = mlp.gate_up_proj.weight.data
        elif hasattr(mlp, 'dense_h_to_4h'):
            fused = mlp.dense_h_to_4h.weight.data
        if fused is not None:
            # fused shape: [2*intermediate, hidden]; up is second chunk along dim 0
            _gate, up = fused.chunk(2, dim=0)
            return up
        # Fallback: non-fused up_proj (shouldn't happen for glm4 in practice)
        if hasattr(mlp, 'up_proj'):
            return mlp.up_proj.weight.data
        raise AttributeError(f"glm4 layer.mlp has no recognised up projection")
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
        return layer.mlp.up_proj.weight.data
    elif "yi" in model_name:
        return layer.mlp.up_proj.weight.data
    elif "gpt2" in model_name:
        return layer.mlp.c_fc.weight.data.T  # Conv1D: transpose
    elif "gptj" in model_name or "gpt-j" in model_name:
        return layer.mlp.fc_in.weight.data
    elif "bloom" in model_name:
        return layer.mlp.dense_h_to_4h.weight.data
    elif "falcon" in model_name:
        return layer.mlp.dense_h_to_4h.weight.data
    elif "opt" in model_name:
        return layer.fc1.weight.data
    elif "phi" in model_name:
        return layer.mlp.fc1.weight.data
    elif "mpt" in model_name:
        return layer.ffn.up_proj.weight.data
    elif "pythia" in model_name:
        return layer.mlp.dense_h_to_4h.weight.data
    else:
        raise ValueError(f"Cannot find MLP up-projection weight for model '{model_name}'")


def get_mlp_submodules(model_name, layer):
    """
    Get MLP submodule references for hook registration.
    Returns a dict with module references for up_proj, activation, down_proj,
    and optionally gate_proj (for SwiGLU models).

    Args:
        model_name: str
        layer: the layer module

    Returns:
        dict with keys: 'up_proj', 'activation', 'down_proj', 'gate_proj', 'is_gated'
    """
    def _try_getattr(obj, *attrs):
        for attr in attrs:
            parts = attr.split('.')
            cur = obj
            try:
                for p in parts:
                    cur = getattr(cur, p)
                return cur
            except AttributeError:
                continue
        return None

    # Bug B2 fix (2026-04-22): MoE — experts stored as stacked Parameter
    # (Qwen3MoE: `mlp.experts.gate_up_proj` / `down_proj`, each [E, ...])
    # or as nn.ModuleList (Mixtral). There is no single module to hook as
    # `down_proj` for per-layer h2 capture; the per-token activation passes
    # through a different subset of experts. Callers that need hookable
    # submodules (RQ3 h2 capture, RQ4 pre-hook, HC entropy) must detect
    # `is_moe=True` and either (a) skip the model, or (b) use the dedicated
    # MoE scripts (exp5_v_ablation_moe.py, exp5_macro_v_ablation_moe.py).
    if _is_moe_layer(layer):
        return {
            'up_proj': None,
            'down_proj': None,
            'gate_proj': None,
            'activation': _try_getattr(layer, 'mlp.act_fn', 'mlp.activation_fn'),
            'is_gated': True,
            'is_moe': True,
            'experts': layer.mlp.experts,
        }

    # Bug B11 fix (2026-04-21): glm4_9b is loaded via ChatGLM remote code (shim),
    # not HF's native Glm4MLP. The ChatGLM MLP uses BLOOM-style naming
    # `dense_h_to_4h` / `dense_4h_to_h` but with a FUSED gate+up projection
    # (e.g., 13696 intermediate → dense_h_to_4h: [hidden, 27392=2*13696]).
    # Handle both cases defensively:
    #   - HF native glm4 (gate_up_proj + down_proj)
    #   - ChatGLM shim (dense_h_to_4h fused + dense_4h_to_h)
    if "glm4" in model_name:
        mlp = layer.mlp
        # ChatGLM shim path (typical for glm4_9b loaded with trust_remote_code)
        if hasattr(mlp, 'dense_h_to_4h') and hasattr(mlp, 'dense_4h_to_h'):
            return {
                'up_proj': None,
                'gate_up_proj': mlp.dense_h_to_4h,  # fused [hidden, 2*intermediate]
                'activation': _try_getattr(layer, 'mlp.activation_func', 'mlp.activation_fn', 'mlp.act_fn'),
                'down_proj': mlp.dense_4h_to_h,
                'gate_proj': None,
                'is_gated': True,
                'is_fused_gate_up': True,
            }
        # HF native Glm4MLP path (gate_up_proj + down_proj)
        if hasattr(mlp, 'gate_up_proj'):
            return {
                'up_proj': None,
                'gate_up_proj': mlp.gate_up_proj,
                'activation': _try_getattr(layer, 'mlp.activation_fn', 'mlp.act_fn'),
                'down_proj': mlp.down_proj,
                'gate_proj': None,
                'is_gated': True,
                'is_fused_gate_up': True,
            }
        # Fallback to standard SwiGLU naming (should not hit for glm4 in practice)
        return {
            'up_proj': getattr(mlp, 'up_proj', None),
            'activation': _try_getattr(layer, 'mlp.act_fn', 'mlp.activation_fn'),
            'down_proj': getattr(mlp, 'down_proj', None),
            'gate_proj': getattr(mlp, 'gate_proj', None),
            'is_gated': True,
        }

    # Bug B3 fix (2026-04-21): yi uses standard SwiGLU (up_proj/gate_proj/down_proj).
    # Also covers llama/mistral/qwen which were always here.
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name \
            or "yi" in model_name:
        return {
            'up_proj': layer.mlp.up_proj,
            'activation': _try_getattr(layer, 'mlp.act_fn', 'mlp.activation_fn'),
            'down_proj': layer.mlp.down_proj,
            'gate_proj': _try_getattr(layer, 'mlp.gate_proj'),
            'is_gated': True,
        }
    elif "gpt2" in model_name:
        return {
            'up_proj': layer.mlp.c_fc,
            'activation': layer.mlp.act,
            'down_proj': layer.mlp.c_proj,
            'gate_proj': None,
            'is_gated': False,
        }
    elif "gptj" in model_name or "gpt-j" in model_name:
        return {
            'up_proj': layer.mlp.fc_in,
            'activation': _try_getattr(layer, 'mlp.act', 'mlp.activation_fn'),
            'down_proj': layer.mlp.fc_out,
            'gate_proj': None,
            'is_gated': False,
        }
    elif "bloom" in model_name:
        return {
            'up_proj': layer.mlp.dense_h_to_4h,
            'activation': _try_getattr(layer, 'mlp.gelu_impl', 'mlp.act'),
            'down_proj': layer.mlp.dense_4h_to_h,
            'gate_proj': None,
            'is_gated': False,
        }
    elif "falcon" in model_name:
        return {
            'up_proj': layer.mlp.dense_h_to_4h,
            'activation': _try_getattr(layer, 'mlp.act', 'mlp.activation_fn'),
            'down_proj': layer.mlp.dense_4h_to_h,
            'gate_proj': None,
            'is_gated': False,
        }
    elif "opt" in model_name:
        return {
            'up_proj': layer.fc1,
            'activation': _try_getattr(layer, 'activation_fn'),
            'down_proj': layer.fc2,
            'gate_proj': None,
            'is_gated': False,
        }
    elif "phi" in model_name:
        return {
            'up_proj': layer.mlp.fc1,
            'activation': _try_getattr(layer, 'mlp.activation_fn', 'mlp.act'),
            'down_proj': layer.mlp.fc2,
            'gate_proj': None,
            'is_gated': False,
        }
    elif "pythia" in model_name:
        return {
            'up_proj': layer.mlp.dense_h_to_4h,
            'activation': _try_getattr(layer, 'mlp.act', 'mlp.activation_fn'),
            'down_proj': layer.mlp.dense_4h_to_h,
            'gate_proj': None,
            'is_gated': False,
        }
    else:
        raise ValueError(f"Cannot identify MLP submodules for model '{model_name}'. Add support in model_utils.py.")
