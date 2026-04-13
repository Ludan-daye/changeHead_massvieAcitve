"""
Model-agnostic utilities for experiment scripts.
Handles monkey_patch selection and MA dimension detection.
"""

import torch
import numpy as np


def enable_custom_block(model_name, layer, layer_id):
    """
    Apply the correct monkey_patch based on model name.

    Args:
        model_name: str, the --model argument value
        layer: the layer module to patch
        layer_id: int, layer index
    """
    import monkey_patch as mp

    if "llama" in model_name or "qwen" in model_name:
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

    if "llama" in model_name or "qwen" in model_name:
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
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
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
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
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
    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
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

    if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
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
