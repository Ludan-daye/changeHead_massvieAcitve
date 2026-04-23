import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)


def llama_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    past_key_values=None,  # transformers 5.x uses this (plural)
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    position_embeddings=None,  # transformers 5.x extra
    **kwargs,
) -> torch.Tensor:
    """Capture feat after MLP residual-add, then return hidden_states tensor.

    Bug B8 fix (2026-04-21): rewritten for transformers 5.x API compatibility.
      - DecoderLayer.forward now returns just Tensor (not tuple)
      - self_attn returns 2-tuple (hidden, _) instead of 3-tuple
      - New kwargs: past_key_values (plural), position_embeddings
      - Old kwargs (past_key_value, output_attentions) kept for backward compat
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention — defensive unpacking for both old and new APIs.
    # Bug B10 fix (2026-04-21): qwen3.5 hybrid architecture uses either
    #   self_attn (Qwen3_5Attention) OR linear_attn (Qwen3_5GatedDeltaNet)
    # depending on config.layer_types[layer_idx]. We detect which one exists
    # and dispatch appropriately. For vanilla llama/qwen/mistral, self_attn
    # always exists so behavior is unchanged.
    attn_kwargs = dict(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )
    # Add position_ids / use_cache for self_attn path (linear_attn doesn't take these)
    if hasattr(self, 'self_attn'):
        attn_kwargs["position_ids"] = position_ids
        attn_kwargs["use_cache"] = use_cache
        # transformers 5.x expects these (plural + position_embeddings)
        if past_key_values is not None:
            attn_kwargs["past_key_values"] = past_key_values
        elif past_key_value is not None:
            attn_kwargs["past_key_value"] = past_key_value
        if position_embeddings is not None:
            attn_kwargs["position_embeddings"] = position_embeddings
        if output_attentions:
            attn_kwargs["output_attentions"] = output_attentions
        attn_kwargs.update(kwargs)
        _attn_outputs = self.self_attn(**attn_kwargs)
    elif hasattr(self, 'linear_attn'):
        # Qwen3.5 hybrid: some layers use linear attention (GatedDeltaNet).
        # Different API — takes cache_params instead of past_key_values.
        linear_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        if past_key_values is not None:
            linear_kwargs["cache_params"] = past_key_values
        _attn_outputs = self.linear_attn(**linear_kwargs)
    else:
        # No attention at this layer — identity pass-through
        _attn_outputs = hidden_states

    if isinstance(_attn_outputs, tuple):
        hidden_states = _attn_outputs[0]
    else:
        hidden_states = _attn_outputs

    if residual.device.index != hidden_states.device.index:
        residual = residual.to(hidden_states.device)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    self.feat = hidden_states.clone().detach().cpu().double()

    # Bug B8 fix: transformers 5.x expects Tensor return, not tuple.
    # (old 4.36 returned (hidden,) tuple with optional extras)
    return hidden_states

def enable_llama_custom_decoderlayer(layer, layer_id):
    """
    replace the forward function of LlamaDecoderLayer with a custom forward function `llama_custom_decoderlayer_forward`
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        llama_custom_decoderlayer_forward, layer
    )


def apply_rotary_pos_emb_single(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def llama_custom_attention_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    ##################################################################
    self.query_states = query_states.detach().cpu().clone()
    self.key_states = key_states.detach().cpu().clone()
    self.value_states = value_states.detach().cpu().clone()
    # ###################################################

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # ##################################################################
    # self.query_states = query_states.detach().cpu().clone()
    # self.key_states = key_states.detach().cpu().clone()
    # self.value_states = value_states.detach().cpu().clone()
    # # ###################################################

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # ###################################################
    self.attn_logits = attn_weights
    # ###################################################

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # ###################################################
    self.attn_probs = attn_weights
    # ###################################################

    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def enable_llama_custom_attention(layer, layer_id):
    """
    replace the forward function of LlamaAttention with a custom forward function `llama_custom_attention_forward`
    """
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(llama_custom_attention_forward, modified_module)

    return modified_module