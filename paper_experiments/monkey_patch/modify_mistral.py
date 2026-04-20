import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb, 
    repeat_kv,
)


def mistral_custom_decoderlayer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    past_key_values=None,  # transformers 4.57+ uses this name
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """transformers-4.57-compatible MistralDecoderLayer.forward with feat capture."""
    # harmonize deprecated arg names
    if past_key_values is None:
        past_key_values = past_key_value

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # self_attn kwargs — include only keys the current tx version accepts.
    # In tx 4.57 the signature is (hidden_states, attention_mask, position_ids,
    # past_key_values, use_cache, cache_position, position_embeddings, **kwargs).
    attn_kwargs = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
    }
    if cache_position is not None:
        attn_kwargs["cache_position"] = cache_position
    if position_embeddings is not None:
        attn_kwargs["position_embeddings"] = position_embeddings
    # Pass past_key_values under whichever name the current self_attn accepts.
    import inspect
    _sig = inspect.signature(self.self_attn.forward).parameters
    if "past_key_values" in _sig:
        attn_kwargs["past_key_values"] = past_key_values
    elif "past_key_value" in _sig:
        attn_kwargs["past_key_value"] = past_key_values
    if "output_attentions" in _sig:
        attn_kwargs["output_attentions"] = output_attentions

    attn_out = self.self_attn(**attn_kwargs)
    # tx 4.57 returns (hidden_states, attn_weights); older returns 3-tuple
    if isinstance(attn_out, tuple):
        if len(attn_out) >= 3:
            hidden_states, self_attn_weights, present_key_value = attn_out[:3]
        elif len(attn_out) == 2:
            hidden_states, self_attn_weights = attn_out
            present_key_value = None
        else:
            hidden_states = attn_out[0]
            self_attn_weights, present_key_value = None, None
    else:
        hidden_states = attn_out
        self_attn_weights, present_key_value = None, None

    if hasattr(residual, "device") and hasattr(hidden_states, "device"):
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    if getattr(self, "_save_feat", True):
        self.feat = hidden_states.detach().cpu().float()

    # transformers 4.57+ MistralDecoderLayer.forward returns a bare Tensor.
    # We always return a Tensor to avoid breaking layer stacking.
    return hidden_states

def enable_mistral_custom_decoderlayer(layer, layer_id):
    """
    replace the forward function of MistralDecoderLayer with a custom forward function `mistral_custom_decoderlayer_forward`
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        mistral_custom_decoderlayer_forward, layer
    )

def mistral_custom_attention_forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ###################################################
        # self.hidden_states = hidden_states.detach().cpu().clone()
        self.query_states = query_states.detach().cpu().clone()
        self.key_states = key_states.detach().cpu().clone()
        self.value_states = value_states.detach().cpu().clone()
        # ###################################################

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

        # repeat k/v heads if n_kv_heads < n_heads
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

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def enable_mistral_custom_attention(layer, layer_id):
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(mistral_custom_attention_forward, modified_module)

    return modified_module
