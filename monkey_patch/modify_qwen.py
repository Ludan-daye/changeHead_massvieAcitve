import types
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache


def qwen_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Custom forward for Qwen2DecoderLayer that saves hidden states.
    Qwen2 uses Pre-LN + RoPE + SwiGLU (similar to LLaMA)
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    # Save the final hidden state of the block
    self.feat = hidden_states.clone().detach().cpu().double()

    return hidden_states


def enable_qwen_custom_decoderlayer(layer, layer_id):
    """
    Replace the forward function of Qwen2DecoderLayer with a custom forward function.
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        qwen_custom_decoderlayer_forward, layer
    )
