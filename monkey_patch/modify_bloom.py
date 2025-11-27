import types
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache


def bloom_custom_block_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Cache] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    """
    Custom forward function for BloomBlock that saves hidden states.
    BLOOM uses ALiBi (Attention with Linear Biases) instead of positional embeddings.
    """
    # hidden_states: [batch_size, seq_length, hidden_size]
    
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states)
    
    # Layer norm post the self attention.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    # Self attention.
    attention_output, attn_weights = self.self_attention(
        layernorm_output,
        residual,
        layer_past=layer_past,
        attention_mask=attention_mask,
        alibi=alibi,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
    )

    layernorm_output = self.post_attention_layernorm(attention_output)
    
    # Get residual
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = attention_output

    # MLP.
    output = self.mlp(layernorm_output, residual)

    # Save the final hidden state of the block
    self.feat = output.clone().detach().cpu().double()

    return output, attn_weights  # hidden_states, attentions


def enable_bloom_custom_block(layer, layer_id):
    """
    Replace the forward function of BloomBlock with a custom forward function.
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        bloom_custom_block_forward, layer
    )
