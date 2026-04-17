import types
from typing import Optional, Tuple

import torch
from torch import nn


def gpt2_custom_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_values: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    hidden_states = attn_output + residual

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    # Save feature for visualization
    self.feat = hidden_states.clone().detach().cpu().double()

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def gpt2_attention_hook(module, input, output):
    """Hook to capture attention probabilities from GPT2Attention."""
    # Request attention outputs by calling with output_attentions=True
    # Output format: (attn_output, present, attentions) if output_attentions else (attn_output, present)
    # We need to capture this during the actual forward pass
    pass


def gpt2_custom_attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    past_key_values: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    """Custom GPT2Attention forward that captures attention probabilities."""

    # Call original forward with output_attentions=True to get attention weights
    # Store the original forward if not already stored
    if not hasattr(self, '_original_forward'):
        # This shouldn't happen, but just in case
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
        self._original_forward = GPT2Attention.forward

    # Call with output_attentions to get the weights
    outputs = self._original_forward(
        self,
        hidden_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        attention_mask=attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=True,  # Force to get attention weights
        **kwargs
    )

    # Extract attention weights from output
    # Output format: (attn_output, present_key_value, attention_weights)
    if len(outputs) >= 3:
        attn_weights = outputs[2]
        # Store attention probabilities
        self.attn_probs = attn_weights.clone().detach().cpu().double()

    # Return outputs in expected format based on original output_attentions flag
    if not output_attentions and len(outputs) >= 3:
        # Remove attention weights if not requested
        outputs = outputs[:2]

    return outputs


def enable_gpt2_custom_block(layer, layer_id):
    """
    Enable custom forward function for GPT2Block to capture features.

    Args:
        layer: The GPT2Block layer to modify
        layer_id: The index of the layer
    """
    layer.forward = types.MethodType(gpt2_custom_block_forward, layer)
    layer.layer_id = layer_id
    print(f"Enabled custom forward for GPT2Block layer {layer_id}")


def enable_gpt2_custom_attention(layer, layer_id):
    """
    Enable custom forward function for GPT2Attention to capture attention patterns.

    Args:
        layer: The GPT2Block layer containing the attention module
        layer_id: The index of the layer
    """
    # Store original forward before replacing
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    layer.attn._original_forward = GPT2Attention.forward

    layer.attn.forward = types.MethodType(gpt2_custom_attention_forward, layer.attn)
    layer.attn.layer_id = layer_id
    print(f"Enabled custom attention for GPT2Block layer {layer_id}")
