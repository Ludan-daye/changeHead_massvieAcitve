import types
from typing import Optional, Tuple

import torch


def opt_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    position_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
    # OPT DecoderLayer structure: residual -> norm -> attention -> residual -> norm -> mlp
    
    residual = hidden_states

    # Self-Attention with layer norm
    if self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        past_key_values=past_key_values,
        position_ids=position_ids,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        cache_position=cache_position,
        **kwargs,
    )
    
    hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    if not self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Fully Connected
    hidden_states_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states

    if self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    hidden_states = (residual + hidden_states).view(hidden_states_shape)

    if not self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    # Save the final hidden state of the block
    self.feat = hidden_states.clone().detach().cpu().double()

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (past_key_values,)

    return outputs


def enable_opt_custom_decoderlayer(layer, layer_id):
    """
    Replace the forward function of OPTDecoderLayer with a custom forward function.
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        opt_custom_decoderlayer_forward, layer
    )
