import types
from typing import Optional, Tuple

import torch


def mpt_custom_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
    # MPT Block structure is norm -> attention -> residual -> norm -> mlp -> residual
    
    # 1. Layer Norm before Attention
    residual = hidden_states
    hidden_states = self.norm_1(hidden_states)

    # 2. Self Attention
    attn_outputs, self_attn_weights, present_key_value = self.attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = attn_outputs[0] # attn_outputs is a tuple
    
    hidden_states = residual + hidden_states

    # 3. Layer Norm before MLP
    residual = hidden_states
    hidden_states = self.norm_2(hidden_states)
    
    # 4. MLP
    hidden_states = self.ffn(hidden_states)
    
    hidden_states = residual + hidden_states
    
    # Save the final hidden state of the block
    self.feat = hidden_states.clone().detach().cpu().double()

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def enable_mpt_custom_block(layer, layer_id):
    """
    Replace the forward function of MPTBlock with a custom forward function.
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        mpt_custom_block_forward, layer
    )

# Note: MPTAttention forward pass is complex and might not need patching 
# if we only capture the final hidden state from the block.
# We will add a custom attention forward if deeper analysis is needed later.
