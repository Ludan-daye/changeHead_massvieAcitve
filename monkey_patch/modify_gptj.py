"""
GPT-J模型的Monkey Patch
"""

import torch
from typing import Optional, Tuple


def gptj_custom_block_forward(
    self,
    hidden_states: Optional[torch.FloatTensor],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor]:
    """GPT-J block的自定义forward"""
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    
    attn_outputs = self.attn(
        hidden_states=hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]
    outputs = attn_outputs[1:]
    
    feed_forward_hidden_states = self.mlp(hidden_states)
    hidden_states = attn_output + feed_forward_hidden_states + residual
    
    # 保存激活
    self.feat = hidden_states.clone().detach().cpu().double()
    
    if use_cache:
        return (hidden_states,) + outputs
    return (hidden_states,)


def enable_gptj_custom_block(layer, layer_id):
    """启用GPT-J block的自定义forward"""
    import types
    layer.forward = types.MethodType(gptj_custom_block_forward, layer)
    print(f"Enabled custom forward for GPTJBlock layer {layer_id}")
