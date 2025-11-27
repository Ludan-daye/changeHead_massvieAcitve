"""
Falcon模型的Monkey Patch
用于捕获中间激活和禁用attention heads
"""

import torch
from typing import Optional, Tuple


def falcon_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    **kwargs,
):
    """Falcon decoder layer的自定义forward"""
    residual = hidden_states
    
    # Falcon-7B使用单个input_layernorm（并行attention和MLP）
    if hasattr(self, 'input_layernorm'):
        hidden_states = self.input_layernorm(hidden_states)
    
    # Self attention
    attn_outputs = self.self_attention(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        alibi=alibi,
        **kwargs,
    )
    
    attn_output = attn_outputs[0]
    outputs = attn_outputs[1:]  # (present, attentions)
    
    # MLP
    mlp_output = self.mlp(hidden_states)
    
    # Falcon-7B: attention和MLP输出都加到residual上
    hidden_states = residual + attn_output + mlp_output
    
    # 保存激活用于分析
    self.feat = hidden_states.clone().detach().cpu().double()
    
    if use_cache:
        return (hidden_states,) + outputs
    return (hidden_states,)


def enable_falcon_custom_decoderlayer(layer, layer_id):
    """启用Falcon decoder layer的自定义forward"""
    import types
    layer.forward = types.MethodType(falcon_custom_decoderlayer_forward, layer)
    print(f"Enabled custom forward for FalconDecoderLayer layer {layer_id}")


def falcon_attention_forward_with_head_disable(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    **kwargs,
):
    """Falcon attention的forward，支持禁用特定heads"""
    # 调用原始forward
    outputs = self._original_forward(
        hidden_states,
        alibi=alibi,
        attention_mask=attention_mask,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )
    
    attn_output = outputs[0]
    
    # 如果有需要禁用的heads
    if hasattr(self, 'disabled_heads') and self.disabled_heads:
        batch_size, seq_len, hidden_dim = attn_output.shape
        num_heads = self.num_heads
        head_dim = hidden_dim // num_heads
        
        # 重塑为 (batch, seq, heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
        
        # 禁用指定heads
        for head_id in self.disabled_heads:
            attn_output[:, :, head_id, :] = 0
        
        # 重塑回原形状
        attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
    
    return (attn_output,) + outputs[1:]


def register_falcon_head_disabler(layer, heads_to_disable):
    """注册Falcon的head禁用器"""
    import types
    
    attn = layer.self_attention
    
    # 保存原始forward
    if not hasattr(attn, '_original_forward'):
        attn._original_forward = attn.forward
    
    # 设置要禁用的heads
    attn.disabled_heads = heads_to_disable
    
    # 替换forward
    attn.forward = types.MethodType(falcon_attention_forward_with_head_disable, attn)
