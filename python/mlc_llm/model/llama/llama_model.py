"""
Implementation for Llama3 architecture.
"""
import dataclasses
from typing import Any, Dict, Optional
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Llama3Config(ConfigBase):
    """Configuration for Llama3 models."""
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    rope_theta: float = 500000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    
    # Rope scaling parameters
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Optional FFN dimension multiplier
    ffn_dim_multiplier: Optional[float] = None
    
    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        
        self.head_dim = self.hidden_size // self.num_attention_heads


class Llama3Attention(nn.Module):
    """Multi-head attention for Llama3 architecture."""
    
    def __init__(self, config: Llama3Config, tp_mode: str = "none"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.tp_mode = tp_mode
        self.rope_mode = RopeMode.NORMAL
        self.config = config
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
    ) -> Tensor:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Project inputs to queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary positional embeddings
        if position_ids is not None:
            query_states, key_states = op_ext.apply_rotary_emb(
                query_states,
                key_states,
                position_ids,
                theta= self.config.rope_theta,
                scaling_factor=self.config.rope_scaling["factor"] if self.config.rope_scaling else None,
                mode=self.rope_mode,
            )
        
        # Handle KV caching
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states)
        
        # Repeat KV heads if num_heads > num_kv_heads
        if self.num_kv_heads < self.num_heads:
            key_states = op.repeat_interleave(key_states, self.num_heads // self.num_kv_heads, axis=2)
            value_states = op.repeat_interleave(value_states, self.num_heads // self.num_kv_heads, axis=2)
        
        # Calculate attention scores
        attn_output = op.attention(
            query_states, key_states, value_states, attention_mask, self.head_dim
        )
        
        # Project outputs back to hidden size
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class Llama3MLP(nn.Module):
    """Llama3 MLP module."""
    
    def __init__(self, config: Llama3Config, tp_mode: str = "none"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tp_mode = tp_mode
        
        # Apply FFN dimension multiplier if provided
        if config.ffn_dim_multiplier is not None:
            self.intermediate_size = int(config.ffn_dim_multiplier * self.intermediate_size)
        
        # Parallel FFN implementation with SiLU activation and gate
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SiLU activation with gate
        gate_output = op.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        
        # Element-wise multiplication of gate and up projection
        intermediate_output = gate_output * up_output
        
        # Project back to hidden size
        output = self.down_proj(intermediate_output)
        
        return output


class Llama3Layer(nn.Module):
    """Llama3 transformer layer."""
    
    def __init__(self, config: Llama3Config, layer_idx: int = 0, tp_mode: str = "none"):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_mode = tp_mode
        
        # Layer normalization
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Self-attention
        self.self_attn = Llama3Attention(config, tp_mode=tp_mode)
        
        # MLP
        self.mlp = Llama3MLP(config, tp_mode=tp_mode)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
    ) -> Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        
        hidden_states = residual + attn_output
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states)
        
        hidden_states = residual + mlp_output
        
        return hidden_states


class Llama3Model(nn.Module):
    """Llama3 model implementation."""
    
    def __init__(
        self, 
        config: Llama3Config, 
        tp_mode: str = "none", 
        start_layer: int = 0, 
        end_layer: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.tp_mode = tp_mode
        self.start_layer = start_layer
        self.end_layer = end_layer or config.num_hidden_layers
        
        # Embedding layer
        if self.start_layer == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList(
            [
                Llama3Layer(config, layer_idx=i, tp_mode=tp_mode)
                for i in range(self.start_layer, self.end_layer)
            ]
        )
        
        # Final layer normalization
        if self.end_layer == config.num_hidden_layers:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head
        if self.end_layer == config.num_hidden_layers:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
    ) -> Tensor:
        # Get embeddings
        if self.start_layer == 0:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_ids  # In this case, input_ids should already be embeddings
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        
        # Apply final normalization and LM head if this is the final block
        if self.end_layer == self.config.num_hidden_layers:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        
        return hidden_states