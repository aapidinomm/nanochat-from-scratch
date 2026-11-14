import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (rearrange, einsum, repeat)
from nanochat_from_scratch.rope import (
    rotary_cos_sin,
    apply_rope,
)
from nanochat_from_scratch.model import RMSNorm

class CausalSelfAttention(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 num_kv_heads: int | None = None,):
        super().__init__()

        assert hidden_size % num_heads == 0
        
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = (num_kv_heads if num_kv_heads is not None else num_heads)
        self.group_size = self.num_heads // self.num_kv_heads

        assert self.head_dim % 2 == 0

        # Project hidden states into Q, K, V
        # K and V use fewer heads in GQA: num_kv_heads * head_dim outputs
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=False)

        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()

        # Project concatenated heads back to hidden size
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.hidden_size

        # (B, T, C) -> (B, H, T, D)
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h = self.num_heads)

        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h = self.num_kv_heads)

        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h = self.num_kv_heads)

        # Generate position-dependent rotations.
        cos, sin = rotary_cos_sin(
            seq_len=T,
            head_dim=self.head_dim,
            device=x.device,
        )

        # Apply RoPE to queries and keys.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Apply QK Norm - Normalization occurs independently for each [D] vector in the [B, H, T, D] tensor.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Value vectors remain unchanged.

                
        context = F.scaled_dot_product_attention(q, k, v,
            is_causal=True,
            enable_gqa = (self.num_kv_heads != self.num_heads)
        )

        # # QKT -> (B, H, T, D) @ ((B, H, D, T) => (B, H, T, T)

        # k = repeat(k, "b g t d -> b (g r) t d", r=self.group_size)
        # v = repeat(v, "b g t d -> b (g r) t d", r=self.group_size)

        
        # scores = einsum(q, k, "b h t d, b h s d -> b h t s")
        # scores /= math.sqrt(self.head_dim)

        # # Disallow attention to future tokens
        # causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        # scores = scores.masked_fill(~causal_mask, float("-inf"))

        # # Numerically stable softmax
        # weights = F.softmax(scores.float(), dim=-1).to(v.dtype)

        # # Weighted sum of value vectors
        # context = weights @ v  # (B H T D)

        # Concatenate the heads
        context = rearrange(
            context,
            "b h t d -> b t (h d)",
        )
        return self.out_proj(context)
        
        
        
        
        
        
        
        
        
        
        

    
