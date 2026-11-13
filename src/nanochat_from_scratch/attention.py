import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import einsum

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Project hidden states into Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Project concatenated heads back to hidden size
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.hidden_size

        # (B, T, C) -> (B, H, T, D)
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h = self.num_heads)

        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h = self.num_heads)

        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h = self.num_heads)

        # QKT -> (B, H, T, D) @ ((B, H, D, T) => (B, H, T, T)

        scores = einsum(q, k, "b h t d, b h s d -> b h t s")
        scores /= math.sqrt(self.head_dim)

        # Disallow attention to future tokens
        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Numerically stable softmax
        weights = F.softmax(scores.float(), dim=-1).to(v.dtype)

        # Weighted sum of value vectors
        context = weights @ v  # (B H T D)

        # Concatenate the heads
        context = rearrange(
            context,
            "b h t d -> b t (h d)",
        )
        return self.out_proj(context)
        
        
        
        
        
        
        
        
        
        
        

    
