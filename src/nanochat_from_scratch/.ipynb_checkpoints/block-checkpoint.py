
import torch.nn as nn

from nanochat_from_scratch.model import RMSNorm
from nanochat_from_scratch.attention import CausalSelfAttention
from nanochat_from_scratch.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int,
                num_heads: int, num_kv_heads: int | None):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = CausalSelfAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp_norm = RMSNorm()
        self.mlp = MLP(hidden_size)

    def forward(self, x):
        # Pre-norm attention with residual
        x = x + self.attn(self.attn_norm(x))

        # Pre-norm MLP with residual
        x = x + self.mlp(self.mlp_norm(x))
        return x
        