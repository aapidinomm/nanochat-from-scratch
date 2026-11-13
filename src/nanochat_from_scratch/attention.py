import torch.nn as nn
import torch.nn.Functional as F
from einops import rearrange

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear()

        
        

    
