import torch
from einops import einsum

def rotary_cos_sin(seq_len: int, head_dim: int, device: torch.device, base: float = 100000.0):
    assert head_dim % 2 == 0
    
    # positions identifies where each token is located in the sequence. 
    # creates a tensor containing token positions from 0 to n-1.
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # channels identifies which pair of dimensions within an attention head gets rotated.
    # Since RoPE rotates dimensions in pairs, and each pair requires only one rotation frequency.
    channels = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)

    '''
    The frequencies are typically calculated using:
\[
\theta_i=\frac{1}{\text{base}^{2i/d_{\text{head}}}}
\]
    '''
    inv_freq = 1.0 / (base ** (channels / head_dim))

    '''
    t corresponds to the token-position dimension.
    f corresponds to the frequency dimension.
    t f means we want an output containing every combination of token position and frequency.

    # (T,) x (D/2,) -> (T, D/2)
    '''
    angles = einsum(
        positions,
        inv_freq,
        "t, f -> t f",
    )
    
    # Broadcast across batch and attention heads
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    return cos, sin

def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    # cos, sin: (1, 1, T, D/2)

    assert x.shape[-1] % 2 == 0

    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    '''
    This code is applying ROPE using the split-half rotation convention. This code splits the query or key vector (which is provided by x) into two halves.
    '''
    x1, x2 = x.chunk(2, dim=-1) 

    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos

    return torch.cat([y1, y2], dim=-1)

    

    
    