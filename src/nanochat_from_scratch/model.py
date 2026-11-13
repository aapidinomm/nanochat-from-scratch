import torch
import torch.nn as nn
from einops import reduce


class RMSNorm(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        '''
        The ... hidden -> ... 1 expression ensures that the reduction 
        collapses only the last dimension and preserves all leading dimensions for broadcasting.
        '''        
        mean_squared = reduce(
            x.float().square(),
            "... hidden -> ... 1",
            "mean",
        )
        inv_rms = torch.rsqrt(mean_squared + self.eps)
        # normalize each token's hidden vector by multiplying all of its features 
        # by the inverse RMS, using the original input precision for the final multiplication
        return x * inv_rms.to(x.dtype)
        
        