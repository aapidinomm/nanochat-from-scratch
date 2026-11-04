import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean_squared = reduce(x.float().square(), "... hidden -> ... 1", mean)
        
        # Normalize each token independently.
        inv_rms = torch.rsqrt(mean_squared + self.eps)
        
        return x * inv_rms.to(x.dtype)