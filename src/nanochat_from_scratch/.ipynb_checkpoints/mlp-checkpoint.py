import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.fc = nn.Linear(
            hidden_size, 4*hidden_size, bias=False
        )
        self.proj = nn.Linear(
            4*hidden_size, hidden_size, bias = False
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.proj(x)
        return x