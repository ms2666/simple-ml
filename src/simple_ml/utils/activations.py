import torch
from torch import nn

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()

        self.W = nn.Linear(hidden_size, output_size, bias=False)
        self.V = nn.Linear(hidden_size, output_size, bias=False)
        self.silu = SiLU()

    def forward(self, x):
        return self.silu(self.W(x)) * self.V(x)
