import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x_scaled = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.w * x_scaled
