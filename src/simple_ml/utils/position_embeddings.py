import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, base: float, d: int):
        super().__init__()

        self.base = base
        self.d = d

    @torch.no_grad()
    def forward(self, position_ids):
        """Adds rotary embeddings to hidden states.

        Args:
            position_ids: tensor of shape [batch_size, seq_len]

        Returns:
            cos: tensor of shape [batch_size, seq_len, self.d]
            sin: tensor of shape [batch_size, seq_len, self.d]
        """

        # compute angle frequencies
        theta = 1 / (self.base ** (torch.arange(0, self.d, 2) / self.d))

        # for each position, compute the angle
        angle = torch.einsum("a,bc->bca", theta, position_ids)

        anglex2 = angle.repeat(1, 1, 2)
        sin, cos = anglex2.sin(), anglex2.cos()

        return cos, sin
