import torch
from torch import nn

from simple_ml.utils.activations import SwiGLU


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.swiglu = SwiGLU(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.swiglu(x))


class Attention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_key_value_heads: int, head_dim: int
    ):
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads

        # define weights
        self.W_q = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.W_k = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.W_v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.W_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_heads // self.num_key_value_heads

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_embeddings(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # apply rotary embeddings to queries and keys
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot

    def get_causal_mask(
        self, A: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        causal_mask = torch.full(A.shape[-2:], float("-inf"), device=A.device)
        causal_mask = causal_mask.triu(1)[None, None, ...].repeat(A.shape[0], 1, 1, 1)

        return causal_mask.masked_fill(
            attention_mask[:, None, None, :] == 0, float("-inf")
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        q = self.W_q(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        k = self.W_k(
            hidden_states
        )  # [batch_size, seq_len, num_key_value_heads * head_dim]
        v = self.W_v(
            hidden_states
        )  # [batch_size, seq_len, num_key_value_heads * head_dim]

        # reshape q, k, v into [batch_size, num_heads, seq_len, head_dim]
        q = q.view(q.size(0), q.size(1), -1, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), -1, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), -1, self.head_dim).transpose(1, 2)

        # apply rotary embeddings to queries and keys
        cos, sin = position_embeddings
        q, k = self.apply_rotary_embeddings(q, k, cos, sin)

        # repeat keys and values for each head
        k = torch.repeat_interleave(k, self.num_key_value_groups, dim=1)
        v = torch.repeat_interleave(v, self.num_key_value_groups, dim=1)

        # compute attention scores
        A = torch.einsum("abcd,abed->abce", q, k) / (self.head_dim**0.5)
        causal_mask = self.get_causal_mask(A, attention_mask)

        # apply causal mask
        A = A + causal_mask

        # apply softmax to A across the last dimension
        A = A.softmax(dim=-1)

        # apply attention to values
        o = torch.einsum(
            "abcd,abde->abce", A, v
        )  # [batch_size, num_heads, seq_len, head_dim]

        # reshape o into [batch_size, seq_len, num_heads * head_dim]
        o = o.transpose(1, 2).contiguous()
        o = o.view(o.size(0), o.size(1), -1)

        # apply output projection
        o = self.W_o(o)

        return o
