import torch
from torch import nn

from simple_ml.utils.activations import SwiGLU
from simple_ml.utils.normalizations import RMSNorm
from simple_ml.utils.position_embeddings import RotaryEmbedding
from dataclasses import dataclass


@dataclass
class LlamaConfig:
    emb_hidden_size: int
    num_attn_heads: int
    num_key_value_heads: int
    mlp_intermediate_size: int
    num_layers: int
    rope_base: float
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.emb_hidden_size // self.num_attn_heads

    def __post_init__(self):
        assert (
            self.emb_hidden_size % self.num_attn_heads == 0
        ), "hidden_size must be divisible by num_heads"
        assert (
            self.emb_hidden_size % self.num_key_value_heads == 0
        ), "hidden_size must be divisible by num_key_value_heads"
        assert (
            self.num_attn_heads % self.num_key_value_heads == 0
        ), "num_heads must be divisible by num_key_value_heads"


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.swiglu = SwiGLU(config.emb_hidden_size, config.mlp_intermediate_size)
        self.down_proj = nn.Linear(
            config.mlp_intermediate_size, config.emb_hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(self.swiglu(x))


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        # define attributes
        self.hidden_size = config.emb_hidden_size
        self.num_heads = config.num_attn_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        # define weights
        self.W_q = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.W_k = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.W_v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.W_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.reshape_and_rotate = lambda x: x.view(
            x.size(0), x.size(1), -1, self.head_dim
        ).transpose(1, 2)

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ):
        q = self.W_q(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        k = self.W_k(
            hidden_states
        )  # [batch_size, seq_len, num_key_value_heads * head_dim]
        v = self.W_v(
            hidden_states
        )  # [batch_size, seq_len, num_key_value_heads * head_dim]

        # reshape q, k, v into [batch_size, num_heads, seq_len, head_dim]
        q = self.reshape_and_rotate(q)
        k = self.reshape_and_rotate(k)
        v = self.reshape_and_rotate(v)

        # apply rotary embeddings to queries and keys if provided
        if position_embeddings is not None:
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


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()

        self.hidden_size = config.emb_hidden_size

        self.input_norm = RMSNorm(config.emb_hidden_size)
        self.self_attention = Attention(config)
        self.attention_norm = RMSNorm(config.emb_hidden_size)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        # apply input normalization
        x = self.input_norm(hidden_states)

        # apply self attention
        attn_output = self.self_attention(x, attention_mask, position_embeddings)

        # add residual connection
        x = attn_output + hidden_states

        # apply attention normalization
        x = self.attention_norm(x)

        # apply mlp
        mlp_output = self.mlp(x)

        # add residual connection
        x = mlp_output + hidden_states

        return x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.rotary_embedding = RotaryEmbedding(
            base=config.rope_base, d=config.head_dim
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.emb_hidden_size)
        self.lm_head = nn.Linear(config.emb_hidden_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the LLamaModel.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token IDs.
            attention_mask (torch.Tensor): Tensor of shape (batch_size, seq_length) where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_length, vocab_size).
        """
        # Get embeddings
        hidden_states = self.embed_tokens(
            input_ids
        )  # [batch_size, seq_length, hidden_size]

        # Compute rotary position embeddings
        cos, sin = self.rotary_embedding(input_ids)

        # Pass through each decoder layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, (cos, sin))

        # Compute logits
        logits = self.lm_head(hidden_states)  # [batch_size, seq_length, vocab_size]

        return logits
