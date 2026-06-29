import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, d_model)
        in_type = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(mean_square + self.eps)
        result = x * rms_inv * self.weight
        return result.to(in_type)


class SwiGLUFFN(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.w1_3 = Linear(d_model, 2 * d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out, w3_out = torch.chunk(self.w1_3(x), 2, dim=-1)
        hidden = w1_out * torch.sigmoid(w1_out) * w3_out
        return self.w2(hidden)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding.

    Applies RoPE to the last dimension of `x`.

    Expected shapes:
        x: (..., seq_len, d_k)
        token_positions: broadcastable to (..., seq_len)

    The last dimension `d_k` is interpreted as interleaved 2D pairs:
        (x_0, x_1), (x_2, x_3), ...
    """

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.d_k = d_k

        rope_base = torch.tensor(theta, dtype=torch.float32, device=device)
        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = rope_base.pow(-dim_indices / d_k)

        positions = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)

        cos_cached = angles.cos()
        sin_cached = angles.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to `x`.

        Args:
            x: Tensor of shape (..., seq_len, d_k).
            token_positions: Integer tensor with shape compatible with x.shape[:-1].
                Each value is the absolute position index of the corresponding token.

        Returns:
            Tensor with the same shape and dtype as `x`.
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected x.shape[-1] == {self.d_k}, got {x.shape[-1]}")

        in_type = x.dtype
        x = x.to(torch.float32)
        rope_cos = self.cos_cached[token_positions]
        rope_sin = self.sin_cached[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * rope_cos - x_odd * rope_sin
        out_odd = x_odd * rope_cos + x_even * rope_sin
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(in_type)
