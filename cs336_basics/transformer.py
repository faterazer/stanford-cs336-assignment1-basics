import math

import torch
from torch import nn, Tensor


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
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1_3 = Linear(d_model, 2 * d_ff)
        self.w2 = Linear(d_ff, d_model)

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


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    in_type = x.dtype
    x = x.to(torch.float32)
    max_val = x.amax(dim=dim, keepdim=True)
    x = x - max_val
    x_exp = x.exp()
    x_out = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return x_out.to(in_type)


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None, scale: float | None = None
) -> Tensor:
    """
    Compute the scaled dot product attention.

    Args:
        query (Tensor): The query tensor of shape (batch_size, ..., seq_len, embed_dim).
        key (Tensor): The key tensor of shape (batch_size, ..., seq_len, embed_dim).
        value (Tensor): The value tensor of shape (batch_size, ..., seq_len, embed_dim).
        attn_mask (Optional[Tensor]): The attention mask tensor of shape (..., seq_len, seq_len).
        scale (Optional[float]): The scale factor for the dot product. If None, it defaults to 1 / sqrt(embed_dim).

    Returns:
        Tensor: The tensor of shape (batch_size, ..., seq_len, embed_dim).
    """
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if attn_mask is not None:
        attn_scores.masked_fill_(attn_mask.logical_not(), float("-inf"))
    attn_weights = softmax(attn_scores, dim=-1)
    outputs = torch.matmul(attn_weights, value)
    return outputs


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"Expected d_model % num_heads == 0, got d_model = {d_model}, num_heads = {num_heads}.")
        if rope is not None and rope.d_k != d_model // num_heads:
            raise ValueError(
                f"Expected rope.d_k == d_model // num_heads, got rope.d_k = {rope.d_k}, d_model // num_heads = {d_model // num_heads}."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.qkv_proj = Linear(d_model, d_model * 3)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        prefix_shape, seq_len = x.shape[:-2], x.shape[-2]
        qkv = self.qkv_proj(x)

        # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_model // num_heads)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.reshape(*prefix_shape, seq_len, self.num_heads, -1).transpose(-2, -3)
        key = key.reshape(*prefix_shape, seq_len, self.num_heads, -1).transpose(-2, -3)
        value = value.reshape(*prefix_shape, seq_len, self.num_heads, -1).transpose(-2, -3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        pos = torch.arange(seq_len, device=x.device)
        causal_mask = pos[None, :] <= pos[:, None]
        outputs = scaled_dot_product_attention(query, key, value, causal_mask)
        outputs = outputs.transpose(-2, -3)
        outputs = outputs.reshape(*prefix_shape, seq_len, self.d_model)
        outputs = self.out_proj(outputs)
        return outputs


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.causal_attention = CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        y = self.norm1(x)
        y = self.causal_attention(y, token_positions)
        x = x + y

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + y
        return x


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = self.token_embeddings(token_ids)
        if token_positions is None:
            token_positions = torch.arange(token_ids.shape[-1], device=x.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.norm_final(x)
        x = self.lm_head(x)
        return x
