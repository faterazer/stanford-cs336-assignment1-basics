import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
