from __future__ import annotations
import math
import torch
import torch.nn as nn

class TargetAwareAttention(nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, key_mask: torch.Tensor | None=None) -> torch.Tensor:
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(keys)
        v = self.value_proj(keys)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        if key_mask is not None:
            scores = scores.masked_fill(key_mask.unsqueeze(1) == 0, -10000.0)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, v).squeeze(1)
