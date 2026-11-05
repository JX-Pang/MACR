from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosLinear(nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, torch.clamp(self.weight, min=0.0), self.bias)

class NCDInteraction(nn.Module):

    def __init__(self, hidden_size: int, use_prednet: bool=True, prednet_hidden: int=512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_prednet = use_prednet
        if self.use_prednet:
            self.prednet = nn.Sequential(PosLinear(hidden_size, prednet_hidden), nn.Sigmoid(), nn.Dropout(p=0.2), PosLinear(prednet_hidden, prednet_hidden // 2), nn.Sigmoid(), nn.Dropout(p=0.2), PosLinear(prednet_hidden // 2, 1), nn.Sigmoid())

    def forward(self, anchors: torch.Tensor, q: torch.Tensor, beta: torch.Tensor, alpha: torch.Tensor | None=None) -> torch.Tensor:
        theta = torch.sigmoid(anchors).unsqueeze(0)
        beta = torch.sigmoid(beta).unsqueeze(1)
        q = q.unsqueeze(1)
        x = (theta - beta) * q
        if self.use_prednet:
            if alpha is not None:
                x = alpha.float().unsqueeze(1) * x
            prob = self.prednet(x).squeeze(-1)
            prob = torch.clamp(prob, min=1e-07, max=1.0 - 1e-07)
            logits = torch.log(prob / (1.0 - prob))
        else:
            scale = math.sqrt(self.hidden_size) / 3.0
            if alpha is not None:
                logits = alpha.float() * x.float().sum(dim=-1) / scale
            else:
                logits = x.float().sum(dim=-1) / scale
        return logits
