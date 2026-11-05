from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

class StudentAnchorTable(nn.Module):

    def __init__(self, anchors: torch.Tensor, trainable: bool=False) -> None:
        super().__init__()
        self.anchors = nn.Parameter(anchors.float(), requires_grad=trainable)

    @classmethod
    def from_teacher_params(cls, path: str | Path, trainable: bool=False) -> 'StudentAnchorTable':
        params = torch.load(path, map_location='cpu')
        if 'theta' not in params:
            raise KeyError(f"{path} does not contain teacher_params['theta']")
        return cls(params['theta'], trainable=trainable)

    def forward(self) -> torch.Tensor:
        return self.anchors
