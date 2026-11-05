from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels != -1.0
    if not torch.any(mask):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())

def supervised_alignment_loss(pred_q: torch.Tensor, pred_beta: torch.Tensor, pred_alpha: torch.Tensor, teacher_q: torch.Tensor | None=None, teacher_beta: torch.Tensor | None=None, teacher_alpha: torch.Tensor | None=None) -> torch.Tensor:
    loss = pred_q.new_tensor(0.0)
    if teacher_q is not None:
        loss = loss + F.mse_loss(pred_q.float(), teacher_q.float())
    if teacher_beta is not None:
        loss = loss + F.mse_loss(pred_beta.float(), teacher_beta.float())
    if teacher_alpha is not None:
        loss = loss + F.mse_loss(pred_alpha.float(), teacher_alpha.float())
    return loss
