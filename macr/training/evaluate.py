from __future__ import annotations
import numpy as np
import torch
from macr.metrics import compute_metrics

@torch.no_grad()
def evaluate_item_centered(model, dataloader, device) -> dict[str, float]:
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs['logits'])
        labels = batch['labels']
        mask = labels != -1.0
        all_probs.extend(probs[mask].detach().cpu().numpy())
        all_labels.extend(labels[mask].detach().cpu().numpy())
    return compute_metrics(np.asarray(all_labels), np.asarray(all_probs))
