from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F

class ContrastivePairIndex:

    def __init__(self, rows: list[dict]) -> None:
        self.by_anchor = {int(row['anchor_item_id']): row for row in rows}

    @classmethod
    def from_jsonl(cls, path: str | Path | None) -> 'ContrastivePairIndex | None':
        if not path or not Path(path).exists():
            return None
        rows = []
        with Path(path).open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return cls(rows)

    def groups_for_batch(self, anchor_item_ids: torch.Tensor, allowed_item_ids: set[int]) -> list[dict[str, int | list[int]]]:
        groups: list[dict[str, int | list[int]]] = []
        for item_id in anchor_item_ids.detach().cpu().tolist():
            anchor_id = int(item_id)
            row = self.by_anchor.get(anchor_id)
            if not row:
                continue
            positives = [int(value) for value in row.get('positive_item_ids', []) if int(value) in allowed_item_ids and int(value) != anchor_id]
            negatives = [int(value) for value in row.get('negative_item_ids', []) if int(value) in allowed_item_ids and int(value) != anchor_id]
            if positives and negatives:
                groups.append({'anchor': anchor_id, 'positives': positives, 'negatives': negatives})
        return groups

    @staticmethod
    def required_item_ids(groups: list[dict[str, int | list[int]]]) -> list[int]:
        item_ids: set[int] = set()
        for group in groups:
            item_ids.add(int(group['anchor']))
            item_ids.update((int(value) for value in group['positives']))
            item_ids.update((int(value) for value in group['negatives']))
        return sorted(item_ids)

def _set_nce(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    numerator = torch.logsumexp(pos_scores, dim=0)
    denominator = torch.logsumexp(torch.cat([pos_scores, neg_scores]), dim=0)
    return denominator - numerator

def batch_contrastive_loss(item_ids: torch.Tensor, measurements: torch.Tensor, logits: torch.Tensor, groups: list[dict[str, int | list[int]]], tau_meas: float, tau_beh: float) -> tuple[torch.Tensor, torch.Tensor]:
    if tau_meas <= 0 or tau_beh <= 0:
        raise ValueError('Contrastive temperatures must be positive')
    if not groups:
        zero = measurements.new_tensor(0.0)
        return (zero, zero)
    id_to_row = {int(item_id): row for row, item_id in enumerate(item_ids.detach().cpu().tolist())}
    normalized_measurements = F.normalize(measurements.float(), dim=-1)
    probabilities = torch.sigmoid(logits.float())
    measurement_losses = []
    behavior_losses = []
    for group in groups:
        anchor_row = id_to_row[int(group['anchor'])]
        positive_rows = torch.tensor([id_to_row[int(value)] for value in group['positives']], device=measurements.device)
        negative_rows = torch.tensor([id_to_row[int(value)] for value in group['negatives']], device=measurements.device)
        anchor_measurement = normalized_measurements[anchor_row]
        positive_measurement_scores = normalized_measurements[positive_rows] @ anchor_measurement / tau_meas
        negative_measurement_scores = normalized_measurements[negative_rows] @ anchor_measurement / tau_meas
        measurement_losses.append(_set_nce(positive_measurement_scores, negative_measurement_scores))
        anchor_behavior = probabilities[anchor_row]
        positive_distances = (probabilities[positive_rows] - anchor_behavior.unsqueeze(0)).pow(2).mean(dim=-1)
        negative_distances = (probabilities[negative_rows] - anchor_behavior.unsqueeze(0)).pow(2).mean(dim=-1)
        behavior_losses.append(_set_nce(-positive_distances / tau_beh, -negative_distances / tau_beh))
    return (torch.stack(measurement_losses).mean(), torch.stack(behavior_losses).mean())
