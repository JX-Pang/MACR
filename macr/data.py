from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def load_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

def load_torch_file(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

def make_prompt(text: str) -> str:
    return f'你是一名教育测量与学科教学专家。请分析下列题目所考查的知识概念、解题机制、数量关系和难度线索。不要预测任何学生的作答结果。\n【题目】{text}'

class ItemResponseDataset(Dataset):

    def __init__(self, question_path: str | Path, kc_path: str | Path, interaction_path: str | Path, num_students: int, teacher_params_path: str | Path | None=None, variants_path: str | Path | None=None) -> None:
        self.questions: dict[str, dict[str, Any]] = load_json(question_path)
        self.kcs: dict[str, dict[str, Any]] = load_json(kc_path)
        self.interactions: list[dict[str, Any]] = load_json(interaction_path)
        self.num_students = int(num_students)
        self.teacher_params = load_torch_file(teacher_params_path) if teacher_params_path else None
        self.variants = self._load_variants(variants_path)
        self.item_ids, self.labels = self._build_item_labels()
        self.item_id_set = set(self.item_ids)
        self.label_by_item = dict(zip(self.item_ids, self.labels))

    def _load_variants(self, variants_path: str | Path | None) -> dict[int, dict[str, str]]:
        if not variants_path or not Path(variants_path).exists():
            return {}
        variants: dict[int, dict[str, str]] = {}
        with Path(variants_path).open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                variants[int(row['item_id'])] = row['variants']
        return variants

    def _build_item_labels(self) -> tuple[list[int], list[torch.Tensor]]:
        grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for row in self.interactions:
            grouped[int(row['item_id'])].append((int(row['user_id']), float(row['score'])))
        item_ids = sorted(grouped)
        labels: list[torch.Tensor] = []
        for item_id in item_ids:
            y = torch.full((self.num_students,), -1.0, dtype=torch.float32)
            for user_id, score in grouped[item_id]:
                if not 0 <= user_id < self.num_students:
                    raise ValueError(f'user_id={user_id} is outside [0, {self.num_students}) for item_id={item_id}')
                y[user_id] = score
            labels.append(y)
        return (item_ids, labels)

    def __len__(self) -> int:
        return len(self.item_ids)

    def item_feature(self, item_id: int, *, include_labels: bool=True, include_teacher: bool=True) -> dict[str, Any]:
        item_id = int(item_id)
        if item_id not in self.item_id_set:
            raise KeyError(f'Item {item_id} is not part of this source split')
        item = self.questions[str(item_id)]
        kc_ids = item.get('kc_ids', [0]) or [0]
        text = item.get('content', '')
        sample: dict[str, Any] = {'item_id': item_id, 'text': text, 'prompt': make_prompt(text), 'kc_ids': torch.tensor(kc_ids, dtype=torch.long)}
        if include_labels:
            sample['labels'] = self.label_by_item[item_id]
        if item_id in self.variants:
            sample['variants'] = self.variants[item_id]
        if include_teacher and self.teacher_params is not None:
            sample['teacher_q'] = self.teacher_params['q'][item_id]
            sample['teacher_beta'] = self.teacher_params['beta'][item_id]
            if 'alpha' in self.teacher_params:
                sample['teacher_alpha'] = self.teacher_params['alpha'][item_id]
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.item_feature(self.item_ids[index])

class MACRCollator:

    def __init__(self, tokenizer, max_length: int=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        tokenized = self.tokenizer([f['prompt'] for f in features], max_length=self.max_length, truncation=True, padding=True, return_tensors='pt')
        kc_ids = pad_sequence([f['kc_ids'] for f in features], batch_first=True, padding_value=-1)
        batch: dict[str, Any] = {'item_id': torch.tensor([f['item_id'] for f in features], dtype=torch.long), 'input_ids_prompt': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'kc_ids': kc_ids, 'kc_mask': (kc_ids != -1).long(), 'texts': [f['text'] for f in features]}
        if all(('labels' in f for f in features)):
            batch['labels'] = torch.stack([f['labels'] for f in features])
        if all(('teacher_q' in f for f in features)):
            batch['teacher_q'] = torch.stack([f['teacher_q'] for f in features])
            batch['teacher_beta'] = torch.stack([f['teacher_beta'] for f in features])
            if all(('teacher_alpha' in f for f in features)):
                batch['teacher_alpha'] = torch.stack([f['teacher_alpha'] for f in features])
        if any(('variants' in f for f in features)):
            batch['variants'] = [f.get('variants', {}) for f in features]
        return batch
