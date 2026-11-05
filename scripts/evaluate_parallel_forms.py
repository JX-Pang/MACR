from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from macr.config import load_config
from macr.data import ItemResponseDataset, MACRCollator, load_json, make_prompt
from macr.inference.parallel_form import predict_parallel_forms
from macr.metrics import compute_metrics
from macr.training.factory import build_macr_model, load_recalibrator_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate MACR with parallel-form reliability fusion.')
    parser.add_argument('--config', default='configs/macr_xes3g5m.yaml')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--variants-path', default=None)
    parser.add_argument('--lora-dir', default=None)
    parser.add_argument('--head-path', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fold_cfg = load_json(cfg.paths.fold_config_path)
    num_students = int(fold_cfg['student_num'])
    variants_path = args.variants_path or cfg.paths.generated_dir / f'{args.split}_parallel_forms.jsonl'
    split_path = {'val': cfg.paths.val_path, 'test': cfg.paths.test_path}[args.split]
    variant_names = cfg.inference.form_names[:cfg.inference.parallel_forms]
    if len(variant_names) != cfg.inference.parallel_forms:
        raise ValueError('inference.form_names must contain at least parallel_forms entries')
    variant_rows = {}
    with open(variants_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('debug_fallback'):
                raise ValueError(f"Item {row['item_id']} contains debug fallback forms")
            missing = set(variant_names) - set(row.get('variants', {}))
            if missing:
                raise ValueError(f"Item {row['item_id']} is missing forms: {sorted(missing)}")
            variant_rows[int(row['item_id'])] = row
    model, tokenizer = build_macr_model(cfg, inference=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lora_dir = cfg.paths.run_dir / 'best_lora' if args.lora_dir is None else args.lora_dir
    head_path = cfg.paths.run_dir / 'best_head.pt' if args.head_path is None else args.head_path
    load_recalibrator_checkpoint(model, lora_dir, head_path, device)
    model.to(device).eval()
    dataset = ItemResponseDataset(cfg.paths.question_path, cfg.paths.kc_path, split_path, num_students, variants_path=variants_path)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=MACRCollator(tokenizer, cfg.train.max_length))
    all_labels: list[float] = []
    all_probs: list[float] = []
    all_variance: list[float] = []
    for batch in loader:
        item_ids = batch['item_id']
        missing_items = set(item_ids.tolist()) - set(variant_rows)
        if missing_items:
            raise ValueError(f'Missing parallel forms for item ids: {sorted(missing_items)}')
        variant_prompts = []
        for name in variant_names:
            texts = []
            for variants in batch['variants']:
                texts.append(make_prompt(variants[name]))
            variant_prompts.append(texts)
        original_prompts = [make_prompt(text) for text in batch['texts']]
        fused = predict_parallel_forms(model, tokenizer, item_ids, original_prompts, variant_prompts, batch['kc_ids'], batch['kc_mask'], cfg.train.max_length, cfg.inference.reliability_tau, device)
        labels = batch['labels'].to(device)
        mask = labels != -1.0
        all_probs.extend(fused['prob'][mask].detach().cpu().numpy())
        all_labels.extend(labels[mask].detach().cpu().numpy())
        all_variance.extend(fused['variance'][mask].detach().cpu().numpy())
    metrics = compute_metrics(np.asarray(all_labels), np.asarray(all_probs))
    metrics['parallel_form_variance'] = float(np.mean(all_variance)) if all_variance else 0.0
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
