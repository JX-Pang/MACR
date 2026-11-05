from __future__ import annotations
import argparse
import json
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from macr.config import load_config
from macr.data import ItemResponseDataset, MACRCollator, load_json
from macr.training.contrastive import ContrastivePairIndex, batch_contrastive_loss
from macr.training.evaluate import evaluate_item_centered
from macr.training.factory import build_macr_model, save_recalibrator_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(description='Train MACR item recalibration.')
    parser.add_argument('--config', default='configs/macr_xes3g5m.yaml')
    parser.add_argument('--contrastive-pairs', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.paths.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
    fold_cfg = load_json(cfg.paths.fold_config_path)
    num_students = int(fold_cfg['student_num'])
    model, tokenizer = build_macr_model(cfg, inference=False)
    collator = MACRCollator(tokenizer, max_length=cfg.train.max_length)
    train_data = ItemResponseDataset(cfg.paths.question_path, cfg.paths.kc_path, cfg.paths.train_path, num_students, teacher_params_path=cfg.paths.phase2_dir / 'teacher_params.pt')
    val_data = ItemResponseDataset(cfg.paths.question_path, cfg.paths.kc_path, cfg.paths.val_path, num_students)
    train_loader = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_data, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collator)
    pair_path = args.contrastive_pairs or cfg.paths.generated_dir / 'contrastive_pairs.jsonl'
    pair_index = ContrastivePairIndex.from_jsonl(pair_path)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_auc = -1.0
    for epoch in range(cfg.train.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f'epoch {epoch + 1}/{cfg.train.epochs}')
        for batch in progress:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            groups = pair_index.groups_for_batch(batch['item_id'], train_data.item_id_set) if pair_index is not None else []
            if groups:
                pair_item_ids = pair_index.required_item_ids(groups)
                pair_features = [train_data.item_feature(item_id, include_labels=False, include_teacher=False) for item_id in pair_item_ids]
                pair_batch = collator(pair_features)
                pair_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in pair_batch.items()}
                pair_outputs = model(**pair_batch)
                meas_loss, beh_loss = batch_contrastive_loss(pair_batch['item_id'], pair_outputs['measurement'], pair_outputs['logits'], groups, cfg.train.tau_meas, cfg.train.tau_beh)
                loss = loss + cfg.train.lambda_meas * meas_loss + cfg.train.lambda_beh * beh_loss
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f'{float(loss.detach().cpu()):.4f}')
        metrics = evaluate_item_centered(model, val_loader, device)
        print(f'epoch={epoch + 1} val={json.dumps(metrics, ensure_ascii=False)}')
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            save_recalibrator_checkpoint(model, cfg.paths.run_dir / 'best_lora', cfg.paths.run_dir / 'best_head.pt', cfg.paths.model_id)
            (cfg.paths.run_dir / 'best_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
