from __future__ import annotations
import argparse
import json
import torch
from torch.utils.data import DataLoader
from macr.config import load_config
from macr.data import ItemResponseDataset, MACRCollator, load_json
from macr.training.evaluate import evaluate_item_centered
from macr.training.factory import build_macr_model, load_recalibrator_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained MACR checkpoint.')
    parser.add_argument('--config', default='configs/macr_xes3g5m.yaml')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--lora-dir', default=None)
    parser.add_argument('--head-path', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fold_cfg = load_json(cfg.paths.fold_config_path)
    num_students = int(fold_cfg['student_num'])
    model, tokenizer = build_macr_model(cfg, inference=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lora_dir = cfg.paths.run_dir / 'best_lora' if args.lora_dir is None else args.lora_dir
    head_path = cfg.paths.run_dir / 'best_head.pt' if args.head_path is None else args.head_path
    load_recalibrator_checkpoint(model, lora_dir, head_path, device)
    model.to(device)
    split_path = {'train': cfg.paths.train_path, 'val': cfg.paths.val_path, 'test': cfg.paths.test_path}[args.split]
    dataset = ItemResponseDataset(cfg.paths.question_path, cfg.paths.kc_path, split_path, num_students)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=MACRCollator(tokenizer, cfg.train.max_length))
    metrics = evaluate_item_centered(model, loader, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
