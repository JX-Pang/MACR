from __future__ import annotations
import argparse
import json
import torch
import torch.nn.functional as F
from macr.config import load_config
from macr.data import load_json, load_torch_file

def jaccard(left: set[int], right: set[int]) -> float:
    if not left and (not right):
        return 0.0
    return len(left & right) / max(len(left | right), 1)

def main() -> None:
    parser = argparse.ArgumentParser(description='Build source-only MACR mechanism positives and hard negatives.')
    parser.add_argument('--config', default='configs/macr_xes3g5m.yaml')
    parser.add_argument('--split', default='train', choices=['train'])
    parser.add_argument('--top-k', type=int, default=80)
    parser.add_argument('--num-positives', type=int, default=3)
    parser.add_argument('--num-negatives', type=int, default=3)
    parser.add_argument('--positive-kc-threshold', type=float, default=0.8)
    parser.add_argument('--negative-kc-threshold', type=float, default=0.2)
    parser.add_argument('--max-items', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.paths.generated_dir.mkdir(parents=True, exist_ok=True)
    questions = load_json(cfg.paths.question_path)
    interactions = load_json(cfg.paths.train_path)
    embedding_path = cfg.paths.phase1_dir / 'item_embeddings_base.pt'
    if not embedding_path.exists():
        raise FileNotFoundError(f'Missing item embedding file: {embedding_path}')
    embeddings = F.normalize(load_torch_file(embedding_path).float(), dim=-1)
    item_ids = sorted({int(row['item_id']) for row in interactions})
    if args.max_items:
        item_ids = item_ids[:args.max_items]
    item_ids = [item_id for item_id in item_ids if str(item_id) in questions and item_id < embeddings.shape[0]]
    if len(item_ids) < 3:
        raise ValueError('At least three valid source items are required')
    item_tensor = torch.tensor(item_ids, dtype=torch.long)
    source_embeddings = embeddings[item_tensor]
    local_by_item = {item_id: index for index, item_id in enumerate(item_ids)}
    kc_sets = {item_id: set((int(kc_id) for kc_id in questions[str(item_id)].get('kc_ids', []))) for item_id in item_ids}
    items_by_kc: dict[int, set[int]] = {}
    for item_id, kc_ids in kc_sets.items():
        for kc_id in kc_ids:
            items_by_kc.setdefault(kc_id, set()).add(item_id)
    out_path = cfg.paths.generated_dir / 'contrastive_pairs.jsonl'
    report_path = cfg.paths.generated_dir / 'contrastive_pairs_report.json'
    kept = 0
    missing: list[dict[str, int | str]] = []
    with out_path.open('w', encoding='utf-8') as f:
        for local_index, item_id in enumerate(item_ids):
            anchor_kcs = kc_sets[item_id]
            anchor_embedding = source_embeddings[local_index]
            positive_pool: set[int] = set()
            for kc_id in anchor_kcs:
                positive_pool.update(items_by_kc.get(kc_id, set()))
            positive_pool.discard(item_id)
            positive_pool = {candidate for candidate in positive_pool if jaccard(anchor_kcs, kc_sets[candidate]) >= args.positive_kc_threshold}
            positive_ids = sorted(positive_pool, key=lambda candidate: float(torch.dot(anchor_embedding, source_embeddings[local_by_item[candidate]])))[:args.num_positives]
            similarities = source_embeddings @ anchor_embedding
            neighbor_rows = torch.topk(similarities, k=min(args.top_k + 1, similarities.numel())).indices.tolist()
            negative_ids = []
            for neighbor_row in neighbor_rows:
                candidate = item_ids[int(neighbor_row)]
                if candidate == item_id:
                    continue
                if jaccard(anchor_kcs, kc_sets[candidate]) <= args.negative_kc_threshold:
                    negative_ids.append(candidate)
                if len(negative_ids) >= args.num_negatives:
                    break
            if positive_ids and negative_ids:
                row = {'anchor_item_id': item_id, 'positive_item_ids': positive_ids, 'negative_item_ids': negative_ids, 'candidate_split': 'train', 'rule': 'positive=KC-aligned_and_lexically_divergent; negative=semantic_neighbor_and_KC-divergent'}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                kept += 1
            else:
                missing.append({'item_id': item_id, 'reason': 'no_positive_or_negative'})
    report = {'dataset': cfg.paths.dataset, 'fold': cfg.paths.fold, 'candidate_split': 'train', 'source_items': len(item_ids), 'pairs': kept, 'missing': len(missing), 'missing_examples': missing[:50], 'output': out_path.relative_to(cfg.paths.project_root).as_posix()}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
