from __future__ import annotations
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

def load_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_exercise_text(text: str) -> str:
    if not text:
        return ''
    text = re.sub('(\\d+)\\s*[×x]\\s*(\\d+)', '$\\1 \\\\times \\2$', text)
    text = re.sub('(\\d+)\\s*÷\\s*(\\d+)', '$\\1 \\\\div \\2$', text)
    text = re.sub('(\\d+)\\s*([\\+\\-])\\s*(\\d+)', '$\\1 \\2 \\3$', text)
    text = re.sub('\\[图片\\]|\\[image\\]|<img[^>]*>|\\[formula_img\\]', '[图示]', text, flags=re.IGNORECASE)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def build_cold_start_folds(interactions: list[dict[str, Any]], questions: dict[str, dict[str, Any]], min_student_logs: int=15, min_exercise_responses: int=10, target_ratio: float=0.2, val_ratio: float=0.1, num_folds: int=5, subsample_students: int | None=None, seed: int=42) -> list[dict[str, Any]]:
    valid_qids = set()
    for qid_str, qdata in questions.items():
        kcs = qdata.get('kc_ids', [])
        if kcs and len(kcs) > 0:
            valid_qids.add(int(qid_str))
    student_counts: dict[int, int] = defaultdict(int)
    exercise_counts: dict[int, int] = defaultdict(int)
    for row in interactions:
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        if iid in valid_qids:
            student_counts[uid] += 1
            exercise_counts[iid] += 1
    active_students = {uid for uid, cnt in student_counts.items() if cnt >= min_student_logs}
    active_exercises = {iid for iid, cnt in exercise_counts.items() if cnt >= min_exercise_responses}
    print(f'Qualified active students (>= {min_student_logs} logs): {len(active_students)}')
    print(f'Qualified active exercises (>= {min_exercise_responses} logs & has KC): {len(active_exercises)}')
    if subsample_students and len(active_students) > subsample_students:
        rng_sub = random.Random(seed)
        sampled_uids = set(rng_sub.sample(sorted(active_students), subsample_students))
        active_students = sampled_uids
        print(f'Subsampled students to: {len(active_students)}')
    valid_interactions = [row for row in interactions if int(row['user_id']) in active_students and int(row['item_id']) in active_exercises]
    print(f'Total valid interactions: {len(valid_interactions)}')
    student_map = {old: new for new, old in enumerate(sorted(active_students))}
    for row in valid_interactions:
        row['user_id'] = student_map[int(row['user_id'])]
        row['item_id'] = int(row['item_id'])
        row['score'] = float(row['score'])
    num_students = len(student_map)
    sorted_exercises = sorted(active_exercises)
    folds = []
    for fold_idx in range(num_folds):
        fold_seed = seed + fold_idx * 100
        rng = random.Random(fold_seed)
        shuffled_exercises = list(sorted_exercises)
        rng.shuffle(shuffled_exercises)
        n_target = int(len(shuffled_exercises) * target_ratio)
        target_set = set(shuffled_exercises[:n_target])
        source_candidates = shuffled_exercises[n_target:]
        rng.shuffle(source_candidates)
        n_val = int(len(source_candidates) * val_ratio)
        val_set = set(source_candidates[:n_val])
        train_set = set(source_candidates[n_val:])
        train_rows = [r for r in valid_interactions if r['item_id'] in train_set]
        val_rows = [r for r in valid_interactions if r['item_id'] in val_set]
        test_rows = [r for r in valid_interactions if r['item_id'] in target_set]
        folds.append({'fold': fold_idx + 1, 'student_num': num_students, 'exercise_num': len(sorted_exercises), 'train': train_rows, 'val': val_rows, 'test': test_rows, 'stats': {'train_exercises': len(train_set), 'val_exercises': len(val_set), 'test_exercises': len(target_set), 'train_logs': len(train_rows), 'val_logs': len(val_rows), 'test_logs': len(test_rows)}})
    return folds

def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 0: Complete data preprocessing and cold-start split generator for MACR.')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw/XES3G5M'), help='Directory containing raw questions, kcs, and interaction files.')
    parser.add_argument('--output-dir', type=Path, default=Path('data/XES3G5M'), help='Directory to save the processed splits.')
    parser.add_argument('--min-student-logs', type=int, default=15)
    parser.add_argument('--min-exercise-responses', type=int, default=10)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--subsample-students', type=int, default=None, help='Optional student subsampling size (e.g. 1154 for lightweight split, or None for full)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(f'=== MACR Phase 0: Data Preprocessing ===')
    print(f'Raw directory: {args.raw_dir}')
    print(f'Output directory: {args.output_dir}')
    q_file = args.raw_dir / 'questions_id_map.json'
    k_file = args.raw_dir / 'kc_id_map.json'
    i_file = args.raw_dir / 'interactions.json'
    if not (q_file.exists() and k_file.exists() and i_file.exists()):
        print(f'\n[INFO] Raw files not found at {args.raw_dir}.')
        print('Expected:')
        print(f'  {q_file}')
        print(f'  {k_file}')
        print(f'  {i_file}')
        print('\nNote: MACR uses the preprocessed dataset from the LMCD baseline:')
        print('  https://github.com/TAL-auroraX/LMCD')
        print('If you already have LMCD preprocessed data, place it directly under:')
        print(f'  {args.output_dir}/cold_questions/fold1/')
        print(f'  {args.output_dir}/questions_and_kc_map/')
        return
    questions = load_json(q_file)
    kcs = load_json(k_file)
    interactions = load_json(i_file)
    for qid, qdata in questions.items():
        if 'content' in qdata:
            qdata['content'] = normalize_exercise_text(qdata['content'])
    folds = build_cold_start_folds(interactions=interactions, questions=questions, min_student_logs=args.min_student_logs, min_exercise_responses=args.min_exercise_responses, num_folds=args.num_folds, subsample_students=args.subsample_students, seed=args.seed)
    save_json(questions, args.output_dir / 'questions_and_kc_map' / 'questions_id_map.json')
    save_json(kcs, args.output_dir / 'questions_and_kc_map' / 'kc_id_map.json')
    for f in folds:
        fold_dir = args.output_dir / 'cold_questions' / f"fold{f['fold']}"
        save_json(f['train'], fold_dir / 'train.json')
        save_json(f['val'], fold_dir / 'val.json')
        save_json(f['test'], fold_dir / 'test.json')
        save_json({'exercise_num': f['exercise_num'], 'student_num': f['student_num'], 'knowledge_num': len(kcs)}, fold_dir / 'config.json')
        print(f"Fold {f['fold']} saved: {f['stats']}")
    print('\nPhase 0 preprocessing completed successfully!')
if __name__ == '__main__':
    main()
