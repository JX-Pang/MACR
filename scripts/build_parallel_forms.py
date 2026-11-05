from __future__ import annotations
import argparse
import json
import os
from typing import Any
from macr.config import load_config
from macr.data import load_json
COMMON_CONSTRAINTS = '\n必须同时满足：\n1. 保持原题的知识点、求解机制、数字、运算关系、难度和正确答案不变；\n2. 不得增加或删除影响答案的条件；\n3. 不输出答案、解析、说明或标题，只输出改写后的完整题目。\n'
PROMPTS = {'heuristic': '在严格保持原题测量机制不变的前提下，为题目加入教师式、非泄题的分步提示，引导学生识别条件和数量关系。\n{constraints}\n{context}', 'logic_distillation': '去除题目的生活叙事和无关修饰，只保留核心逻辑、数量关系、必要条件与求解要求。\n{constraints}\n{context}', 'isomorphic': '完全替换题目的生活情境和表层措辞，但保持所有数学结构与测量机制不变。\n{constraints}\n{context}'}

def call_zhipu(prompt: str, model: str) -> str | None:
    api_key = os.environ.get('ZHIPUAI_API_KEY')
    if not api_key:
        return None
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': '你是严谨的教育测量与学科题目改写专家。你必须保持题目的认知机制和正确答案不变。'}, {'role': 'user', 'content': prompt}], temperature=0.3)
    content = response.choices[0].message.content
    return content.strip() if content else None

def deterministic_fallback(content: str, variant_name: str) -> str:
    prefix = {'heuristic': '请分步思考：先识别条件，再列出数量关系，最后完成计算。题目：', 'logic_distillation': '核心问题：', 'isomorphic': '等价改写题：'}[variant_name]
    return prefix + content

def format_context(item: dict[str, Any], kcs: dict[str, dict[str, Any]]) -> str:
    kc_names = [kcs.get(str(kc_id), {}).get('name', str(kc_id)) for kc_id in item.get('kc_ids', [])]
    return f"原题：{item.get('content', '')}\n选项：{json.dumps(item.get('options', {}), ensure_ascii=False)}\n原解析：{item.get('analysis', '')}\n正确答案：{json.dumps(item.get('answer', []), ensure_ascii=False)}\n知识点：{json.dumps(kc_names, ensure_ascii=False)}"

def main() -> None:
    parser = argparse.ArgumentParser(description='Build MACR mechanism-preserving parallel forms.')
    parser.add_argument('--config', default='configs/macr_xes3g5m.yaml')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model', default='glm-4')
    parser.add_argument('--allow-fallback', action='store_true', help='Allow prefix-only debug forms when ZHIPUAI_API_KEY is unavailable.')
    args = parser.parse_args()
    if not os.environ.get('ZHIPUAI_API_KEY') and (not args.allow_fallback):
        raise RuntimeError('ZHIPUAI_API_KEY is not set. Export prompts with a valid key, or use --allow-fallback only for a pipeline smoke test.')
    cfg = load_config(args.config)
    cfg.paths.generated_dir.mkdir(parents=True, exist_ok=True)
    questions = load_json(cfg.paths.question_path)
    kcs = load_json(cfg.paths.kc_path)
    split_path = {'train': cfg.paths.train_path, 'val': cfg.paths.val_path, 'test': cfg.paths.test_path}[args.split]
    interactions = load_json(split_path)
    item_ids = sorted({int(row['item_id']) for row in interactions})
    if args.limit:
        item_ids = item_ids[:args.limit]
    out_path = cfg.paths.generated_dir / f'{args.split}_parallel_forms.jsonl'
    done = set()
    if out_path.exists():
        with out_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    done.add(int(json.loads(line)['item_id']))
    with out_path.open('a', encoding='utf-8') as f:
        for item_id in item_ids:
            if item_id in done:
                continue
            item = questions[str(item_id)]
            content = item.get('content', '')
            context = format_context(item, kcs)
            variants = {'original': content}
            used_fallback = False
            for name, template in PROMPTS.items():
                prompt = template.format(constraints=COMMON_CONSTRAINTS.strip(), context=context)
                generated = call_zhipu(prompt, args.model)
                if generated:
                    variants[name] = generated
                else:
                    used_fallback = True
                    variants[name] = deterministic_fallback(content, name)
            row = {'item_id': item_id, 'variants': variants, 'debug_fallback': used_fallback}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
    relative_output = out_path.relative_to(cfg.paths.project_root).as_posix()
    print(f'Wrote {len(item_ids) - len(done)} item records to {relative_output}')
if __name__ == '__main__':
    main()
