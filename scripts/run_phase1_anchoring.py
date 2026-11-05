from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from macr.config import load_config
from macr.data import load_json
from macr.models.interactions import PosLinear
EXERCISE_INSTRUCTION = 'Analyze the cognitive requirement and mathematical logic of the following exercise: {stem}'
KC_INSTRUCTION = 'Define the educational knowledge concept: {name}'

def extract_llm_embeddings(texts: list[str], model: nn.Module, tokenizer: Any, batch_size: int=8, max_length: int=512, device: torch.device=torch.device('cpu')) -> torch.Tensor:
    all_embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokenized = tokenizer(batch_texts, max_length=max_length, truncation=True, padding=True, return_tensors='pt').to(device)
            outputs = model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'], output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            last_token = tokenized['attention_mask'].long().sum(dim=1).sub(1).clamp_min(0)
            row_index = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[row_index, last_token]
            normalized = F.normalize(pooled.float(), dim=-1)
            all_embeddings.append(normalized.cpu())
    return torch.cat(all_embeddings, dim=0)

class Stage1TeacherCDM(nn.Module):

    def __init__(self, num_students: int, num_exercises: int, latent_dim: int, cdm_type: str='ncd', prednet_hidden: int=512) -> None:
        super().__init__()
        self.cdm_type = cdm_type
        self.latent_dim = latent_dim
        self.theta = nn.Parameter(torch.empty(num_students, latent_dim))
        self.q = nn.Parameter(torch.empty(num_exercises, latent_dim))
        self.beta = nn.Parameter(torch.empty(num_exercises, latent_dim))
        self.alpha = nn.Parameter(torch.empty(num_exercises, 1))
        nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.beta)
        nn.init.xavier_uniform_(self.alpha)
        if cdm_type == 'ncd':
            self.prednet = nn.Sequential(PosLinear(latent_dim, prednet_hidden), nn.Sigmoid(), nn.Dropout(p=0.2), PosLinear(prednet_hidden, prednet_hidden // 2), nn.Sigmoid(), nn.Dropout(p=0.2), PosLinear(prednet_hidden // 2, 1), nn.Sigmoid())

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        theta_i = torch.sigmoid(self.theta[user_ids])
        q_j = torch.sigmoid(self.q[item_ids])
        beta_j = torch.sigmoid(self.beta[item_ids])
        alpha_j = torch.sigmoid(self.alpha[item_ids])
        x = (theta_i - beta_j) * q_j
        if self.cdm_type == 'ncd':
            x = alpha_j * x
            prob = self.prednet(x).squeeze(-1)
            prob = torch.clamp(prob, min=1e-07, max=1.0 - 1e-07)
            return prob
        else:
            scale = math.sqrt(self.latent_dim) / 3.0
            logit = (alpha_j * x).sum(dim=-1) / scale
            return torch.sigmoid(logit)

def train_stage1_anchors(interactions: list[dict[str, Any]], num_students: int, num_exercises: int, latent_dim: int, cdm_type: str='ncd', lr: float=0.001, weight_decay: float=0.0001, batch_size: int=1024, epochs: int=50, prednet_hidden: int=512, device: torch.device=torch.device('cpu')) -> dict[str, Any]:
    user_ids = torch.tensor([int(r['user_id']) for r in interactions], dtype=torch.long)
    item_ids = torch.tensor([int(r['item_id']) for r in interactions], dtype=torch.long)
    scores = torch.tensor([float(r['score']) for r in interactions], dtype=torch.float32)
    dataset = TensorDataset(user_ids, item_ids, scores)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Stage1TeacherCDM(num_students=num_students, num_exercises=num_exercises, latent_dim=latent_dim, cdm_type=cdm_type, prednet_hidden=prednet_hidden).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for b_users, b_items, b_scores in loader:
            b_users = b_users.to(device)
            b_items = b_items.to(device)
            b_scores = b_scores.to(device)
            probs = model(b_users, b_items)
            loss = F.binary_cross_entropy(probs, b_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_users)
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f'  Stage 1 Epoch [{epoch + 1}/{epochs}] BCE Loss: {avg_loss:.5f}')
    model.eval()
    params: dict[str, Any] = {'theta': model.theta.detach().cpu(), 'q': model.q.detach().cpu(), 'beta': model.beta.detach().cpu(), 'alpha': model.alpha.detach().cpu()}
    if cdm_type == 'ncd':
        params['prednet_state_dict'] = {k: v.detach().cpu() for k, v in model.prednet.state_dict().items()}
    return params

def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 1 & 2: Student Anchoring & Semantic Prior Extraction.')
    parser.add_argument('--config', default='configs/macr_xes3g5m.yaml')
    parser.add_argument('--cdm-type', default='ncd', choices=['ncd', 'mirt'])
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip LLM embedding extraction if already cached.')
    parser.add_argument('--embedding-batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001, help='Stage 1 learning rate (Table III: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=1024, help='Stage 1 batch size (Table III: 1024)')
    parser.add_argument('--epochs', type=int, default=50, help='Stage 1 epochs (Table III: 50)')
    parser.add_argument('--prednet-hidden', type=int, default=512, help='PredNet hidden dim (Table III: 512)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    phase1_dir = cfg.paths.phase1_dir
    phase2_dir = cfg.paths.phase2_dir
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_embeddings:
        print('\n' + '=' * 60)
        print('Stage 1 Part A: Extracting LLM Semantic Priors')
        print('=' * 60)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_source = cfg.paths.model_source
        print(f'Loading backbone LLM: {model_source}')
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        llm = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype).to(device)
        kcs = load_json(cfg.paths.kc_path)
        kc_ids = sorted((int(k) for k in kcs.keys()))
        kc_texts = [KC_INSTRUCTION.format(name=kcs[str(kid)].get('name', str(kid))) for kid in kc_ids]
        print(f'Extracting embeddings for {len(kc_texts)} Knowledge Concepts...')
        kc_emb = extract_llm_embeddings(kc_texts, llm, tokenizer, args.embedding_batch_size, device=device)
        torch.save(kc_emb, phase1_dir / 'kc_embeddings_base.pt')
        print(f"Saved KC embeddings to {phase1_dir / 'kc_embeddings_base.pt'} (shape: {kc_emb.shape})")
        questions = load_json(cfg.paths.question_path)
        item_ids = sorted((int(k) for k in questions.keys()))
        item_texts = [EXERCISE_INSTRUCTION.format(stem=questions[str(iid)].get('content', '')) for iid in item_ids]
        print(f'Extracting embeddings for {len(item_texts)} Exercises...')
        item_emb = extract_llm_embeddings(item_texts, llm, tokenizer, args.embedding_batch_size, device=device)
        torch.save(item_emb, phase1_dir / 'item_embeddings_base.pt')
        print(f"Saved Item embeddings to {phase1_dir / 'item_embeddings_base.pt'} (shape: {item_emb.shape})")
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f'\nSkipping semantic embedding extraction (--skip-embeddings).')
    print('\n' + '=' * 60)
    print(f'Stage 1 Part B: Metric-Preserving Student Anchoring ({args.cdm_type.upper()})')
    print('=' * 60)
    fold_cfg = load_json(cfg.paths.fold_config_path)
    num_students = int(fold_cfg['student_num'])
    interactions = load_json(cfg.paths.train_path)
    questions = load_json(cfg.paths.question_path)
    num_exercises = max((int(k) for k in questions.keys())) + 1
    kcs = load_json(cfg.paths.kc_path)
    latent_dim = len(kcs)
    print(f'Configuration:')
    print(f'  - Students: {num_students}')
    print(f'  - Exercises: {num_exercises}')
    print(f'  - Latent Dimension (d = |K|): {latent_dim}')
    print(f'  - CDM Type: {args.cdm_type}')
    print(f'  - Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})')
    print(f'  - Batch Size: {args.batch_size}, Epochs: {args.epochs}')
    params = train_stage1_anchors(interactions=interactions, num_students=num_students, num_exercises=num_exercises, latent_dim=latent_dim, cdm_type=args.cdm_type, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs, prednet_hidden=args.prednet_hidden, device=device)
    out_file = phase2_dir / 'teacher_params.pt'
    torch.save(params, out_file)
    print(f'\nStudent anchors and teacher parameters saved to {out_file}:')
    print(f"  - Student Anchors (A): {params['theta'].shape}")
    print(f"  - Knowledge Demand (q): {params['q'].shape}")
    print(f"  - Difficulty (beta): {params['beta'].shape}")
    print(f"  - Discrimination (alpha): {params['alpha'].shape}")
    if 'prednet_state_dict' in params:
        print(f"  - Monotonic PredNet (psi): {len(params['prednet_state_dict'])} weight tensors")
    print('\nStage 1 completed successfully!')
if __name__ == '__main__':
    main()
