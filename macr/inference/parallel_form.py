from __future__ import annotations
import torch

def reliability_fusion(prob_stack: torch.Tensor, tau: float=0.1) -> dict[str, torch.Tensor]:
    if prob_stack.ndim != 3:
        raise ValueError(f'Expected [T, B, N] probabilities, got {tuple(prob_stack.shape)}')
    if prob_stack.size(0) < 1:
        raise ValueError('At least one parallel form is required')
    if tau <= 0:
        raise ValueError('Reliability temperature tau must be positive')
    mean_prob = prob_stack.mean(dim=0)
    deviation = (prob_stack - mean_prob.unsqueeze(0)).pow(2)
    weights = torch.softmax(-deviation / tau, dim=0)
    fused = (weights * prob_stack).sum(dim=0)
    return {'prob': fused, 'mean': mean_prob, 'variance': deviation.mean(dim=0), 'weights': weights}

@torch.no_grad()
def predict_parallel_forms(model, tokenizer, item_ids: torch.Tensor, original_texts: list[str] | None, variant_texts: list[list[str]], kc_ids: torch.Tensor, kc_mask: torch.Tensor, max_length: int, tau: float, device: torch.device) -> dict[str, torch.Tensor]:
    all_probs = []
    if original_texts is not None:
        orig_tokenized = tokenizer(original_texts, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        orig_out = model(item_id=item_ids.to(device), input_ids_prompt=orig_tokenized['input_ids'].to(device), attention_mask=orig_tokenized['attention_mask'].to(device), kc_ids=kc_ids.to(device), kc_mask=kc_mask.to(device))
        all_probs.append(torch.sigmoid(orig_out['logits']).detach())
    for texts in variant_texts:
        tokenized = tokenizer(texts, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        outputs = model(item_id=item_ids.to(device), input_ids_prompt=tokenized['input_ids'].to(device), attention_mask=tokenized['attention_mask'].to(device), kc_ids=kc_ids.to(device), kc_mask=kc_mask.to(device))
        all_probs.append(torch.sigmoid(outputs['logits']).detach())
    prob_stack = torch.stack(all_probs, dim=0)
    return reliability_fusion(prob_stack, tau=tau)
