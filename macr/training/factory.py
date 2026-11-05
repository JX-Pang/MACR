from __future__ import annotations
import json
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from macr.config import MACRConfig
from macr.data import load_torch_file
from macr.models.item_recalibrator import MACRItemRecalibrator

def load_tokenizer_and_model(cfg: MACRConfig, inference: bool=False):
    model_source = cfg.paths.model_source
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    llm_config = AutoConfig.from_pretrained(model_source)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_llm = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=inference, r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout, target_modules=list(cfg.lora.target_modules))
    base_llm = get_peft_model(base_llm, peft_config)
    base_llm.peft_config['default'].base_model_name_or_path = cfg.paths.model_id
    return (tokenizer, llm_config, base_llm)

def build_macr_model(cfg: MACRConfig, inference: bool=False) -> tuple[MACRItemRecalibrator, object]:
    tokenizer, llm_config, base_llm = load_tokenizer_and_model(cfg, inference=inference)
    teacher_params = load_torch_file(cfg.paths.phase2_dir / 'teacher_params.pt')
    kc_embeddings = load_torch_file(cfg.paths.phase1_dir / 'kc_embeddings_base.pt')
    use_prednet = getattr(cfg.train, 'use_prednet', True)
    model = MACRItemRecalibrator(llm_config, base_llm, student_anchors=teacher_params['theta'], kc_embeddings=kc_embeddings, align_weight=cfg.train.align_weight, use_prednet=use_prednet)
    if use_prednet and 'prednet_state_dict' in teacher_params:
        model.interaction.prednet.load_state_dict(teacher_params['prednet_state_dict'])
    return (model, tokenizer)

def save_recalibrator_checkpoint(model: MACRItemRecalibrator, adapter_dir: str | Path, head_path: str | Path, public_model_id: str) -> None:
    adapter_dir = Path(adapter_dir)
    head_path = Path(head_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    head_path.parent.mkdir(parents=True, exist_ok=True)
    model.encoder.peft_config['default'].base_model_name_or_path = public_model_id
    model.encoder.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_config_path = adapter_dir / 'adapter_config.json'
    adapter_config = json.loads(adapter_config_path.read_text(encoding='utf-8'))
    adapter_config['base_model_name_or_path'] = public_model_id
    adapter_config_path.write_text(json.dumps(adapter_config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (adapter_dir / 'README.md').write_text(f'---\nbase_model: {public_model_id}\nlibrary_name: peft\npipeline_tag: text-generation\ntags:\n- base_model:adapter:{public_model_id}\n- lora\n- transformers\n---\n\n# MACR LoRA Adapter\n\nAnonymous artifact for mechanism-aware cognitive recalibration.\n', encoding='utf-8')
    head_state = {key: value.detach().cpu() for key, value in model.state_dict().items() if 'encoder' not in key}
    torch.save(head_state, head_path)

def load_recalibrator_checkpoint(model: MACRItemRecalibrator, lora_dir: str | Path, head_path: str | Path, device) -> None:
    lora_dir = Path(lora_dir)
    head_path = Path(head_path)
    adapter_path = lora_dir / 'adapter_model.safetensors'
    if not adapter_path.exists():
        raise FileNotFoundError(f'LoRA checkpoint not found: {adapter_path}')
    if not head_path.exists():
        raise FileNotFoundError(f'Recalibration head checkpoint not found: {head_path}')
    weights = load_file(str(adapter_path), device='cpu')
    set_peft_model_state_dict(model.encoder, weights)
    head_state = load_torch_file(head_path)
    incompatible = model.load_state_dict(head_state, strict=False)
    allowed_missing = {'student_anchors', 'head_beta_kc.weight', 'head_alpha_kc.weight'}
    allowed_unexpected = {'student_theta.weight'}
    missing = {k for k in incompatible.missing_keys if not k.startswith('interaction.prednet')} - allowed_missing
    unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
    if missing or unexpected:
        raise RuntimeError(f'Incompatible head checkpoint; missing={sorted(missing)}, unexpected={sorted(unexpected)}')
