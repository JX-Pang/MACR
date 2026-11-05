from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class PathConfig:
    project_root: Path = Path('.')
    dataset: str = 'XES3G5M'
    fold: str = 'fold1'
    model_id: str = 'meta-llama/Llama-3.2-3B-Instruct'
    output_root: Path = Path('runs')
    generated_root: Path = Path('generated')

    @property
    def model_source(self) -> str:
        return os.environ.get('MACR_MODEL_PATH', self.model_id)

    @property
    def dataset_root(self) -> Path:
        return self.project_root / 'data' / self.dataset

    @property
    def fold_root(self) -> Path:
        return self.dataset_root / 'cold_questions' / self.fold

    @property
    def question_path(self) -> Path:
        return self.dataset_root / 'questions_and_kc_map' / 'questions_id_map.json'

    @property
    def kc_path(self) -> Path:
        return self.dataset_root / 'questions_and_kc_map' / 'kc_id_map.json'

    @property
    def train_path(self) -> Path:
        return self.fold_root / 'train.json'

    @property
    def val_path(self) -> Path:
        return self.fold_root / 'val.json'

    @property
    def test_path(self) -> Path:
        return self.fold_root / 'test.json'

    @property
    def fold_config_path(self) -> Path:
        return self.fold_root / 'config.json'

    @property
    def phase1_dir(self) -> Path:
        return self.project_root / 'phase1_save' / self.dataset

    @property
    def phase2_dir(self) -> Path:
        return self.project_root / 'phase2_save' / self.dataset

    @property
    def run_dir(self) -> Path:
        return self.project_root / self.output_root / self.dataset / self.fold

    @property
    def generated_dir(self) -> Path:
        return self.project_root / self.generated_root / self.dataset / self.fold

@dataclass
class TrainConfig:
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 5e-05
    max_length: int = 512
    lambda_meas: float = 0.5
    lambda_beh: float = 1.0
    tau_meas: float = 0.1
    tau_beh: float = 0.1
    align_weight: float = 0.5
    seed: int = 42
    use_prednet: bool = True
    interaction_type: str = 'ncd'

@dataclass
class LoraConfigValues:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

@dataclass
class InferenceConfig:
    parallel_forms: int = 3
    reliability_tau: float = 0.1
    form_names: list[str] = field(default_factory=lambda: ['heuristic', 'logic_distillation', 'isomorphic'])

@dataclass
class MACRConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoraConfigValues = field(default_factory=LoraConfigValues)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f'Unknown config field: {key}')
        current = getattr(instance, key)
        if isinstance(current, Path):
            value = Path(value)
        setattr(instance, key, value)

def load_config(path: str | Path) -> MACRConfig:
    cfg = MACRConfig()
    config_path = Path(path).resolve()
    with config_path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    if 'paths' in raw:
        _update_dataclass(cfg.paths, raw['paths'])
    if 'train' in raw:
        _update_dataclass(cfg.train, raw['train'])
    if 'lora' in raw:
        _update_dataclass(cfg.lora, raw['lora'])
    if 'inference' in raw:
        _update_dataclass(cfg.inference, raw['inference'])
    if not cfg.paths.project_root.is_absolute():
        cfg.paths.project_root = (config_path.parent / cfg.paths.project_root).resolve()
    else:
        cfg.paths.project_root = cfg.paths.project_root.resolve()
    return cfg
