"""Configuration management — loads YAML and exposes typed dataclass."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml


@dataclass
class DataConfig:
    data_root: str = "data/apple_disease_classification"
    class_names: List[str] = field(
        default_factory=lambda: ["Normal_Apple", "Blotch_Apple", "Rot_Apple", "Scab_Apple"]
    )
    image_size: int = 224
    # Augmentation toggles
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = False
    random_rotation: float = 20.0
    color_jitter: bool = True
    random_erasing: bool = True


@dataclass
class ModelConfig:
    backbone: str = "google/vit-base-patch16-224-in21k"
    num_classes: int = 4
    dropout: float = 0.1
    freeze_encoder: bool = False  # fine-tune all layers by default


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"   # "cosine" | "step" | "none"
    warmup_epochs: int = 3
    early_stopping_patience: int = 8
    # Class-weighted loss for imbalanced dataset
    use_class_weights: bool = True
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42


@dataclass
class OutputConfig:
    output_dir: str = "outputs"
    experiment_name: str = "vit_base_run"
    save_best_only: bool = True
    log_interval: int = 10        # steps between console logs
    use_tensorboard: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output.output_dir) / "checkpoints" / self.output.experiment_name

    @property
    def log_dir(self) -> Path:
        return Path(self.output.output_dir) / "logs" / self.output.experiment_name

    @property
    def figure_dir(self) -> Path:
        return Path(self.output.output_dir) / "figures" / self.output.experiment_name

    def make_output_dirs(self) -> None:
        for p in [self.checkpoint_dir, self.log_dir, self.figure_dir]:
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if raw.get("data"):
            for k, v in raw["data"].items():
                setattr(cfg.data, k, v)
        if raw.get("model"):
            for k, v in raw["model"].items():
                setattr(cfg.model, k, v)
        if raw.get("train"):
            for k, v in raw["train"].items():
                setattr(cfg.train, k, v)
        if raw.get("output"):
            for k, v in raw["output"].items():
                setattr(cfg.output, k, v)
        return cfg


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
