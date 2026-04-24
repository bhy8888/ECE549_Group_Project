#!/usr/bin/env python3
"""
Training entry point.

Usage:
    python scripts/train.py --config configs/vit_base.yaml
    python scripts/train.py --config configs/vit_base.yaml --epochs 50 --lr 1e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_vit.data.dataset import build_dataloaders
from apple_vit.models.vit_classifier import ViTClassifier
from apple_vit.training.metrics import print_classification_report, get_confusion_matrix
from apple_vit.training.trainer import Trainer
from apple_vit.utils.config import Config, set_seed
from apple_vit.utils.logger import get_logger
from apple_vit.visualization.plot_utils import plot_confusion_matrix, plot_training_curves

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT for Apple Disease Classification")
    parser.add_argument("--config", type=str, default="configs/vit_base.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment name (for checkpoint/log dirs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.experiment_name is not None:
        cfg.output.experiment_name = args.experiment_name

    set_seed(cfg.train.seed)
    logger = get_logger("train", cfg.log_dir)
    logger.info(f"Config: {cfg}")

    # --- Data ---
    logger.info("Loading datasets…")
    train_loader, val_loader = build_dataloaders(
        cfg.data,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        use_weighted_sampler=True,
    )
    logger.info(
        f"Train: {len(train_loader.dataset)} samples | "
        f"Val: {len(val_loader.dataset)} samples"
    )

    # Class weights for loss (inverse frequency)
    class_weights = train_loader.dataset.class_weights()
    logger.info(f"Class weights: {class_weights.tolist()}")

    # --- Model ---
    logger.info(f"Building model: {cfg.model.backbone}")
    model = ViTClassifier(cfg.model)
    param_info = model.count_parameters()
    logger.info(
        f"Parameters — total: {param_info['total'] / 1e6:.1f}M | "
        f"trainable: {param_info['trainable'] / 1e6:.1f}M"
    )

    # --- Train ---
    trainer = Trainer(model, train_loader, val_loader, cfg, class_weights=class_weights)
    history = trainer.fit()

    # --- Post-training evaluation ---
    logger.info("Running final evaluation on test set…")
    device = trainer.device
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    print_classification_report(all_labels, all_preds)
    cm = get_confusion_matrix(all_labels, all_preds)

    # --- Save figures ---
    fig_dir = cfg.figure_dir
    plot_training_curves(history, save_path=fig_dir / "training_curves.png")
    plot_confusion_matrix(cm, save_path=fig_dir / "confusion_matrix.png")
    logger.info(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
