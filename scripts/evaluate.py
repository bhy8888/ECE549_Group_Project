#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the test set.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
        --config configs/vit_base.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from apple_vit.data.dataset import AppleDiseaseDataset
from apple_vit.data.transforms import build_val_transform
from apple_vit.models.vit_classifier import ViTClassifier
from apple_vit.training.metrics import (
    compute_metrics,
    get_confusion_matrix,
    print_classification_report,
)
from apple_vit.utils.config import Config, set_seed
from apple_vit.utils.logger import get_logger
from apple_vit.visualization.plot_utils import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ViT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default="configs/vit_base.yaml")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Test"],
                        help="Which dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_cm", type=str, default=None,
                        help="Path to save confusion matrix figure (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.train.seed)

    logger = get_logger("evaluate")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load model from checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ViTClassifier(ckpt["config"].model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info(f"Restored from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    # Dataset
    data_root = Path(cfg.data.data_root) / args.split
    dataset = AppleDiseaseDataset(data_root, transform=build_val_transform(cfg.data))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"Evaluating on {args.split} split: {len(dataset)} images")

    # Inference
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    # Report
    metrics = compute_metrics(all_labels, all_preds)
    logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"Macro-F1  : {metrics['macro_f1']:.4f}")
    print("\n--- Full Classification Report ---")
    print_classification_report(all_labels, all_preds)

    cm = get_confusion_matrix(all_labels, all_preds)
    save_path = args.save_cm or cfg.figure_dir / "confusion_matrix_eval.png"
    plot_confusion_matrix(cm, save_path=save_path)
    logger.info(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    main()
