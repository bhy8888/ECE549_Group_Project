#!/usr/bin/env python3
"""
Generate attention rollout visualizations for a set of test images.

Usage:
    # Visualize a single image
    python scripts/visualize_attention.py \
        --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
        --config configs/vit_base.yaml \
        --image data/apple_disease_classification/Test/Scab_Apple/some.jpg

    # Visualize a random subset from all test classes
    python scripts/visualize_attention.py \
        --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
        --config configs/vit_base.yaml \
        --n_samples 8 \
        --save_dir outputs/figures/attention_maps
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from apple_vit.models.vit_classifier import ViTClassifier
from apple_vit.utils.config import Config, set_seed
from apple_vit.utils.logger import get_logger
from apple_vit.visualization.attention_maps import AttentionVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ViT Attention Rollout")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/vit_base.yaml")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image (overrides --n_samples)")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of random test images to visualize")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save output figures")
    parser.add_argument("--discard_ratio", type=float, default=0.9,
                        help="Fraction of low-attention weights to zero out")
    parser.add_argument("--head_fusion", type=str, default="mean",
                        choices=["mean", "max", "min"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    set_seed(args.seed)
    logger = get_logger("visualize_attention")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ViTClassifier(ckpt["config"].model)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Model loaded (epoch {ckpt['epoch']})")

    visualizer = AttentionVisualizer(model, cfg.data, device=str(device))

    save_dir = Path(args.save_dir) if args.save_dir else cfg.figure_dir / "attention_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Single image mode ---
    if args.image is not None:
        logger.info(f"Visualizing single image: {args.image}")
        fig = visualizer.visualize_single(
            args.image,
            save_path=save_dir / f"attn_{Path(args.image).stem}.png",
            discard_ratio=args.discard_ratio,
            head_fusion=args.head_fusion,
        )
        import matplotlib.pyplot as plt
        plt.show()
        return

    # --- Batch mode: pick random images from each class ---
    test_root = Path(cfg.data.data_root) / "Test"
    all_images: list[Path] = []
    for cls_dir in sorted(test_root.iterdir()):
        if cls_dir.is_dir():
            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
            # 2 samples per class if enough
            k = min(max(1, args.n_samples // 4), len(imgs))
            all_images.extend(random.sample(imgs, k))

    logger.info(f"Visualizing {len(all_images)} images → grid figure")
    fig = visualizer.visualize_grid(
        all_images,
        save_path=save_dir / "attention_grid.png",
        discard_ratio=args.discard_ratio,
    )

    # Also save individual per-image plots
    for img_path in all_images:
        visualizer.visualize_single(
            img_path,
            save_path=save_dir / f"attn_{img_path.parent.name}_{img_path.stem}.png",
            discard_ratio=args.discard_ratio,
            head_fusion=args.head_fusion,
        )

    logger.info(f"All attention maps saved to {save_dir}")


if __name__ == "__main__":
    main()
