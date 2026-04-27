#!/usr/bin/env python3
"""
Select representative Test images (correct + incorrect) and export attention rollout figures.

Why this script exists:
  - scripts/visualize_attention.py samples randomly; for a report we want typical *success/failure* cases.
  - This script evaluates the checkpoint on the Test split, picks a small set of:
      - correct, high-confidence
      - incorrect, high-confidence (strong failure)
      - incorrect, low-confidence (ambiguous)
    then generates per-image attention rollout plots via AttentionVisualizer.

Usage:
  python scripts/select_attention_cases.py \
    --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
    --config configs/vit_base.yaml \
    --out_dir outputs/figures/vit_base_run/attention_selected \
    --num_correct 6 --num_incorrect 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apple_vit.data.dataset import AppleDiseaseDataset, IDX_TO_CLASS
from apple_vit.data.transforms import build_val_transform
from apple_vit.models.vit_classifier import ViTClassifier
from apple_vit.utils.config import Config, set_seed
from apple_vit.visualization.attention_maps import AttentionVisualizer


@dataclass(frozen=True)
class PredRow:
    image_path: str
    true_idx: int
    pred_idx: int
    confidence: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick representative attention-map cases from the Test split")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/vit_base.yaml")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--discard_ratio", type=float, default=0.9)
    p.add_argument("--head_fusion", type=str, default="mean", choices=["mean", "max", "min"])
    p.add_argument("--num_correct", type=int, default=6)
    p.add_argument("--num_incorrect", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # PyTorch >=2.6 defaults weights_only=True, but our checkpoint stores a Config object.
    # This repo's checkpoints are trusted (locally produced), so we explicitly allow full load.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ViTClassifier(ckpt["config"].model)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    test_root = Path(cfg.data.data_root) / "Test"
    dataset = AppleDiseaseDataset(test_root, transform=build_val_transform(cfg.data))

    # We need file paths; dataset.samples stores (Path, label)
    rows: List[PredRow] = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for i, (img_path, true_idx) in enumerate(dataset.samples):
            x, _ = dataset[i]  # transform applied
            logits = model(x.unsqueeze(0).to(device))
            probs = softmax(logits)[0]
            pred_idx = int(probs.argmax().item())
            conf = float(probs[pred_idx].item())
            rows.append(PredRow(str(img_path), int(true_idx), pred_idx, conf))

    correct = [r for r in rows if r.true_idx == r.pred_idx]
    incorrect = [r for r in rows if r.true_idx != r.pred_idx]

    # Sort for selection: strong cases first
    correct_sorted = sorted(correct, key=lambda r: r.confidence, reverse=True)
    incorrect_hi_conf = sorted(incorrect, key=lambda r: r.confidence, reverse=True)
    incorrect_lo_conf = sorted(incorrect, key=lambda r: r.confidence, reverse=False)

    picked: List[PredRow] = []
    picked.extend(correct_sorted[: max(0, args.num_correct)])
    # Split incorrect into high-confidence failures + ambiguous failures (half/half)
    n_bad = max(0, args.num_incorrect)
    n_hi = n_bad // 2
    n_lo = n_bad - n_hi
    picked.extend(incorrect_hi_conf[:n_hi])
    picked.extend(incorrect_lo_conf[:n_lo])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export a CSV manifest for the report
    manifest_path = out_dir / "selected_cases.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_class", "pred_class", "confidence"])
        for r in picked:
            w.writerow([r.image_path, IDX_TO_CLASS[r.true_idx], IDX_TO_CLASS[r.pred_idx], f"{r.confidence:.6f}"])

    # Generate attention maps
    visualizer = AttentionVisualizer(model, cfg.data, device=str(device))
    for i, r in enumerate(picked, start=1):
        name = Path(r.image_path).stem
        true_c = IDX_TO_CLASS[r.true_idx]
        pred_c = IDX_TO_CLASS[r.pred_idx]
        tag = "correct" if r.true_idx == r.pred_idx else "wrong"
        save_path = out_dir / f"{i:02d}_{tag}_true-{true_c}_pred-{pred_c}_{name}.png"
        visualizer.visualize_single(
            r.image_path,
            save_path=save_path,
            discard_ratio=args.discard_ratio,
            head_fusion=args.head_fusion,
        )

    print(f"Saved {len(picked)} attention figures to: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

