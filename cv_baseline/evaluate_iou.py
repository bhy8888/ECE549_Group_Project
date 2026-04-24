"""
IoU evaluation for the traditional CV segmentation baseline.

Original author: Hongbin Yang
Refactored into package form.

Usage:
    python -m cv_baseline.evaluate_iou --mask_dir mark_pic
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from cv_baseline.segmentation import segment_defects


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def evaluate_directory(mask_dir: str | Path) -> Tuple[float, int]:
    """
    Evaluate all (image, mask) pairs in a directory.

    Expects files named:  <name>.jpg / .jpeg / .png
                          <name>_mask.png

    Returns:
        (mean_iou, num_images_evaluated)
    """
    mask_dir = Path(mask_dir)
    gt_paths = sorted(mask_dir.glob("*_mask.png"))

    total_iou, count = 0.0, 0
    results: List[Tuple[str, float]] = []

    for gt_path in gt_paths:
        base = mask_dir / gt_path.stem.replace("_mask", "")
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = base.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if img is None or gt_mask is None:
            continue

        _, clean_mask = segment_defects(img)
        iou = calculate_iou(clean_mask, gt_mask)
        total_iou += iou
        count += 1
        results.append((img_path.name, iou))

    return (total_iou / count if count > 0 else 0.0), count, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CV baseline IoU")
    parser.add_argument("--mask_dir", type=str, default="mark_pic",
                        help="Directory containing images and _mask.png files")
    args = parser.parse_args()

    print("Starting traditional CV baseline evaluation…\n")
    mean_iou, count, results = evaluate_directory(args.mask_dir)

    for name, iou in results[:10]:
        print(f"  {name:40s}  IoU = {iou:.4f}")
    if len(results) > 10:
        print(f"  … ({len(results) - 10} more)")

    print(f"\nEvaluated {count} image(s).")
    print(f"Mean IoU (Traditional CV Baseline): {mean_iou:.4f}")


if __name__ == "__main__":
    main()
