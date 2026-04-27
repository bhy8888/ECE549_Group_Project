#!/usr/bin/env python3
"""
Export qualitative CV baseline comparisons (HSV vs adaptive vs watershed) against GT masks.

Creates montage images for a small subset of annotated samples under `mark_pic/`.

Usage:
  python scripts/export_cv_comparison.py --mask_dir mark_pic --out_dir outputs/figures/cv_comparison --n_samples 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_baseline.evaluate_iou import calculate_iou
from cv_baseline.segmentation import segment_defects


def _read_image_pair(mask_dir: Path, gt_path: Path) -> Tuple[Path, np.ndarray, np.ndarray]:
    base = mask_dir / gt_path.stem.replace("_mask", "")
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        cand = base.with_suffix(ext)
        if cand.exists():
            img_path = cand
            break
    if img_path is None:
        raise FileNotFoundError(f"No image for mask: {gt_path.name}")

    img = cv2.imread(str(img_path))
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        raise FileNotFoundError(f"Failed reading: {img_path} or {gt_path}")
    return img_path, img, gt


def _to_3c(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    overlay = img_bgr.copy()
    overlay[m == 1] = (1 - alpha) * overlay[m == 1] + alpha * np.array(color, dtype=np.float32)
    return overlay.astype(np.uint8)


def _tile(rows: List[List[np.ndarray]], pad: int = 6) -> np.ndarray:
    # Ensure same height per row
    row_imgs = []
    for r in rows:
        h = max(im.shape[0] for im in r)
        resized = []
        for im in r:
            if im.shape[0] != h:
                w = int(im.shape[1] * (h / im.shape[0]))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            resized.append(im)
        row_imgs.append(resized)

    # Pad and concat
    def hpad(im: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    padded_rows = [np.concatenate([hpad(im) for im in r], axis=1) for r in row_imgs]
    return np.concatenate(padded_rows, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export qualitative CV comparisons")
    p.add_argument("--mask_dir", type=str, default="mark_pic")
    p.add_argument("--out_dir", type=str, default="outputs/figures/cv_comparison")
    p.add_argument("--n_samples", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_paths = sorted(mask_dir.glob("*_mask.png"))
    if not gt_paths:
        raise FileNotFoundError(f"No *_mask.png found in {mask_dir}")

    # Pick a deterministic subset: first N masks (quick) + some hard cases (lowest IoU under hsv)
    scored: List[Tuple[float, Path]] = []
    for gt_path in gt_paths:
        try:
            _, img, gt = _read_image_pair(mask_dir, gt_path)
        except FileNotFoundError:
            continue
        _, hsv_mask = segment_defects(img, method="hsv")
        iou = calculate_iou(hsv_mask, gt)
        scored.append((iou, gt_path))

    scored.sort(key=lambda x: x[0])  # hardest first
    picks = [p for _, p in scored[: max(0, args.n_samples)]]

    for idx, gt_path in enumerate(picks, start=1):
        img_path, img, gt = _read_image_pair(mask_dir, gt_path)

        _, m_hsv = segment_defects(img, method="hsv")
        _, m_adp = segment_defects(img, method="adaptive")
        _, m_wsd = segment_defects(img, method="watershed")

        i_hsv = calculate_iou(m_hsv, gt)
        i_adp = calculate_iou(m_adp, gt)
        i_wsd = calculate_iou(m_wsd, gt)

        # Build panels
        img_s = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
        gt_s = cv2.resize(gt, (448, 448), interpolation=cv2.INTER_NEAREST)
        m_hsv_s = cv2.resize(m_hsv, (448, 448), interpolation=cv2.INTER_NEAREST)
        m_adp_s = cv2.resize(m_adp, (448, 448), interpolation=cv2.INTER_NEAREST)
        m_wsd_s = cv2.resize(m_wsd, (448, 448), interpolation=cv2.INTER_NEAREST)

        ov_gt = _overlay_mask(img_s, gt_s, color=(0, 255, 0))
        ov_hsv = _overlay_mask(img_s, m_hsv_s, color=(0, 0, 255))
        ov_adp = _overlay_mask(img_s, m_adp_s, color=(0, 0, 255))
        ov_wsd = _overlay_mask(img_s, m_wsd_s, color=(0, 0, 255))

        def label(im: np.ndarray, text: str) -> np.ndarray:
            out = im.copy()
            cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
            return out

        row1 = [
            label(img_s, f"Image: {img_path.name}"),
            label(ov_gt, "GT overlay (green)"),
            label(_to_3c(gt_s), "GT mask"),
        ]
        row2 = [
            label(ov_hsv, f"HSV overlay | IoU={i_hsv:.3f}"),
            label(ov_adp, f"Adaptive overlay | IoU={i_adp:.3f}"),
            label(ov_wsd, f"Watershed overlay | IoU={i_wsd:.3f}"),
        ]

        montage = _tile([row1, row2], pad=8)
        save_path = out_dir / f"cv_compare_{idx:02d}_{img_path.stem}.png"
        cv2.imwrite(str(save_path), montage)

    print(f"Saved {len(picks)} montages to {out_dir}")


if __name__ == "__main__":
    main()

