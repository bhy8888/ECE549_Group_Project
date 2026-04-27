"""
Traditional CV segmentation pipeline using HSV thresholding + morphology.

Original author: Hongbin Yang
Refactored into package form for unified project structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# Default HSV thresholds tuned on apple defect images
_DEFAULT_LOWER = np.array([0, 40, 20], dtype=np.uint8)
_DEFAULT_UPPER = np.array([25, 255, 180], dtype=np.uint8)


def _morph_clean(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean


def segment_defects_hsv(
    img_bgr: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed HSV threshold baseline (original)."""
    lower = lower if lower is not None else _DEFAULT_LOWER
    upper = upper if upper is not None else _DEFAULT_UPPER

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, lower, upper)
    clean_mask = _morph_clean(raw_mask, kernel_size=kernel_size)
    return raw_mask, clean_mask


def segment_defects_adaptive(
    img_bgr: np.ndarray,
    kernel_size: int = 5,
    clahe_clip_limit: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    block_size: int = 35,
    c: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive baseline aiming to be more robust to lighting:
      1) estimate an "apple foreground" region in HSV
      2) apply CLAHE on V channel
      3) adaptive threshold (dark = defect) inside the apple region
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Foreground estimate: apples tend to have non-trivial saturation.
    # We keep only the largest connected component to reduce background bleed.
    fg0 = (s > 30).astype(np.uint8) * 255
    fg0 = cv2.morphologyEx(fg0, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    fg0 = cv2.morphologyEx(fg0, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(fg0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg0)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(fg, [largest], -1, 255, thickness=-1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    else:
        fg = fg0

    # Local contrast normalize
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=clahe_grid)
    v_eq = clahe.apply(v)

    # Ensure odd block size for adaptiveThreshold
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    # Adaptive threshold tends to over-fire on specular highlights + background.
    # We combine two cues:
    #   (1) local adaptive threshold on V (dark-ish regions)
    #   (2) global dynamic threshold based on apple-region statistics
    raw_local = cv2.adaptiveThreshold(
        v_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(c),
    )

    v_fg = v_eq[fg > 0]
    if v_fg.size > 0:
        mu = float(v_fg.mean())
        sigma = float(v_fg.std() + 1e-6)
        thr = int(max(0, min(255, mu - 0.60 * sigma)))
    else:
        thr = 80
    raw_global = ((v_eq < thr).astype(np.uint8) * 255)

    raw = cv2.bitwise_and(raw_local, raw_global)

    # Keep only inside apple foreground
    raw_mask = cv2.bitwise_and(raw, raw, mask=fg)
    clean_mask = _morph_clean(raw_mask, kernel_size=kernel_size)
    return raw_mask, clean_mask


def segment_defects_watershed(
    img_bgr: np.ndarray,
    kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Watershed-based segmentation (experimental):
      - compute sure background/foreground on a coarse defect mask
      - run watershed to sharpen boundaries under uneven illumination
    """
    # Start from adaptive mask as a coarse proposal
    raw_mask, _ = segment_defects_adaptive(img_bgr, kernel_size=kernel_size)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr.copy(), markers)
    # watershed marks boundaries as -1; keep interior markers as defect region
    ws = (markers > 1).astype(np.uint8) * 255
    clean_mask = _morph_clean(ws, kernel_size=kernel_size)
    return raw_mask, clean_mask


def segment_defects(
    img_bgr: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    kernel_size: int = 5,
    method: str = "hsv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified entry point for defect segmentation.

    method:
      - "hsv"       : fixed threshold baseline (original)
      - "adaptive"  : CLAHE + adaptive threshold on V channel
      - "watershed" : watershed refinement (experimental)
    """
    method = method.lower().strip()
    if method == "hsv":
        return segment_defects_hsv(img_bgr, lower=lower, upper=upper, kernel_size=kernel_size)
    if method == "adaptive":
        return segment_defects_adaptive(img_bgr, kernel_size=kernel_size)
    if method == "watershed":
        return segment_defects_watershed(img_bgr, kernel_size=kernel_size)
    raise ValueError(f"Unknown method: {method}")


def defect_ratio(mask: np.ndarray) -> float:
    """Fraction of pixels classified as defective (0–1)."""
    return float(mask.astype(bool).sum()) / mask.size


def grade_apple(ratio: float) -> str:
    """
    Simple grading logic based on defect area ratio.

    Returns one of: "Premium", "Grade I", "Reject"
    """
    if ratio < 0.05:
        return "Premium"
    elif ratio < 0.20:
        return "Grade I"
    else:
        return "Reject"
