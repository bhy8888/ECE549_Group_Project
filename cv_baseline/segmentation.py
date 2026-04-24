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


def segment_defects(
    img_bgr: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    kernel_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply HSV thresholding + morphological cleaning to detect apple defects.

    Args:
        img_bgr:     BGR image (H, W, 3) as loaded by cv2.imread().
        lower:       Lower HSV bound (default: [0, 40, 20]).
        upper:       Upper HSV bound (default: [25, 255, 180]).
        kernel_size: Morphological kernel size.

    Returns:
        raw_mask:   Binary mask before morphological cleaning.
        clean_mask: Binary mask after opening + closing.
    """
    lower = lower if lower is not None else _DEFAULT_LOWER
    upper = upper if upper is not None else _DEFAULT_UPPER

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    return raw_mask, clean_mask


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
