"""Evaluation metrics for multi-class classification."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from apple_vit.data.dataset import IDX_TO_CLASS, FRIENDLY_NAMES


def compute_metrics(
    all_labels: List[int],
    all_preds: List[int],
) -> Dict[str, float]:
    """
    Returns dict with accuracy, macro-F1, and per-class F1.
    """
    labels = np.array(all_labels)
    preds = np.array(all_preds)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, zero_division=0)

    result = {"accuracy": float(acc), "macro_f1": float(macro_f1)}
    for i, f1 in enumerate(per_class):
        cls = FRIENDLY_NAMES.get(IDX_TO_CLASS[i], IDX_TO_CLASS[i])
        result[f"f1_{cls}"] = float(f1)

    return result


def print_classification_report(
    all_labels: List[int],
    all_preds: List[int],
) -> None:
    target_names = [
        FRIENDLY_NAMES.get(IDX_TO_CLASS[i], IDX_TO_CLASS[i]) for i in range(4)
    ]
    print(
        classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    )


def get_confusion_matrix(
    all_labels: List[int],
    all_preds: List[int],
) -> np.ndarray:
    return confusion_matrix(all_labels, all_preds)
