"""Plotting helpers for training curves and confusion matrix."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from apple_vit.data.dataset import FRIENDLY_NAMES, IDX_TO_CLASS


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot loss and accuracy curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", marker="s", markersize=3, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Accuracy / F1 Curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str | Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a 4×4 confusion matrix with seaborn."""
    class_labels = [FRIENDLY_NAMES.get(IDX_TO_CLASS[i], IDX_TO_CLASS[i]) for i in range(4)]

    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
