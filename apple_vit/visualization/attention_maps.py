"""
Attention map extraction and visualization for ViT.

Implements the "Attention Rollout" algorithm (Abnar & Zuidema, 2020) which
recursively multiplies attention weights layer-by-layer to approximate how
information flows from patches to the [CLS] token.

References:
  - Abnar & Zuidema (2020): https://arxiv.org/abs/2005.00928
  - Chefer et al. (2021): Transformer Interpretability Beyond Attention Visualization
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from apple_vit.data.dataset import IDX_TO_CLASS, FRIENDLY_NAMES
from apple_vit.data.transforms import build_inverse_transform, build_val_transform
from apple_vit.models.vit_classifier import ViTClassifier
from apple_vit.utils.config import DataConfig


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def attention_rollout(
    attentions: Tuple[torch.Tensor, ...],
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """
    Compute Attention Rollout for a single image.

    Args:
        attentions:    Tuple of (1, num_heads, seq_len, seq_len) per layer.
        discard_ratio: Fraction of lowest attention weights to zero out per layer.
        head_fusion:   How to merge multi-head attention: "mean" | "max" | "min".

    Returns:
        mask: (num_patches,) – normalized attention score per patch from [CLS].
    """
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)

    with torch.no_grad():
        for attention in attentions:
            # attention: (1, H, N, N)
            if head_fusion == "mean":
                fused = attention.mean(dim=1).squeeze(0)   # (N, N)
            elif head_fusion == "max":
                fused = attention.amax(dim=1).squeeze(0)
            elif head_fusion == "min":
                fused = attention.amin(dim=1).squeeze(0)
            else:
                raise ValueError(f"Unknown head_fusion mode: {head_fusion}")

            # Discard low-attention weights
            flat = fused.flatten()
            threshold = flat.kthvalue(int(discard_ratio * flat.numel())).values
            fused = torch.where(fused >= threshold, fused, torch.zeros_like(fused))

            # Add residual connection (identity)
            fused = fused + torch.eye(fused.size(0), device=fused.device)
            # Row-normalize
            fused = fused / fused.sum(dim=-1, keepdim=True)

            result = fused @ result

    # CLS token is index 0; extract its attention to all patch tokens
    mask = result[0, 1:]   # (num_patches,)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def rollout_to_heatmap(
    mask: torch.Tensor,
    grid_size: int,
    image_size: int,
) -> np.ndarray:
    """
    Reshape 1D patch mask → 2D → upsample to image_size × image_size.

    Returns:
        heatmap: (image_size, image_size) float32 in [0, 1].
    """
    mask_2d = mask.reshape(grid_size, grid_size).cpu().numpy()
    heatmap = np.array(
        Image.fromarray(mask_2d).resize((image_size, image_size), Image.BICUBIC)
    )
    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level visualizer
# ---------------------------------------------------------------------------

class AttentionVisualizer:
    """
    Wraps a trained ViTClassifier and provides:
      1. Single-image attention rollout overlay
      2. Grid visualization for multiple images
    """

    def __init__(
        self,
        model: ViTClassifier,
        cfg: DataConfig,
        device: Optional[str] = None,
    ):
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.cfg = cfg
        self.val_transform = build_val_transform(cfg)
        self.inv_transform = build_inverse_transform()

        patch_size = model.vit.config.patch_size   # 16 for vit-base-patch16
        self.grid_size = cfg.image_size // patch_size

    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualize_single(
        self,
        image_path: str | Path,
        save_path: Optional[str | Path] = None,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean",
        alpha: float = 0.55,
    ) -> plt.Figure:
        """
        Produces a 3-panel figure: original | heatmap | overlay.

        Args:
            image_path:    Path to the input image.
            save_path:     If given, saves the figure there.
            discard_ratio: See attention_rollout().
            head_fusion:   See attention_rollout().
            alpha:         Blend ratio of heatmap over original image.

        Returns:
            matplotlib Figure.
        """
        # Load & preprocess
        pil_img = Image.open(image_path).convert("RGB").resize(
            (self.cfg.image_size, self.cfg.image_size)
        )
        img_tensor = self.val_transform(pil_img).unsqueeze(0).to(self.device)

        # Forward pass with attentions
        logits, attentions = self.model.forward_with_attentions(img_tensor)
        pred_idx = logits.argmax(dim=1).item()
        pred_cls = FRIENDLY_NAMES.get(IDX_TO_CLASS[pred_idx], IDX_TO_CLASS[pred_idx])
        confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()

        # Attention rollout
        mask = attention_rollout(attentions, discard_ratio=discard_ratio, head_fusion=head_fusion)
        heatmap = rollout_to_heatmap(mask, self.grid_size, self.cfg.image_size)

        # Build RGBA overlay
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        heatmap_colored = plt.cm.jet(heatmap)[..., :3]   # (H, W, 3)
        overlay = (1 - alpha) * img_np + alpha * heatmap_colored

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title("Original", fontsize=12)
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Attention Rollout", fontsize=12)
        axes[2].imshow(np.clip(overlay, 0, 1))
        axes[2].set_title(f"Prediction: {pred_cls} ({confidence:.1%})", fontsize=12)
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"Source: {Path(image_path).name}", fontsize=10, y=1.01)
        plt.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig

    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualize_grid(
        self,
        image_paths: List[str | Path],
        save_path: Optional[str | Path] = None,
        discard_ratio: float = 0.9,
        ncols: int = 4,
    ) -> plt.Figure:
        """
        Renders a grid of original-vs-overlay pairs for a list of images.
        """
        n = len(image_paths)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 4 * nrows * 2))
        axes = np.array(axes).reshape(nrows * 2, ncols)

        for i, img_path in enumerate(image_paths):
            row, col = (i // ncols) * 2, i % ncols
            pil_img = Image.open(img_path).convert("RGB").resize(
                (self.cfg.image_size, self.cfg.image_size)
            )
            img_tensor = self.val_transform(pil_img).unsqueeze(0).to(self.device)
            logits, attentions = self.model.forward_with_attentions(img_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_cls = FRIENDLY_NAMES.get(IDX_TO_CLASS[pred_idx], IDX_TO_CLASS[pred_idx])
            confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()

            mask = attention_rollout(attentions, discard_ratio=discard_ratio)
            heatmap = rollout_to_heatmap(mask, self.grid_size, self.cfg.image_size)
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
            overlay = np.clip(0.45 * img_np + 0.55 * plt.cm.jet(heatmap)[..., :3], 0, 1)

            axes[row, col].imshow(pil_img)
            axes[row, col].axis("off")
            axes[row + 1, col].imshow(overlay)
            axes[row + 1, col].set_title(f"{pred_cls} {confidence:.0%}", fontsize=9)
            axes[row + 1, col].axis("off")

        # Hide unused cells
        for j in range(n, nrows * ncols):
            r, c = (j // ncols) * 2, j % ncols
            axes[r, c].axis("off")
            axes[r + 1, c].axis("off")

        plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)

        return fig
