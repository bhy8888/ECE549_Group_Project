"""
ViT-based 4-class apple disease classifier.

Uses HuggingFace `transformers` to load a pre-trained ViT encoder and
replaces the classification head with a custom MLP.

Supported backbones (any HF ViT checkpoint):
  - google/vit-base-patch16-224-in21k   (default, ~86M params)
  - google/vit-large-patch16-224-in21k
  - WinKawaks/vit-small-patch16-224
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

from apple_vit.utils.config import ModelConfig


class ViTClassifier(nn.Module):
    """
    Fine-tunable ViT with a classification head.

    Architecture:
        ViT Encoder  →  [CLS] token  →  LayerNorm  →  Dropout  →  Linear(4)

    Attributes:
        vit:     Pre-trained ViT encoder (from HuggingFace).
        head:    Classification MLP.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Load pre-trained encoder
        self.vit = ViTModel.from_pretrained(
            cfg.backbone,
            add_pooling_layer=False,   # we extract CLS manually
        )

        hidden_size: int = self.vit.config.hidden_size

        # Freeze encoder if requested (linear probing)
        if cfg.freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(hidden_size, cfg.num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            logits:       (B, num_classes)
        """
        outputs = self.vit(pixel_values=pixel_values, output_attentions=False)
        # outputs.last_hidden_state: (B, 1 + num_patches, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0]   # (B, hidden_size)
        return self.head(cls_token)

    # ------------------------------------------------------------------
    def forward_with_attentions(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Same as forward but also returns all attention weight tensors.

        Returns:
            logits:     (B, num_classes)
            attentions: tuple of (B, num_heads, seq_len, seq_len) per layer
        """
        outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.head(cls_token)
        return logits, outputs.attentions

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "config": self.cfg},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ViTClassifier":
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
