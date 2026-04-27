"""
AppleDiseaseDataset — wraps the Kaggle apple-disease folder structure.

Expected layout:
    <data_root>/
        Train/
            Normal_Apple/   *.jpg
            Blotch_Apple/   *.jpg
            Rot_Apple/      *.jpg
            Scab_Apple/     *.jpg
        Test/
            ...same classes...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from apple_vit.data.transforms import build_train_transform, build_val_transform
from apple_vit.utils.config import DataConfig

# Maps folder name → integer label
CLASS_TO_IDX: Dict[str, int] = {
    "Normal_Apple": 0,
    "Blotch_Apple": 1,
    "Rot_Apple": 2,
    "Scab_Apple": 3,
}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
FRIENDLY_NAMES = {
    "Normal_Apple": "Normal",
    "Blotch_Apple": "Blotch",
    "Rot_Apple": "Rot",
    "Scab_Apple": "Scab",
}


class AppleDiseaseDataset(Dataset):
    """
    Args:
        root:      path to either the Train/ or Test/ directory.
        transform: torchvision transform to apply.
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for cls_name, idx in CLASS_TO_IDX.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((img_path, idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found under {self.root}. Check the path.")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ------------------------------------------------------------------
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {c: 0 for c in CLASS_TO_IDX}
        for _, lbl in self.samples:
            counts[IDX_TO_CLASS[lbl]] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights — one weight per class (used by loss or sampler)."""
        counts = self.class_counts()
        total = len(self.samples)
        weights = torch.tensor(
            [total / counts[IDX_TO_CLASS[i]] for i in range(len(CLASS_TO_IDX))],
            dtype=torch.float32,
        )
        return weights / weights.sum() * len(CLASS_TO_IDX)   # normalize

    def sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        cw = self.class_weights()
        return [cw[lbl].item() for _, lbl in self.samples]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: DataConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).

    Weighted sampler over-samples minority classes during training.
    """
    data_root = Path(cfg.data_root)
    train_ds = AppleDiseaseDataset(data_root / "Train", transform=build_train_transform(cfg))
    val_ds = AppleDiseaseDataset(data_root / "Test", transform=build_val_transform(cfg))

    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
