"""
Image transforms for training and inference.

Training:  aggressive augmentation to combat overfitting on ~380 images.
Inference: deterministic center-crop only.
"""
from __future__ import annotations

from torchvision import transforms

from apple_vit.utils.config import DataConfig

# ImageNet statistics used by ViT pre-training
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_train_transform(cfg: DataConfig) -> transforms.Compose:
    ops = []

    ops.append(transforms.Resize((cfg.image_size + 32, cfg.image_size + 32)))
    ops.append(transforms.RandomCrop(cfg.image_size))

    if cfg.random_horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.random_vertical_flip:
        ops.append(transforms.RandomVerticalFlip(p=0.5))
    if cfg.random_rotation > 0:
        ops.append(transforms.RandomRotation(degrees=cfg.random_rotation))
    if cfg.color_jitter:
        ops.append(
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        )
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=_MEAN, std=_STD))
    if cfg.random_erasing:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))

    return transforms.Compose(ops)


def build_val_transform(cfg: DataConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def build_inverse_transform() -> transforms.Compose:
    """Undo normalization — used when visualizing an input tensor as an image."""
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=[-m / s for m, s in zip(_MEAN, _STD)],
                std=[1.0 / s for s in _STD],
            )
        ]
    )
