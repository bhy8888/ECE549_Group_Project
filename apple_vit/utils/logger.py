"""Simple logging utility — console + optional TensorBoard."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class TBWriter:
    """Thin wrapper around SummaryWriter; no-op when disabled."""

    def __init__(self, log_dir: Optional[Path], enabled: bool = True):
        self._writer: Optional[SummaryWriter] = None
        if enabled and log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def add_figure(self, tag: str, figure, step: int) -> None:
        if self._writer is not None:
            self._writer.add_figure(tag, figure, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
