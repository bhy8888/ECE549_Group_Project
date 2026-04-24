"""
Training loop for the ViT classifier.

Responsibilities:
  - Epoch-level train / validate loop
  - Cosine LR scheduling with linear warmup
  - Gradient clipping
  - Early stopping
  - Best-checkpoint saving
  - TensorBoard logging
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from apple_vit.training.metrics import compute_metrics
from apple_vit.utils.config import Config
from apple_vit.utils.logger import TBWriter, get_logger


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> None:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class Trainer:
    """
    Orchestrates training and validation.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, cfg)
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.logger = get_logger("Trainer", cfg.log_dir)
        self.tb = TBWriter(cfg.log_dir, enabled=cfg.output.use_tensorboard)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        if class_weights is not None and cfg.train.use_class_weights:
            cw = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

        # LR scheduler: linear warmup → cosine decay
        warmup_steps = cfg.train.warmup_epochs
        total_steps = cfg.train.epochs
        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-7,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

        self.early_stopping = EarlyStopping(patience=cfg.train.early_stopping_patience)
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0

        cfg.make_output_dirs()

    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, List[float]]:
        """Run the full training loop. Returns history dict."""
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []
        }

        for epoch in range(1, self.cfg.train.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc, val_f1 = self._validate(epoch)
            elapsed = time.time() - t0

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch {epoch:03d}/{self.cfg.train.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # TensorBoard
            self.tb.add_scalar("Loss/train", train_loss, epoch)
            self.tb.add_scalar("Loss/val", val_loss, epoch)
            self.tb.add_scalar("Accuracy/train", train_acc, epoch)
            self.tb.add_scalar("Accuracy/val", val_acc, epoch)
            self.tb.add_scalar("F1/val_macro", val_f1, epoch)
            self.tb.add_scalar("LR", lr, epoch)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_f1)

            # Save best
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_acc, is_best=True)

            # Early stopping on macro F1
            self.early_stopping(val_f1)
            if self.early_stopping.should_stop:
                self.logger.info(
                    f"Early stopping at epoch {epoch}. Best val_acc={self.best_val_acc:.4f} "
                    f"at epoch {self.best_epoch}."
                )
                break

        self.tb.close()
        return history

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()

            if self.cfg.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            if step % self.cfg.output.log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        metrics = compute_metrics(all_labels, all_preds)
        return total_loss / total, correct / total, metrics["macro_f1"]

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        tag = "best" if is_best else f"epoch{epoch:03d}"
        path = self.cfg.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        self.logger.info(f"Checkpoint saved: {path}")
