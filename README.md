# ECE549 / CS543 — Automated Fruit Defect Segmentation & Grading

**UIUC 2026 | Yanjun Qin (yanjunq2) · Hongbin Yang (hongbin3)**

> Comparative study of Traditional CV vs. Vision Transformer (ViT) for 4-class apple disease classification.

---

## Project Structure

```
ECE549_Group_Project/
├── apple_vit/                  # Deep-Learning pipeline (ViT)
│   ├── data/
│   │   ├── dataset.py          # AppleDiseaseDataset + DataLoader factory
│   │   └── transforms.py       # Train / val augmentation pipelines
│   ├── models/
│   │   └── vit_classifier.py   # HuggingFace ViT with custom head
│   ├── training/
│   │   ├── trainer.py          # Training loop, LR scheduler, early stopping
│   │   └── metrics.py          # Accuracy, macro-F1, confusion matrix
│   ├── visualization/
│   │   ├── attention_maps.py   # Attention Rollout algorithm + AttentionVisualizer
│   │   └── plot_utils.py       # Training curves, confusion matrix plots
│   └── utils/
│       ├── config.py           # Typed dataclass config + YAML loader
│       └── logger.py           # Console + file logger, TensorBoard writer
│
├── cv_baseline/                # Traditional CV pipeline (HSV + morphology)
│   ├── segmentation.py         # segment_defects(), defect_ratio(), grade_apple()
│   └── evaluate_iou.py         # IoU evaluation against manually-drawn masks
│
├── scripts/
│   ├── train.py                # python scripts/train.py --config configs/vit_base.yaml
│   ├── evaluate.py             # python scripts/evaluate.py --checkpoint <path>
│   └── visualize_attention.py  # python scripts/visualize_attention.py --checkpoint <path>
│
├── configs/
│   ├── vit_base.yaml           # Main experiment: ViT-Base fine-tuning
│   └── vit_small_probe.yaml    # Ablation: ViT-Small linear probe
│
├── outputs/                    # Auto-created; gitignored
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
│
├── data/                       # Dataset; gitignored
│   └── apple_disease_classification/
│       ├── Train/{Normal,Blotch,Rot,Scab}_Apple/
│       └── Test/ ...
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Dataset

| Class | Train | Test |
|---|---|---|
| Normal_Apple | 67 | 24 |
| Blotch_Apple | 116 | 30 |
| Rot_Apple | 114 | 38 |
| Scab_Apple | 85 | 28 |

Source: [Kaggle — Apple Fruit Disease Dataset](https://www.kaggle.com/)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Train the ViT classifier

```bash
python scripts/train.py --config configs/vit_base.yaml
```

Optional CLI overrides:

```bash
python scripts/train.py --config configs/vit_base.yaml \
    --epochs 50 --lr 1e-5 --experiment_name my_run
```

### 3. Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
    --config configs/vit_base.yaml
```

### 4. Visualize Attention Maps

```bash
# Single image
python scripts/visualize_attention.py \
    --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
    --image data/apple_disease_classification/Test/Scab_Apple/some.jpg

# Random grid from all classes
python scripts/visualize_attention.py \
    --checkpoint outputs/checkpoints/vit_base_run/checkpoint_best.pt \
    --n_samples 8 \
    --save_dir outputs/figures/attention_maps
```

### 5. Run Traditional CV baseline evaluation

```bash
python -m cv_baseline.evaluate_iou --mask_dir mark_pic
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **WeightedRandomSampler** | Normal_Apple only has 67 training images — over-sampling prevents class bias |
| **Aggressive augmentation** | ~380 training images is small for ViT; color jitter + random erase mitigate overfitting |
| **Cosine LR + warmup** | Standard best practice for Transformer fine-tuning |
| **Attention Rollout** | Abnar & Zuidema (2020) — more faithful than raw last-layer attention |
| **YAML configs** | Reproducible experiments; every run is fully described by its YAML file |

---

## Running on Google Colab / Campus Cluster

```python
# In a Colab cell:
!git clone <repo_url>
%cd ECE549_Group_Project
!pip install -e . -q
!python scripts/train.py --config configs/vit_base.yaml --num_workers 2
```

GPU is auto-detected via `torch.cuda.is_available()` — no code changes needed.
