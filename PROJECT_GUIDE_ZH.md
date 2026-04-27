# 项目学习与报告整理指南（ECE549_Group_Project）

这份文档面向“需要快速跑通项目、理解代码结构、并整理课程报告”的使用场景。内容基于仓库现状与以下关键文件的实现细节：

- `README.md`（项目结构与 Quick Start）
- `configs/vit_base.yaml`（主实验超参）
- `apple_vit/data/dataset.py`（数据目录约束、采样策略）
- `scripts/train.py`（训练入口、产物输出）

---

## 1. 这项目到底做什么？

目标是 **4 类苹果病害分类**，并对比两条路线：

- **传统 CV baseline**：HSV + 形态学等规则方法（见 `cv_baseline/`）
- **深度学习 ViT**：基于 HuggingFace ViT 进行 fine-tune（见 `apple_vit/`）

你写报告时最核心的“故事线”建议是：

- 数据量小 + 类别不平衡 → baseline 的局限、ViT 的训练风险
- 通过 **WeightedRandomSampler / class weights / 强增强 / cosine+warmup / early stopping** 控制过拟合
- 用 **macro-F1、混淆矩阵、attention rollout 可视化** 解释模型行为与错误模式

---

## 2. 先跑通一条完整链路（最重要）

你学习项目的最高性价比方式是：先得到可复现的输出物，再回头读代码解释“为什么这么设计”。

### 2.1 数据集目录必须长这样

`apple_vit/data/dataset.py` 写死了类名映射（`CLASS_TO_IDX`）并按目录扫描图片。要求：

```
data/apple_disease_classification/
  Train/
    Normal_Apple/
    Blotch_Apple/
    Rot_Apple/
    Scab_Apple/
  Test/
    Normal_Apple/
    Blotch_Apple/
    Rot_Apple/
    Scab_Apple/
```

支持图片后缀：`.jpg/.jpeg/.png/.bmp/.webp`。如果目录里找不到任何图片，会抛：
`FileNotFoundError: No images found under ...`

### 2.2 训练（ViT 主线）

主实验配置在 `configs/vit_base.yaml`，运行：

```bash
python scripts/train.py --config configs/vit_base.yaml
```

你也可以用 CLI 覆盖少量超参（`scripts/train.py` 支持）：

```bash
python scripts/train.py --config configs/vit_base.yaml \
  --epochs 50 --lr 1e-5 --batch_size 16 --experiment_name my_run
```

### 2.3 训练的输出物在哪里？

`configs/vit_base.yaml` 的输出设置：

- `output.output_dir: "outputs"`
- `output.experiment_name: "vit_base_run"`

通常会生成类似：

```
outputs/
  checkpoints/<experiment_name>/...
  logs/<experiment_name>/...
  figures/<experiment_name>/
    training_curves.png
    confusion_matrix.png
```

此外 `scripts/train.py` 在训练结束后会做一次“最终评估”并保存两张图：

- `training_curves.png`
- `confusion_matrix.png`

这两张图非常适合直接塞进报告。

---

## 3. 项目结构怎么读（推荐顺序）

不要从模型/训练细节开始硬啃，按“数据→配置→训练入口→模型→训练器→评估/可视化”的顺序更快建立整体心智模型。

- **配置层（实验可复现的核心）**
  - `configs/vit_base.yaml`：数据路径、增强、batch size、lr、scheduler、warmup、early stopping、workers、seed、输出目录等
  - `apple_vit/utils/config.py`：把 YAML 读进 dataclass（建议你读一遍，能立刻知道所有可控项）

- **数据层（最常见报错来源）**
  - `apple_vit/data/dataset.py`：目录结构、类名映射、`WeightedRandomSampler`、class weights
  - `apple_vit/data/transforms.py`：训练/验证增强（与 `vit_base.yaml` 对应）

- **训练入口（把所有模块串起来的“骨架”）**
  - `scripts/train.py`：组 dataloader → 算 class weights → 建模型 → Trainer.fit → 最终评估 → 出图

- **模型**
  - `apple_vit/models/vit_classifier.py`：backbone（默认 `google/vit-base-patch16-224-in21k`）、dropout、冻结策略等

- **训练与指标**
  - `apple_vit/training/trainer.py`：优化器、scheduler、early stopping、保存 checkpoint 等
  - `apple_vit/training/metrics.py`：Accuracy / macro-F1 / confusion matrix / classification report

- **解释性可视化**
  - `apple_vit/visualization/attention_maps.py`：attention rollout
  - `apple_vit/visualization/plot_utils.py`：训练曲线、混淆矩阵出图

- **传统 CV baseline（对比实验）**
  - `cv_baseline/segmentation.py`：HSV+形态学 pipeline
  - `cv_baseline/evaluate_iou.py`：IoU 评估（需要你准备 mask 数据）

---

## 4. 关键设计点：你报告里应该怎么解释

这部分几乎可以直接搬进报告的“Method / Design Decisions / Discussion”。

- **类别不平衡**
  - `Normal_Apple` 训练样本最少（README 的表里能看到）
  - `dataset.py` 中提供了 `class_weights()`（逆频率）和 `WeightedRandomSampler`
  - 训练入口 `scripts/train.py` 固定启用 `use_weighted_sampler=True`

- **强增强**
  - `vit_base.yaml` 里默认 `color_jitter: true`、`random_erasing: true`、`random_rotation: 20`
  - 目的：小数据集上减少过拟合

- **学习率策略**
  - `vit_base.yaml`：`lr_scheduler: cosine` + `warmup_epochs: 3`
  - 目的：Transformer fine-tune 常见稳定策略

- **早停**
  - `early_stopping_patience: 8`
  - 目的：防止小数据集训练后期过拟合

- **Attention Rollout**
  - 用注意力图解释模型关注区域，适合展示“模型学到了什么/失败在哪里”

---

## 5. 报告素材怎么“系统化”产出（实验记录模板）

建议你每一次实验都记录下面这些字段（复制到你的实验日志里即可）：

- **实验名**：`experiment_name`
- **配置文件**：`configs/xxx.yaml`（以及任何 CLI override）
- **环境**：OS、Python、PyTorch 版本、CUDA 版本、GPU 型号、显存
- **数据**：训练/测试每类样本数（可从数据集目录统计）
- **训练超参**：epochs、batch_size、lr、weight_decay、scheduler、warmup、seed、增强开关、sampler 开关
- **最佳 checkpoint 路径**：`outputs/checkpoints/.../checkpoint_best.pt`
- **最终指标**：Accuracy、macro-F1、每类 precision/recall/F1
- **图表**：`training_curves.png`、`confusion_matrix.png`、若干 attention maps
- **观察与结论（2-5 行）**：错误模式、可能原因、下一步改进

你可以把实验组织成如下“最小消融套件”（建议 3~5 个就够写报告）：

- **E0（默认）**：`vit_base.yaml` 原样
- **E1（不采样）**：关闭 weighted sampler（如果代码支持开关；否则临时改 `scripts/train.py`）
- **E2（弱增强）**：关掉 `color_jitter` 或 `random_erasing`
- **E3（冻结 encoder 线性探针）**：`freeze_encoder: true`（与 `vit_small_probe.yaml` 对照）
- **E4（学习率/epoch 变体）**：例如 lr×0.5 或 epochs+10

---

## 6. “我需要更多数据”——可行策略

你可以从三个层次扩展数据（报告里分别对应：数据扩增、外部泛化、困难样本分析）：

- **训练时有效扩增**：把增强当作“有效样本扩增”，用 E2 这种消融证明它确实提升了 macro-F1
- **外部数据集**：从 Kaggle 再找 1 个相似 apple disease 数据，作为外部测试集（最加分：泛化能力）
- **困难样本库**：收集模型失败的样本（光照极端、背景复杂、病斑小/遮挡/模糊），在报告中做案例分析

注意：如果你引入新数据集，确保类名能映射到本项目的四类（否则需要新增类或过滤）。

---

## 7. 常见报错与快速定位

- **`ModuleNotFoundError: torchvision`**
  - 基本都是“没在正确的虚拟环境里运行”或“torch/torchvision 没装齐”
  - 用 `which python`、`python -c "import torch, torchvision"` 确认

- **`FileNotFoundError: No images found under .../Train`**
  - 数据目录结构或类文件夹名不对（必须是 `Normal_Apple/Blotch_Apple/Rot_Apple/Scab_Apple`）

- **GPU 可用但不能跑（sm_120 相关）**
  - RTX 50 系（`sm_120`）需要 **CUDA 12.8（cu128）** 及对应 PyTorch wheel
  - 用 `python -c "import torch; print(torch.cuda.get_arch_list())"` 检查是否包含 `sm_120`

---

## 8. 建议你立刻做的三件事（下一步行动）

- 跑一次 `vit_base.yaml` 的端到端训练，确认 `outputs/figures/...` 出现两张图
- 读一遍 `configs/vit_base.yaml` + `scripts/train.py`，把“可控超参列表”整理进报告方法部分
- 做 2~3 个最小消融（第 5 节的 E1/E2/E3），用统一模板记录结果并对比混淆矩阵与 macro-F1

