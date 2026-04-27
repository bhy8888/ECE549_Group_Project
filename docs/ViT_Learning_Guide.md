# ViT 学习指南：从宏观框架到实战细节

> 面向本项目（ECE549 Apple Disease Classification）的 Vision Transformer 系统性学习路径。  
> 作者整理：Yanjun Qin

---

## 目录

1. [推荐学习顺序](#一推荐的学习顺序从宏观到微观)
2. [宏观框架：ViT 的核心思想](#二宏观框架vit-的核心思想)
3. [完整数据流与 Tensor Shape](#三完整数据流最重要)
4. [Self-Attention 的本质](#四self-attention-的本质用直觉理解)
5. [损失函数与优化器](#五损失函数和优化)
6. [HuggingFace 库调用位置](#六huggingface-库在代码里的位置)
7. [如何快速读懂代码框架](#七如何快速读懂一个代码框架)
8. [如何修改模型](#八如何修改模型实战方法论)
9. [行动计划](#九总结你现在应该做什么)

---

## 一、推荐的学习顺序：从宏观到微观

```
第一层（宏观）：理解"为什么用 ViT，它解决了什么问题"
      ↓
第二层（数据流）：搞清楚一张图片如何变成最终分类概率
      ↓
第三层（架构）：Self-Attention 机制的数学原理
      ↓
第四层（工程）：HuggingFace 库如何实现，代码在哪里调用
      ↓
第五层（训练）：损失函数、优化器、学习率调度、如何防过拟合
```

> **最常见的错误学习路径**：直接看 Self-Attention 公式，背了半天，但不知道为什么要这样设计。  
> **正确做法**：先建立直觉，再理解公式。

---

## 二、宏观框架：ViT 的核心思想

**CNN 的局限**：卷积核只看局部邻域，感受野随层数增加才能扩大，全局关系建模弱。

**ViT 的核心想法**：把图片切成 patch，当成"词"，用 Transformer 的 Self-Attention 直接建模任意两个 patch 之间的关系。

```
CNN：局部 → 局部 → ... → 全局（归纳偏置强，小数据也能训）
ViT：一上来就全局（靠数据量/预训练弥补归纳偏置不足）
```

这解释了为什么 **ViT 在小数据集上容易过拟合**（本项目训练集仅 382 张，这正是对比实验有趣的地方）。  
核心对策：**预训练权重（ImageNet-21k）+ 数据增强 + 正则化**。

---

## 三、完整数据流（最重要）

> 专家习惯：读任何模型代码，第一件事就是在纸上画出每步的 shape 变化，这是最快理解模型的方式。

```python
输入图片:   (B, 3, 224, 224)
              │
              ▼
# ── Step 1: Patch Embedding ──────────────────────────────────────────────────
# 将图片切成 16×16 的 patch，每个 patch 展平后线性投影到 768-d
→ Patch Tokens:  (B, 196, 768)     # 196 = (224/16)²

# ── Step 2: 拼接 [CLS] token ──────────────────────────────────────────────────
# 在 sequence 最前面加一个可学习的 [CLS] 向量
→ Tokens:        (B, 197, 768)     # 197 = 196 patches + 1 CLS

# ── Step 3: 加 Positional Encoding ────────────────────────────────────────────
# Shape 不变，但每个位置加了位置信息（可学习的 197×768 矩阵）
→               (B, 197, 768)

# ── Step 4: 经过 12 层 Transformer Encoder ────────────────────────────────────
# 每层内部：LayerNorm → Multi-Head Self-Attention → 残差
#           LayerNorm → MLP (768→3072→768, GELU) → 残差
→               (B, 197, 768)     # shape 全程不变

# ── Step 5: 取 [CLS] token（index=0）────────────────────────────────────────
# [CLS] token 在 Attention 中聚合了全图信息，代表整图语义
→ CLS vector:   (B, 768)

# ── Step 6: 自定义分类头 ──────────────────────────────────────────────────────
# LayerNorm → Dropout(p=0.1) → Linear(768→4)
→ Logits:       (B, 4)            # 4 个类别的未归一化分数

# ── Step 7: 损失计算 ──────────────────────────────────────────────────────────
# CrossEntropyLoss 内部自动做 Softmax，不需要手动加
→ loss:         scalar
```

### 关键数字（ViT-Base-Patch16）

| 参数 | 值 |
|------|----|
| Patch size | 16 × 16 px |
| Patch 数量 N | (224/16)² = **196** |
| Sequence 长度 | **197**（含 [CLS]）|
| Hidden dim D | **768** |
| Attention heads | **12** |
| Head dim | 768 / 12 = **64** |
| MLP dim | **3072**（4 × 768）|
| Encoder 层数 | **12** |
| 总参数量 | ~**86M** |

---

## 四、Self-Attention 的本质（用直觉理解）

**核心问题**：patch i 应该关注哪些其他 patch？

```python
# 对每个 token，分别生成三个向量
Q = X @ W_q   # Query：我想找什么信息？
K = X @ W_k   # Key：  我能提供什么信息？
V = X @ W_v   # Value：我实际携带的内容

# Attention Score：Q 和 K 的点积相似度（除以根号 d_k 防数值爆炸）
score = Q @ K.T / sqrt(d_k)   # → (B, 197, 197)：197×197 的注意力矩阵

# 归一化：每行 softmax → 每个 token 对所有其他 token 的注意力权重
attn  = softmax(score)         # → (B, 197, 197)

# 加权聚合 Value
output = attn @ V              # → (B, 197, 768)
```

**为什么要 Multi-Head（多头）？**  
把 768-d 拆成 12 个 64-d 的子空间，每个头独立做 Attention，最后 concat。  
不同的头可以关注不同的语义关系——有的头关注颜色，有的关注纹理，有的关注形状。  
这也是 **Attention Map 可视化**的来源（Maximum Goal 中的核心内容）。

---

## 五、损失函数和优化

### 损失函数：`nn.CrossEntropyLoss`

```python
# PyTorch 的 CrossEntropyLoss = LogSoftmax + NLLLoss（已内置 Softmax，不要重复加）
# 输入 logits: (B, 4)，label: (B,)，值为 0-3 的整数
loss = F.cross_entropy(logits, labels)

# 类别不平衡时（本项目：Normal 67 张 vs Blotch 116 张），加类别权重
loss = F.cross_entropy(logits, labels, weight=class_weights)
# class_weights 用逆频率：total / count_per_class
```

**为什么本项目需要 class weight**：  
Normal 67 张 vs Blotch 116 张，差了近一倍。不加权则模型偏向多数类，Accuracy 虚高，Macro-F1 才是真实指标。

### 优化器：AdamW

```python
# AdamW = Adam + Weight Decay 解耦（比普通 Adam 正则效果更好）
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
```

### 学习率调度：Warmup + Cosine Decay

```python
# ViT 微调的标准做法：
# 前 3 个 epoch 线性 warmup（从 0.1×lr 升到 lr）
# 之后 cosine 衰减到接近 0

# 为什么要 warmup？
# Transformer 开始时参数随机，学习率太大会
# 破坏预训练权重，warmup 让优化器"温柔地"进入微调节奏
```

---

## 六、HuggingFace 库在代码里的位置

本项目关键文件：`apple_vit/models/vit_classifier.py`

```python
from transformers import ViTModel

class ViTClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # ── 核心调用：加载预训练 ViT Encoder ──────────────────────────────
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",  # HuggingFace Hub 上的模型 ID
            add_pooling_layer=False,               # 不用自带 pooling，手动取 [CLS]
        )
        # from_pretrained 做了什么：
        # 1. 下载 ~343MB 预训练权重（在 ImageNet-21k 上训练）
        # 2. 载入权重到模型
        # 3. 你的模型立刻拥有了 1400 万张图片上学到的特征提取能力

        # ── 自定义分类头（只有这部分是从头训练的）────────────────────────
        hidden_size = self.vit.config.hidden_size   # = 768
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(hidden_size, cfg.num_classes),   # 768 → 4
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]   # 取 index=0 的 [CLS] token
        return self.head(cls_token)
```

**其他关键位置**：

| 功能 | 文件 | 关键行 |
|------|------|--------|
| 数据加载 & 增强 | `apple_vit/data/dataset.py` | `AppleDiseaseDataset.__getitem__` |
| 训练循环 | `apple_vit/training/trainer.py` | `Trainer._train_one_epoch` |
| Attention Map | `apple_vit/visualization/attention_maps.py` | `attention_rollout()` |
| 超参配置 | `configs/vit_base.yaml` | 所有超参均在此集中管理 |
| 训练入口 | `scripts/train.py` | `main()` |

---

## 七、如何快速读懂一个代码框架

### 黄金三步法

**Step 1：找入口，追数据流主干**

```
找 train.py / main.py
↓
找这条主干线：
dataset → dataloader → model(batch) → loss → optimizer.step()

这条线搞清楚，全局就明白了。其余代码都是辅助。
```

**Step 2：只看 `forward()` 方法**

```
不管代码多复杂，Tensor 从哪进、在哪变、从哪出，
全部在 forward() 里。
忽略 __init__、logging、checkpoint 等所有辅助代码。
```

**Step 3：打印 shape，动手跑一个 batch**

```python
# 在关键位置加一行，立刻看清 Tensor 的真实形状
print(f">>> {x.shape}")

# 只需用一个 batch 跑通 forward，不需要全量训练
model.eval()
with torch.no_grad():
    dummy = torch.randn(2, 3, 224, 224)   # batch_size=2 的假数据
    out = model(dummy)
    print(out.shape)   # 应该是 (2, 4)
```

### 专家真正关注的重点

| 关注点 | 为什么重要 |
|--------|-----------|
| **数据增强策略** | 小数据集场景下，80% 的性能来自数据处理 |
| **学习率和 warmup** | ViT 微调最容易在这里踩坑 |
| **是否有数据泄露** | train/val 分割是否正确，影响结果可信度 |
| **类别不平衡** | 直接影响 metric 是否真实 |
| **Metric 选择** | Accuracy 在不平衡数据下会误导，优先看 Macro-F1 |
| **过拟合曲线** | train_acc 和 val_acc 的差距是模型健康度的直接体现 |

---

## 八、如何修改模型（实战方法论）

> **核心原则**：不要一上来就改架构。先建立 baseline，再针对性修改。

### 诊断流程

```
Step 1：跑通 → 看 val 曲线是否正常（loss 下降，不 NaN/爆炸）

Step 2：诊断问题
   train_acc >> val_acc  →  过拟合  →  加 Dropout、增强、正则化
   train_acc ≈ val_acc 都低  →  欠拟合  →  加模型容量、降 lr、更多 epoch

Step 3：针对性改进（本项目具体路径）
   ├── 数据太少   → 加强数据增强（RandomErasing、MixUp、CutMix）
   ├── 类别不平衡 → 调整 class_weights 或 WeightedRandomSampler
   ├── 过拟合严重 → 增大 Dropout、降 lr、加 Early Stopping
   └── 想提精度   → 换 ViT-Large，或加 TTA（测试时增强取平均）
```

### Attention Map 可视化的修改点（Maximum Goal）

```python
# apple_vit/visualization/attention_maps.py

# forward_with_attentions() 返回：
# attentions: tuple of (B, 12, 197, 197) × 12 层
#             12 层 × 12 头 × 197×197 注意力矩阵

# Attention Rollout 核心思路：
# 把 12 层的 attention 矩阵递归相乘（考虑残差连接）
# result[0, 1:] = [CLS] token 对每个 patch 的最终注意力权重
# → reshape 成 14×14（= 196 patches = 14×14）
# → 双线性上采样到 224×224
# → 叠加到原图 → 热力图
```

---

## 九、总结：行动计划

### 今天
- [ ] 把第三节的 shape 变化表（从 `(B,3,224,224)` 到 `(B,4)`）抄在纸上
- [ ] 理解为什么 sequence 长度是 **197** 而不是 196

### 本周
- [ ] 读 `apple_vit/models/vit_classifier.py` 的 `forward()`
- [ ] 在本地写一个测试脚本，用 dummy input 跑通 forward，打印每步 shape

### 跑实验时
- [ ] 先跑 `configs/vit_small_probe.yaml`（线性探测，冻结 Encoder，快速验证数据流）
- [ ] 再跑 `configs/vit_base.yaml`（全量微调，解冻所有层）
- [ ] 对比两者的 val Macro-F1，理解预训练权重的价值

### 最终目标
- [ ] 用 `scripts/visualize_attention.py` 生成 Attention Map 热力图
- [ ] 找几张 Scab/Blotch 病害图，展示模型"看到的"病变区域
- [ ] 对比传统 CV（HSV 分割）和 ViT（Attention Map）在同一张图上的差异

---

> **关键结论**：ViT 在本项目小数据集上成功的关键不是架构本身，  
> 而是 **预训练权重 + 数据增强 + 正确的正则化** 三者的结合。  
> 这也是最值得在报告中重点论述的 insight。

---

*参考资料*
- [An Image is Worth 16x16 Words (ViT 原论文)](https://arxiv.org/abs/2010.11929)
- [HuggingFace ViT 文档](https://huggingface.co/docs/transformers/model_doc/vit)
- [Attention Rollout 论文](https://arxiv.org/abs/2005.00928)
- [How to train your ViT (Google)](https://arxiv.org/abs/2106.10270)
