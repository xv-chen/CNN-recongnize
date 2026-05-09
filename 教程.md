# CNN + Transformer 实现手写数字识别

## 1. 项目是做什么的？

用 **CNN + Transformer 混合模型** 识别 MNIST 手写数字（0~9）。

一张 28×28 的手写数字图片输入模型，输出它属于 0~9 中哪个数字。

```
输入:  [1, 28, 28]  灰度手写数字图片
输出:  [10]          10个数字的概率分布 → 取最大值对应数字
```

---

## 2. 项目结构

```
数字识别-CNN+Transformer/
├── config.py     超参数配置
├── data.py       数据加载
├── model.py      模型定义（核心）
├── trainer.py    训练器
├── main.py       入口
└── 教程.md        本文件
```

每个文件只负责一件事，互不干扰。

---

## 3. 逐个文件解读

### 3.1 config.py — 超参数

```python
@dataclass
class Config:
    batch_size: int = 64       # 每批64张图
    d_model: int = 128         # Transformer 的特征维度
    nhead: int = 4             # 多头注意力头数
    num_encoder_layers: int = 3  # Transformer 层数
    epochs: int = 10           # 训练10轮
    lr: float = 1e-3           # 学习率
```

所有参数集中在这里，想调参数只改这一个文件。

---

### 3.2 data.py — 数据加载

```python
transform = transforms.Compose([
    transforms.ToTensor(),                          # 图片 → Tensor
    transforms.Normalize((0.1307,), (0.3081,)),     # 标准化
])
```

**MNIST** 是 28×28 的灰度图，每张图就是一个数字的手写体。

- `train_set`: 60000 张训练图片
- `test_set`: 10000 张测试图片
- `DataLoader`: 按 batch_size 打包，训练时打乱顺序

`transforms.Normalize((0.1307,), (0.3081,))` 中的 0.1307 和 0.3081 是 MNIST 数据集的**全局均值和标准差**，标准化后模型训练更稳定。

---

### 3.3 model.py — 模型（核心）

```
输入图片 (1×28×28)
    │
    ▼
┌─────────────────────────────────────────┐
│           CNN 特征提取器                  │
│   Conv2d → BN → ReLU → MaxPool  ×3       │
│   1×28×28  →  32×14×14  →  64×7×7  →    │
│   128×7×7                                 │
│   ▲ 把图片逐步压缩，提取局部特征            │
└─────────────────────────┬───────────────┘
    │
    ▼  将 128×7×7 展平为 49 个 128 维向量
    │
┌─────────────────────────────────────────┐
│       Transformer 编码器                 │
│   位置编码 + 3层自注意力                   │
│   49个token 互相"交流"，建模全局依赖关系     │
│   ▲ 不像CNN只看局部，Transformer看全局     │
└─────────────────────────┬───────────────┘
    │
    ▼  全局平均池化（49个token取平均）
    │
  分类头 Linear(128, 10)  →  输出10个数字的得分
```

#### CNN 部分在做什么？

CNN 用 3 层卷积逐步压缩图片：

| 层 | 输入形状 | 输出形状 | 作用 |
|---|---|---|---|
| Conv1 + Pool | 1×28×28 | 32×14×14 | 提取边缘、线条等低级特征 |
| Conv2 + Pool | 32×14×14 | 64×7×7 | 提取形状、笔画等中级特征 |
| Conv3 | 64×7×7 | 128×7×7 | 提取高级语义特征 |

**卷积**：用一个小窗口扫过图片，提取局部模式（类似于看图片的一小块区域）

**池化**：缩小图片尺寸，保留重要信息（类似于给图片缩略图）

最终得到 `128×7×7` 的特征图，也就是 49 个位置，每个位置有 128 维特征。

#### Transformer 在做什么？

把 CNN 输出的 49 个位置看作 49 个"词语"（token），用 Transformer 让它们互相"交流"：

```
位置1的特征 ↔ 位置2的特征 ↔ 位置3的特征 ↔ ... ↔ 位置49的特征
      ↓                                            ↓
   知道"这里是数字的上半部分"              知道"这里是数字的下半部分"
```

每个位置通过**自注意力机制**，能"看到"其他所有位置的信息，从而建立全局理解。比如识别数字 8，需要知道上下两个圈是连在一起的。

**位置编码**：告诉 Transformer 每个 token 在 7×7 网格中的位置，否则它分不清"左上"和"右下"。

#### 分类头

最后把所有 49 个 token 的特征做**全局平均池化**（取平均），得到一个 128 维的向量表示整张图片，然后用 `Linear(128, 10)` 映射到 10 个数字的得分。

---

### 3.4 trainer.py — 训练器

```python
# 每个 epoch 做两件事：
train_epoch()    # 在训练集上学习，更新参数
evaluate()       # 在测试集上打分，不更新参数
```

- `train_epoch`: 前向传播 → 算损失 → 反向传播 → 更新权重
- `evaluate`: 前向传播 → 预测 → 算准确率
- 如果测试准确率创新高，保存模型为 `best_mnist.pth`

#### 关键概念

**损失函数 (CrossEntropyLoss)**：模型预测和真实标签之间的差距，越小越好

**优化器 (AdamW)**：根据损失调整模型参数，让预测越来越准

**epoch**：完整看一遍所有训练数据

---

### 3.5 main.py — 入口

```python
cfg = Config()                          # 读配置
train_loader, test_loader = build_dataloader(cfg)  # 加载数据
model = CNNTransformer(cfg)             # 创建模型
trainer = Trainer(model, cfg, device)   # 创建训练器
trainer.fit(train_loader, test_loader)  # 开始训练
```

4 行代码串联整个流程：配置 → 数据 → 模型 → 训练。

---

## 4. 数据流完整路径

```
main.py
  │
  ├── Config()  →  所有超参数
  │
  ├── build_dataloader()
  │     │
  │     ├── datasets.MNIST(root="数据集/", download=True)
  │     │     └── 从网络下载 MNIST 到 数据集/ 目录
  │     │
  │     └── DataLoader(数据集, batch_size=64, shuffle=True)
  │           └── 每次返回 64 张图片 (64, 1, 28, 28)
  │
  ├── CNNTransformer(cfg)
  │     │
  │     ├── CNN:   (64, 1, 28, 28) → (64, 128, 7, 7)
  │     ├── 重塑:  (64, 128, 7, 7) → (64, 49, 128)
  │     ├── Transformer: (64, 49, 128) → (64, 49, 128)
  │     ├── 池化:  (64, 49, 128) → (64, 128)
  │     └── 分类头: (64, 128) → (64, 10)
  │
  └── Trainer.fit()
        ├── train_epoch: 在 60000 张图上训练
        └── evaluate: 在 10000 张图上测试
              └── 准确率 ≈ 99%
```

---

## 5. 运行

```bash
D:/Miniconda3/envs/pytorch-gpu/python.exe main.py
```

正常输出：
```
Device: cuda
Config: Config(batch_size=64, ..., epochs=10, lr=0.001, ...)

Epoch  1 | loss 0.1812 | acc 96.11% | best 96.11% | 14.6s
Epoch  2 | loss 0.0631 | acc 98.52% | best 98.52% | 14.7s
...
Epoch 10 | loss 0.0222 | acc 98.85% | best 99.15% | 14.4s

Done! Best acc: 99.15%  (model saved: best_mnist.pth)
```

每轮约 14 秒（GPU），10 轮后最高 99.15% 准确率。

---

## 6. 动手实验

试试改 `config.py` 里的参数观察效果：

| 改什么 | 预期效果 |
|--------|---------|
| `epochs=20` | 训练更久，准确率可能再提高一点 |
| `nhead=8` | 更多注意力头，可能更准但更慢 |
| `lr=1e-4` | 学习率变小，收敛变慢但更稳 |
| `dropout=0.3` | 更强正则化，减少过拟合 |

也可以用训练好的模型做推理：

```python
import torch
from model import CNNTransformer
from config import Config

model = CNNTransformer(Config())
model.load_state_dict(torch.load("best_mnist.pth"))
model.eval()

# 用你自己的图片测试
# ... 预处理成 (1, 28, 28) 的 tensor ...
# pred = model(img.unsqueeze(0)).argmax(1).item()
# print(f"预测结果: {pred}")
```
