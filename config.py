"""超参数配置"""
from dataclasses import dataclass


@dataclass
class Config:
    # 数据
    batch_size: int = 64
    num_workers: int = 4
    data_root: str = "../../数据集"  # 相对于本项目目录

    # 模型
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_classes: int = 10

    # 训练
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    save_path: str = "best_mnist.pth"
    log_interval: int = 100
