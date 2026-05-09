"""CNN + Transformer 混合模型"""
import torch
import torch.nn as nn
from config import Config


class CNNTransformer(nn.Module):
    """CNN 提取局部特征 → Transformer 建模全局依赖 → 分类"""

    def __init__(self, cfg: Config):
        super().__init__()
        # CNN backbone: 1×28×28 → 128×7×7
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, cfg.d_model, 3, padding=1), nn.BatchNorm2d(cfg.d_model), nn.ReLU(),
        )
        # Transformer encoder: 49 tokens × d_model
        self.pos_embed = nn.Parameter(torch.randn(1, 49, cfg.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.num_encoder_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, x):
        x = self.cnn(x)                            # (B, d, 7, 7)
        B = x.shape[0]
        x = x.view(B, x.size(1), -1).transpose(1, 2)  # (B, 49, d)
        x = x + self.pos_embed
        x = self.transformer(x)                    # (B, 49, d)
        x = x.mean(dim=1)                          # (B, d)
        x = self.norm(x)
        return self.head(x)
