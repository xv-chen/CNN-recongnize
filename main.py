"""入口：训练 CNN+Transformer 数字识别模型"""
import torch
from config import Config
from data import build_dataloader
from model import CNNTransformer
from trainer import Trainer


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}\n")

    train_loader, test_loader = build_dataloader(cfg)
    model = CNNTransformer(cfg)
    trainer = Trainer(model, cfg, device)
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()
