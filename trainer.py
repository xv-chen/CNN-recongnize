"""训练器"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config


class Trainer:
    def __init__(self, model: nn.Module, cfg: Config, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
        self.best_acc = 0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / total * 100

    def fit(self, train_loader, test_loader, epochs=None):
        epochs = epochs or self.cfg.epochs
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = self.train_epoch(train_loader)
            acc = self.evaluate(test_loader)

            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.model.state_dict(), self.cfg.save_path)

            t = time.time() - t0
            print(f"Epoch {epoch:2d} | loss {loss:.4f} | acc {acc:.2f}% | best {self.best_acc:.2f}% | {t:.1f}s")

        print(f"\nDone! Best acc: {self.best_acc:.2f}%  (model saved: {self.cfg.save_path})")
        return self.best_acc
