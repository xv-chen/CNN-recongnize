"""数据加载"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import Config


def build_dataloader(cfg: Config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root=cfg.data_root, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=cfg.data_root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, test_loader
