import torch
from torch.utils.data import Dataset
from src.registry.dataset import register_dataset

@register_dataset("dummy")
class DummyDataset(Dataset):
    def __init__(self, size=100, input_dim=10):
        self.x = torch.randn(size, input_dim)
        self.y = torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "input_ids": self.x[idx],
            "labels": self.y[idx]
        }