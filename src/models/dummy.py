import torch.nn as nn
from src.registry.model import register_model

@register_model("DummyModel")
class DummyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, input_ids, **kwargs):
        return self.linear(input_ids)

    def __repr__(self):
        return f"DummyModel(input_dim={self.linear.in_features})"