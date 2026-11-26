import torch

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}

def register_optimizer(name):
    def wrapper(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return wrapper

def get_optimizer(name):
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{name}' not registered")
    return OPTIMIZER_REGISTRY[name]