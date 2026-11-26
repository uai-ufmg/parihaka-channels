import torch.optim as optim
from src.registry.optimizer import register_optimizer

@register_optimizer("SGD")
class SGD(optim.SGD):
    pass

@register_optimizer("Adam")
class Adam(optim.Adam):
    pass

@register_optimizer("AdamW")
class AdamW(optim.AdamW):
    pass