import torch.nn as nn
from src.registry.criterion import register_criterion

@register_criterion("MSELoss")
class MSELoss(nn.MSELoss):
    pass

@register_criterion("CrossEntropyLoss")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass