import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("exponential")
class ExponentialLRScheduler:
    def __init__(self, optimizer, gamma=0.1, **kwargs):
        self.scheduler = lr_schedulers.ExponentialLR(optimizer, gamma=gamma)

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
