import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("cosine")
class CosineAnnealingLRScheduler:
    def __init__(self, optimizer, T_max=10, eta_min=0.0, **kwargs):
        self.scheduler = lr_schedulers.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
