import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("cyclic")
class CyclicLRScheduler:
    def __init__(self, optimizer, base_lr=0.001, max_lr=0.1, step_size_up=2000, **kwargs):
        self.scheduler = lr_schedulers.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
        )

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
