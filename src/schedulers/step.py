import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("step")
class StepLRScheduler:
    def __init__(self, optimizer, step_size=10, gamma=0.1, **kwargs):
        self.scheduler = lr_schedulers.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
