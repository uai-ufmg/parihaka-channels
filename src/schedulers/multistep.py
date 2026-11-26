import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("multistep")
class MultiStepLRScheduler:
    def __init__(self, optimizer, milestones=None, gamma=0.1, **kwargs):
        self.scheduler = lr_schedulers.MultiStepLR(
            optimizer,
            milestones=milestones or [],
            gamma=gamma,
        )

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
