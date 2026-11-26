import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("lambda")
class LambdaLRScheduler:
    def __init__(self, optimizer, lr_lambda=None, **kwargs):
        if lr_lambda is None:
            lr_lambda = lambda epoch: 1.0
        self.scheduler = lr_schedulers.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
