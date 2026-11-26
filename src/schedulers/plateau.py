import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("plateau")
class ReduceLROnPlateauScheduler:
    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, verbose=True, **kwargs):
        self.scheduler = lr_schedulers.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
        )

    def step(self, metrics=None):
        return self.scheduler.step(metrics)

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
