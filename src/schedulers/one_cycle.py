import torch.optim.lr_scheduler as lr_schedulers
from src.registry.scheduler import register_scheduler


@register_scheduler("one_cycle")
class OneCycleLRScheduler:
    def __init__(
        self,
        optimizer,
        max_lr=0.1,
        total_steps=100,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        **kwargs
    ):
        self.scheduler = lr_schedulers.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
        )

    def step(self):
        return self.scheduler.step()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)
