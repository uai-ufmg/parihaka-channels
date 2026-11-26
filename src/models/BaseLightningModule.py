import torch
import lightning as L
from typing import Any, Dict, Optional, Tuple, Union
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt

from src.utils.torch_utils import logits_to_preds


class BaseLightningModule(L.LightningModule):
    """Base Lightning Module for training segmentation models.

    This class provides a standardized interface for training, validation, and testing
    with support for:
    - Custom model building (_build_model)
    - Loss functions and metrics
    - Optimizer and learning rate scheduler configuration
    - Metric logging and tracking
    - Image prediction logging to TensorBoard
    - Multi-stage training (training, validation, testing)

    Subclasses must implement:
        - _build_model(): Build and return the model architecture

    Subclasses can optionally implement:
        - _post_init_setup(): Setup after model initialization (e.g., load weights)
        - configure_lr_scheduler_config(): Customize scheduler configuration
        - setup_image_logging(dataloader): Initialize image logging with custom dataloader

    Args:
        model_config: Configuration dictionary for the model.
        loss_fn: Loss function to use.
        optimizer_builder: Callable that takes model parameters and returns an optimizer.
        metrics: Metrics object to compute and track performance.

    Example config with scheduler and image logging:
        config = {
            'num_classes': 1,
            'log_images': True,
            'ignore_index': 2,
            'cmap': 'viridis',
            'logging': {
                'num_positive': 8,
                'num_negative': 8,
                'threshold': 0.1,
                'ignore_index': 2
            },
            'scheduler': {
                'type': 'cosine',
                'T_max': 100,
                'interval': 'epoch',
                'monitor': 'val_loss'
            }
        }

    After creating the trainer and before training, call:
        module.setup_image_logging(val_dataloader)
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_fn,
        optimizer_builder,
        metrics,
        scheduler_builder=None,
        scheduler_config=None,
    ) -> None:
        super().__init__()
        self.config = model_config
        self.save_hyperparameters()
        self.dist = False  # default to False; set to True if using distributed training

        self.model = self._build_model()
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer_builder = optimizer_builder
        self.scheduler_builder = scheduler_builder
        self.scheduler_config = scheduler_config or {}

        # Image logging flag (callback will handle actual logging)
        self.log_images_enabled = model_config.get("logging", {}).get("log_images", False)
        self.log_steps = model_config.get("logging", {}).get("log_steps", False)

        self._post_init_setup()


    # -------------------
    # To be implemented by subclasses
    # -------------------
    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _post_init_setup(self):
        """Hook opcional pra coisas como carregar pesos"""
        pass

    # -------------------
    # Forward
    # -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -------------------
    # Shared step
    # -------------------
    def _shared_step(self, batch):
        x, y = batch["data"], batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        is_multiclass = self.config["num_classes"] > 1
        preds = logits_to_preds(logits, is_multiclass=is_multiclass)

        return loss, preds, y

    def log_dict(
        self,
        dictionary,
        prog_bar=False,
        on_step=None,
        on_epoch=None,
        sync_dist=False,
        batch_size=None,
        prefix="",
    ):
        """
        The Lightining method wasn't propagating the `sync_dist` parameter,
        So I'm reimplementing it
        """
        for k, v in dictionary.items():
            self.log(
                name=prefix + k,
                value=v,
                prog_bar=prog_bar,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=sync_dist,
                batch_size=batch_size,
            )
        return None

    # Image logging is handled externally by ImageLoggingCallback;
    # BaseLightningModule exposes `log_images_enabled` flag only.

    # -------------------
    # Logging helper
    # -------------------
    def _log_metrics(self, stage: str, loss, preds, targets):
        targets = targets.squeeze(1) if targets.dim() == 4 and targets.size(1) == 1 else targets

        batch_metrics = self.metrics(preds, targets)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=self.log_steps,
            on_epoch=True,
            prog_bar=self.log_steps,
            sync_dist=self.dist,
        )

        self.log_dict(
            batch_metrics,
            on_step=self.log_steps,
            on_epoch=True,
            prog_bar=self.log_steps,
            sync_dist=self.dist,
            prefix=f"{stage}_",
        )

    # -------------------
    # Steps
    # -------------------
    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self._log_metrics("train", loss, preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self._log_metrics("val", loss, preds, y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self._log_metrics("test", loss, preds, y)
        return loss

    # -------------------
    # Epoch hooks
    # -------------------
    def on_train_epoch_start(self):
        """Hook called at the start of each training epoch."""
        pass

    def on_validation_epoch_start(self):
        """Hook called at the start of each validation epoch."""
        pass

    def on_test_epoch_start(self):
        """Hook called at the start of each test epoch."""
        pass

    def on_train_epoch_end(self):
        self.metrics.reset()

    def on_validation_epoch_end(self):
        """Called at end of validation epoch. Logs image predictions."""
        self.metrics.reset()
        # Image logging is handled by ImageLoggingCallback

    def on_test_epoch_end(self):
        """Called at end of test epoch. Logs image predictions."""
        self.metrics.reset()
        # Image logging is handled by ImageLoggingCallback

    # -------------------
    # Optimizer and Scheduler
    # -------------------
    def configure_optimizers(self) -> Union[Dict, Tuple]:
        """Configure optimizer and optional learning rate scheduler.

        Returns:
            Union[Dict, Tuple]: Optimizer configuration for PyTorch Lightning.
                If only optimizer: returns the optimizer directly.
                If with scheduler: returns dict with 'optimizer' and 'lr_scheduler' keys.
        """
        optimizer = self.optimizer_builder(self.parameters())

        scheduler = None
        if self.scheduler_builder is not None:
            scheduler = self.scheduler_builder(optimizer)

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_config.get("interval", "epoch"),
                "frequency": self.scheduler_config.get("frequency", 1),
                "monitor": self.scheduler_config.get("monitor", "val_loss"),
            },
        }
