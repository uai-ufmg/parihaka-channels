from typing import Optional
import torch
try:
    from lightning.pytorch.callbacks import Callback
except Exception:
    class Callback:  # fallback base so module can be imported without Lightning installed
        pass

from src.utils.model_utils import select_images_for_logging, plot_predictions


class ImageLoggingCallback(Callback):
    def __init__(
        self,
        num_images: int = 16,
        num_positive: int = 8,
        num_negative: int = 8,
        threshold: float = 0.1,
        ignore_index: int = 2,
        cmap: str = "viridis",
    ) -> None:
        super().__init__()
        self.num_images = num_images
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.cmap = cmap
        self._cached_images = None

    def _ensure_images(self, trainer, pl_module):
        if self._cached_images is not None:
            return

        # Try to obtain a validation dataloader
        dl = None
        if trainer.datamodule is not None:
            try:
                dl = trainer.datamodule.val_dataloader()
            except Exception:
                dl = None
        if dl is None and getattr(trainer, "val_dataloaders", None):
            dl = trainer.val_dataloaders[0]

        if dl is None:
            return

        self._cached_images = select_images_for_logging(
            dl,
            num_positive=self.num_positive,
            num_negative=self.num_negative,
            threshold=self.threshold,
            ignore_index=self.ignore_index,
        )

    def _maybe_log(self, trainer, pl_module, stage: str):
        if not getattr(pl_module, "log_images_enabled", False):
            print("log_image not enabled")
            return
        if pl_module.logger is None:
            print("no_logger")
            return

        self._ensure_images(trainer, pl_module)
        if self._cached_images is None:
            print("no cached_images")
            return
        
        plot_predictions(
            pl_module,
            self._cached_images,
            num_images=self.num_images,
            title=f"{stage}_predictions",
            ignore_index=self.ignore_index,
            cmap=self.cmap,
        )
            

    def on_validation_epoch_end(self, trainer, pl_module):
        print("Logging images...")
        self._maybe_log(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        print("Logging images...")
        self._maybe_log(trainer, pl_module, stage="test")
