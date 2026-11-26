import segmentation_models_pytorch as smp

from src.registry.model import register_model
from src.models.BaseLightningModule import BaseLightningModule


@register_model("DeepLabV3Plus")
class DeepLabV3Plus(BaseLightningModule):
    def _build_model(self):
        return smp.DeepLabV3Plus(
            in_channels=self.config.get("num_channels", 3),
            classes=self.config.get("num_classes", 1),
            encoder_name=self.config.get("encoder_name", "resnet50"),
            encoder_weights=self.config.get("encoder_weights", "imagenet"),
            activation=self.config.get("activation", None)
        )