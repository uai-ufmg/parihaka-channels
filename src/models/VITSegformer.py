import torch
import timm
import torch.nn as nn
from src.registry.model import register_model
from src.models.BaseLightningModule import BaseLightningModule
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from segmentation_models_pytorch.base import SegmentationHead


class VITSegformer(nn.Module):
    def __init__(
        self,
        vit_config="vit_large_patch16_224",
        img_size=224,
        backbone_weights=None,
        freeze_backbone=False,
        out_indices=[-4, -3, -2, -1],
        input_dim=3,
        vit_dim=1024,
        decoder_dim=1024,
        segmentation_kernel=1,
        num_classes=1,
    ):
        super(VITSegformer, self).__init__()
        self.backbone = timm.create_model(
            vit_config,
            pretrained=False,
            img_size=img_size,
            dynamic_img_size=True,
            features_only=True,
            out_indices=out_indices,
        )
        self.decoder = SegformerDecoder(
            encoder_channels=[input_dim, *([vit_dim] * 4)],
            encoder_depth=len(out_indices),
            segmentation_channels=decoder_dim,
        )
        self.head = self.segmentation_head = SegmentationHead(
            in_channels=decoder_dim,
            out_channels=num_classes,
            kernel_size=segmentation_kernel,
            upsampling=4,
        )

        if backbone_weights is not None:
            #state_dict = torch.load(backbone_weights)
            state_dict = torch.load(backbone_weights, map_location="cpu")
            self.backbone.model.load_state_dict(state_dict, strict=False)

        if freeze_backbone == True:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        elif isinstance(freeze_backbone, float):
            backbone_len = len(self.backbone.model.blocks)
            for i, block in enumerate(self.backbone.model.blocks):
                if (i / backbone_len) < freeze_backbone:
                    for p in block.parameters():
                        p.requires_grad = False

    def forward(self, x):
        # segformer expects input as first feature
        features = [x]

        x = self.backbone(x)
        features.extend(x)

        seg_features = self.decoder(features)

        pred = self.head(seg_features)

        return pred

@register_model("SegFormer")
class VITSeismicObserverFormer(BaseLightningModule):
    def _build_model(self):
        return VITSegformer(
            vit_config=self.config.get("vit_config", "vit_large_patch16_224"),
            img_size=self.config.get("img_size", 224),
            backbone_weights=self.config.get("backbone_weights", None),
            freeze_backbone=self.config.get("freeze_backbone", False),
            out_indices=self.config.get("out_indices", [-4, -3, -2, -1]),
            input_dim=self.config.get("input_dim", 3),
            vit_dim=self.config.get("vit_dim", 1024),
            decoder_dim=self.config.get("decoder_dim", 1024),
            segmentation_kernel=self.config.get("segmentation_kernel", 1),
            num_classes=self.config.get("num_classes", 1),
        )

        
        
        
        