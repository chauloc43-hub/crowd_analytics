from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


GENDER_LABELS = ("female", "male")
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


class GenderClassifier(nn.Module):
    """MobileNetV3-Large classifier with a fixed, documented label order."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        if num_classes != len(GENDER_LABELS):
            raise ValueError(f"This classifier expects {len(GENDER_LABELS)} classes")
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
