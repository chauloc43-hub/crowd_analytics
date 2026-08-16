"""Body visual-presentation classifier kept separate from the face classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.gender_classifier import GENDER_LABELS


BODY_MODEL_ROLE = "body_visual_presentation"
BODY_MODEL_ARCHITECTURE = "mobilenet_v3_small"


class BodyGenderClassifier(nn.Module):
    """MobileNetV3-Small used by the PA-100K body checkpoint."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, len(GENDER_LABELS))
        # Preserve torchvision's native key layout (features.*, avgpool.*, classifier.*)
        # because the trained checkpoint was saved directly from that layout.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.avgpool(features)
        return self.classifier(torch.flatten(pooled, 1))
