"""Pretrained backbone loaders."""

from __future__ import annotations

from typing import Any


BACKBONES = {
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}


def load_resnet_backbone(backbone_name: str = "resnet101", pretrained: bool = True) -> Any:
    """Load a torchvision ResNet without its pooling and classification head."""
    name = backbone_name.lower()
    if name not in BACKBONES:
        raise ValueError(f"Unsupported ResNet backbone: {backbone_name}. Choose from {sorted(BACKBONES)}")

    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise ImportError("ResNet backbones require torch and torchvision") from exc

    builder = getattr(models, name)
    weights_enum = getattr(models, f"ResNet{name.removeprefix('resnet')}_Weights", None)
    try:
        model = builder(weights=weights_enum.DEFAULT if pretrained and weights_enum else None)
    except TypeError:  # torchvision < 0.13
        model = builder(pretrained=pretrained)

    backbone = nn.Sequential(*list(model.children())[:-2])
    backbone.backbone_name = name
    backbone.feature_dim = BACKBONES[name]
    backbone.num_features = BACKBONES[name]
    return backbone


def load_pretrained_backbone(backbone_name: str, pretrained: bool = True) -> Any:
    """Load one of the backbones implemented in this checkout."""
    if backbone_name.lower() in BACKBONES:
        return load_resnet_backbone(backbone_name, pretrained)
    raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from {sorted(BACKBONES)}")


def load_tresnet_backbone(*args, **kwargs):
    raise NotImplementedError("TResNet-L support has not been implemented")


def load_cvt_backbone(*args, **kwargs):
    raise NotImplementedError("CvT-W24 support has not been implemented")
