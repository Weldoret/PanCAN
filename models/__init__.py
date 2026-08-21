"""Model creation helpers and implemented backbone loaders."""

from .pretrained import (
    BACKBONES,
    load_cvt_backbone,
    load_pretrained_backbone,
    load_resnet_backbone,
    load_tresnet_backbone,
)
from .random_walk import MultiOrderContextMappingNetwork, RandomWalkAttention


def create_model(
        config,
        num_classes=None,
        device=None,
        class_groups=None,
        group_weights=None):
    from .network import MultiScaleContextAwareNetwork

    num_classes = num_classes or config.network.num_classes
    backbone = load_pretrained_backbone(
        config.network.backbone_name,
        pretrained=config.network.backbone_pretrained,
    )
    return MultiScaleContextAwareNetwork(
        backbone=backbone,
        num_classes=num_classes,
        config=config.network,
        device=device,
        class_groups=class_groups,
        group_weights=group_weights,
    )


def create_custom_network(config, device=None):
    from .neighborhood import generate_adjacency_index_matrix
    from .network import CustomNetwork

    connect_idx, weights_flag = generate_adjacency_index_matrix(
        config.network.grid_rows,
        config.network.grid_cols,
    )
    if device is not None:
        connect_idx = connect_idx.to(device)
        weights_flag = weights_flag.to(device)
    return CustomNetwork(
        num_classes=config.network.num_classes,
        connect_idx=connect_idx,
        weights_flag=weights_flag,
        device=device,
        num_blocks=config.network.num_blocks,
    )


__all__ = [
    "BACKBONES",
    "create_model",
    "create_custom_network",
    "load_pretrained_backbone",
    "load_resnet_backbone",
    "load_tresnet_backbone",
    "load_cvt_backbone",
    "RandomWalkAttention",
    "MultiOrderContextMappingNetwork",
]
