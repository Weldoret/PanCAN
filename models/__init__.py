"""
Models module initialization

Contains all components for multi-scale context-aware network.
"""

from .neighborhood import (
    NeighborhoodSystem,
    AdjacencyMatrixGenerator,
    generate_adjacency_index_matrix,
    generate_weight_matrix,
    DirectionalNeighborhood
)

from .context_kernel import (
    ContextAwareKernel,
    ContextAwareKernelMap,
    KernelMappingLayer,
    FidelityCriterion,
    ContextCriterion,
    KernelRegularizer
)

from .multi_order import (
    MultiOrderNeighborhood,
    MultiOrderContextAggregator,
    MultiOrderContextLayer,
    HigherOrderContext,
    NeighborhoodOrderManager
)

from .random_walk import (
    RandomWalkAttention,
    RandomWalkContextAggregator,
    TransitionProbabilityCalculator,
    AttentionBasedRandomWalk,
    RandomWalkWithThreshold
)

from .multi_scale import (
    MultiScaleFeatureAggregator,
    SlidingWindowAggregator,
    MultiScaleFusionLayer,
    AnchorBoxGenerator,
    CenteredSelfAttention,
    SubregionSelector,
    MultiHeadScaleAttention
)

from .deep_kernel import (
    DeepKernelMappingNetwork,
    ContextAwareKernelLayer,
    KernelRecursiveMapping,
    ExplicitKernelMap,
    DynamicKernelNetwork
)

from .network import (
    MultiScaleContextAwareNetwork,
    MultiLabelClassifier,
    FeatureExtractor,
    BackboneFactory,
    CustomNetwork
)

from .pretrained import (
    load_resnet_backbone,
    load_tresnet_backbone,
    load_cvt_backbone,
    load_pretrained_backbone,
    BACKBONES
)


def create_model(config, num_classes=None, device=None):
    from .network import MultiScaleContextAwareNetwork
    from .pretrained import load_pretrained_backbone
    
    # 设置默认值
    if num_classes is None:
        num_classes = config.network.num_classes
    
    # 创建主干网络
    backbone = load_pretrained_backbone(
        config.network.backbone_name,
        pretrained=config.network.backbone_pretrained
    )
    
    # 创建模型
    model = MultiScaleContextAwareNetwork(
        backbone=backbone,
        num_classes=num_classes,
        config=config.network,
        device=device
    )
    
    return model


def create_custom_network(config, device=None):
    from .network import CustomNetwork
    
    # 生成邻域矩阵
    connect_idx, weights_flag = generate_adjacency_index_matrix(
        config.network.grid_rows,
        config.network.grid_cols
    )
    
    # 转换到设备
    if device is not None:
        connect_idx = connect_idx.to(device)
        weights_flag = weights_flag.to(device)
    
    # 创建网络
    model = CustomNetwork(
        num_classes=config.network.num_classes,
        connect_idx=connect_idx,
        weights_flag=weights_flag,
        device=device,
        num_blocks=config.network.num_blocks
    )
    
    return model


# 导出列表
__all__ = [
    'MultiScaleContextAwareNetwork',
    'CustomNetwork',
    'create_model',
    'create_custom_network',
    'NeighborhoodSystem',
    'ContextAwareKernel',
    'MultiOrderContextAggregator',
    'RandomWalkAttention',
    'MultiScaleFeatureAggregator',
    'DeepKernelMappingNetwork',
    'load_resnet_backbone',
    'load_tresnet_backbone',
    'load_cvt_backbone',
    'load_pretrained_backbone',
    'MultiLabelClassifier',
    'FeatureExtractor',
    'BackboneFactory',
    'generate_adjacency_index_matrix',
    'generate_weight_matrix'
]