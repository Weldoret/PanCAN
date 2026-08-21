"""PanCAN model, configuration, training, and utility exports."""

from .config import (
    COCOConfig,
    DatasetConfig,
    ExperimentConfig,
    NetworkConfig,
    NUSWIDEConfig,
    TrainingConfig,
    VOC2007Config,
)
from .models.context_kernel import ContextAwareKernel, ContextAwareKernelMap
from .models.deep_kernel import DeepKernelMappingNetwork
from .models.multi_order import MultiOrderContextAggregator
from .models.multi_scale import MultiScaleFeatureAggregator
from .models.network import MultiLabelClassifier, MultiScaleContextAwareNetwork
from .models.neighborhood import NeighborhoodSystem, generate_adjacency_index_matrix
from .models.pretrained import (
    BACKBONES,
    load_cvt_backbone,
    load_pretrained_backbone,
    load_resnet_backbone,
    load_tresnet_backbone,
)
from .models.random_walk import MultiOrderContextMappingNetwork, RandomWalkAttention
from .training import (
    Evaluator,
    MultiLabelMetrics,
    MultiLabelTrainer,
    PerformanceMetrics,
    TrainingManager,
    compute_CF1,
    compute_OF1,
    compute_all_metrics,
    compute_mAP,
    create_evaluator,
    create_trainer,
)
from .utils import (
    CheckpointManager,
    create_data_loaders,
    load_checkpoint,
    load_model,
    plot_metrics,
    save_checkpoint,
    save_model,
    setup_experiment,
    setup_logger,
)

__version__ = "1.0.0"


def create_network(config_path=None, **kwargs):
    """Create a network from a config file or ``NetworkConfig`` arguments."""
    if config_path is not None:
        config = NetworkConfig.from_file(config_path)
    else:
        config = NetworkConfig(**kwargs)

    backbone = load_resnet_backbone(config.backbone_name)
    return MultiScaleContextAwareNetwork(
        backbone=backbone,
        num_classes=config.num_classes,
        config=config,
    )


__all__ = [
    "BACKBONES",
    "COCOConfig",
    "ContextAwareKernel",
    "ContextAwareKernelMap",
    "DatasetConfig",
    "DeepKernelMappingNetwork",
    "Evaluator",
    "ExperimentConfig",
    "MultiLabelClassifier",
    "MultiLabelMetrics",
    "MultiOrderContextAggregator",
    "MultiScaleContextAwareNetwork",
    "MultiScaleFeatureAggregator",
    "MultiLabelTrainer",
    "NetworkConfig",
    "NeighborhoodSystem",
    "NUSWIDEConfig",
    "PerformanceMetrics",
    "RandomWalkAttention",
    "MultiOrderContextMappingNetwork",
    "TrainingConfig",
    "TrainingManager",
    "VOC2007Config",
    "compute_CF1",
    "compute_OF1",
    "compute_all_metrics",
    "compute_mAP",
    "create_data_loaders",
    "create_evaluator",
    "create_network",
    "create_trainer",
    "generate_adjacency_index_matrix",
    "load_checkpoint",
    "load_cvt_backbone",
    "load_model",
    "load_pretrained_backbone",
    "load_resnet_backbone",
    "load_tresnet_backbone",
    "plot_metrics",
    "save_checkpoint",
    "save_model",
    "setup_experiment",
    "setup_logger",
]
