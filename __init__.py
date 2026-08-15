"""
Multi-Scale Context-Aware Deep Kernel Mapping Network for Multi-Label Classification

Context-aware deep kernel mapping network based on multi-scale feature fusion for multi-label image classification.
This implementation is based on the paper "Multi-Scale Feature Fusion-based Context-Aware Deep Kernel Mapping Network for Multi-label Classification".
"""

# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import core modules
from .config.base_config import NetworkConfig, DatasetConfig, TrainingConfig
from .config.dataset_configs import NUSWIDEConfig, VOC2007Config, COCOConfig

# Import neighborhood system modules
from .models.neighborhood import (
    NeighborhoodSystem,
    generate_adjacency_matrix,
    generate_adjacency_index_matrix
)

# Import context-aware kernel modules
from .models.context_kernel import (
    ContextAwareKernel,
    ContextAwareKernelMap,
    KernelMappingLayer
)

# Import multi-order context modules
from .models.multi_order import (
    MultiOrderNeighborhood,
    MultiOrderContextAggregator,
    MultiOrderContextLayer
)

# Import random-walk attention modules
from .models.random_walk import (
    RandomWalkAttention,
    RandomWalkContextAggregator,
    TransitionProbabilityCalculator
)

# Import multi-scale feature aggregation modules
from .models.multi_scale import (
    MultiScaleFeatureAggregator,
    SlidingWindowAggregator,
    MultiScaleFusionLayer,
    AnchorBoxGenerator,
    CenteredSelfAttention
)

# Import deep kernel mapping modules
from .models.deep_kernel import (
    DeepKernelMappingNetwork,
    ContextAwareKernelLayer,
    KernelRecursiveMapping
)

# Import network architecture modules
from .models.network import (
    MultiScaleContextAwareNetwork,
    MultiLabelClassifier,
    FeatureExtractor,
    BackboneFactory
)

# Import training modules
from .training.trainer import (
    TrainingManager,
    MultiLabelTrainer,
    EarlyStopping,
    LearningRateScheduler
)

from .training.evaluator import (
    Evaluator,
    MetricsCalculator,
    PerformanceMetrics
)

from .training.metrics import (
    compute_mAP,
    compute_CF1,
    compute_OF1,
    compute_all_metrics
)

# Import dataset modules
from .datasets.nuswide import NUSWIDEDataset
from .datasets.voc2007 import VOC2007Dataset
from .datasets.coco import COCODataset
from .datasets.data_loader import (
    create_dataloader,
    create_multilabel_dataloader,
    DataLoaderFactory
)

# Import utility modules
from .utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model,
    CheckpointManager
)

from .utils.visualization import (
    visualize_attention_maps,
    visualize_neighborhood,
    visualize_multi_scale_features,
    plot_metrics,
    save_visualization
)

from .utils.logger import (
    setup_logger,
    get_logger,
    log_metrics,
    log_config
)

# Import script modules
from .scripts.train import train_main
from .scripts.test import test_main
from .scripts.inference import inference_main

# Pretrained models
from .models.pretrained import (
    load_resnet_backbone,
    load_tresnet_backbone,
    load_cvt_backbone,
    BACKBONES
)

# Simplified API
def create_network(config_path=None, **kwargs):
    """
    Create a multi-scale context-aware network
    
    Args:
        config_path (str, optional): Path to the configuration file
        **kwargs: Configuration parameters
    
    Returns:
        MultiScaleContextAwareNetwork: Initialized network
    """
    from .config.base_config import NetworkConfig
    from .models.network import MultiScaleContextAwareNetwork
    from .models.pretrained import load_resnet_backbone
    
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = NetworkConfig.from_dict(config_dict)
    else:
        config = NetworkConfig(**kwargs)
    
    # Create backbone network
    backbone = load_resnet_backbone(config.backbone_name)
    
    # Create network
    network = MultiScaleContextAwareNetwork(
        backbone=backbone,
        num_classes=config.num_classes,
        config=config
    )
    
    return network

def create_dataset(dataset_name, config=None):
    """
    Create a dataset
    
    Args:
        dataset_name (str): Dataset name ('nuswide', 'voc2007', 'coco')
        config: Configuration object
    
    Returns:
        Dataset: Dataset object
    """
    dataset_map = {
        'nuswide': NUSWIDEDataset,
        'voc2007': VOC2007Dataset,
        'coco': COCODataset
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[dataset_name.lower()]
    return dataset_class(config=config)

def create_trainer(model, config):
    """
    Create a training manager
    
    Args:
        model: Model to train
        config: Training configuration
    
    Returns:
        TrainingManager: Training manager
    """
    return TrainingManager(model=model, config=config)

# Export commonly used functions
__all__ = [
    # Creation functions
    'create_network',
    'create_dataset',
    'create_trainer',
    
    # Configuration classes
    'NetworkConfig',
    'DatasetConfig',
    'TrainingConfig',
    
    # Network classes
    'MultiScaleContextAwareNetwork',
    'MultiLabelClassifier',
    
    # Core modules
    'ContextAwareKernel',
    'MultiOrderContextAggregator',
    'RandomWalkAttention',
    'MultiScaleFeatureAggregator',
    'DeepKernelMappingNetwork',
    
    # Datasets
    'NUSWIDEDataset',
    'VOC2007Dataset',
    'COCODataset',
    
    # Training
    'TrainingManager',
    'Evaluator',
    
    # Evaluation metrics
    'compute_mAP',
    'compute_CF1',
    'compute_OF1',
    'compute_all_metrics',
    
    # Utility functions
    'save_checkpoint',
    'load_checkpoint',
    'visualize_attention_maps',
    'setup_logger'
]

# Set up the default logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Initialization messages
logger.info(f"Multi-Scale Context-Aware Network v{__version__}")
logger.info("Initialized with modules: config, models, training, datasets, utils")

# Environment check
try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
    else:
        logger.warning("CUDA not available, using CPU")
except ImportError:
    logger.error("PyTorch not installed. Please install PyTorch to use this package.")
