"""
Multi-Scale Context-Aware Deep Kernel Mapping Network for Multi-Label Classification

基于多尺度特征融合的上下文感知深度核映射网络，用于多标签图像分类。
该实现基于论文 "Multi-Scale Feature Fusion-based Context-Aware Deep Kernel Mapping Network for Multi-label Classification"。
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入核心模块
from .config.base_config import NetworkConfig, DatasetConfig, TrainingConfig
from .config.dataset_configs import NUSWIDEConfig, VOC2007Config, COCOConfig

# 导入邻域系统模块
from .models.neighborhood import (
    NeighborhoodSystem,
    generate_adjacency_matrix,
    generate_adjacency_index_matrix
)

# 导入上下文感知核模块
from .models.context_kernel import (
    ContextAwareKernel,
    ContextAwareKernelMap,
    KernelMappingLayer
)

# 导入多阶上下文模块
from .models.multi_order import (
    MultiOrderNeighborhood,
    MultiOrderContextAggregator,
    MultiOrderContextLayer
)

# 导入随机游走注意力模块
from .models.random_walk import (
    RandomWalkAttention,
    RandomWalkContextAggregator,
    TransitionProbabilityCalculator
)

# 导入多尺度特征聚合模块
from .models.multi_scale import (
    MultiScaleFeatureAggregator,
    SlidingWindowAggregator,
    MultiScaleFusionLayer,
    AnchorBoxGenerator,
    CenteredSelfAttention
)

# 导入深度核映射模块
from .models.deep_kernel import (
    DeepKernelMappingNetwork,
    ContextAwareKernelLayer,
    KernelRecursiveMapping
)

# 导入网络架构模块
from .models.network import (
    MultiScaleContextAwareNetwork,
    MultiLabelClassifier,
    FeatureExtractor,
    BackboneFactory
)

# 导入训练模块
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

# 导入数据集模块
from .datasets.nuswide import NUSWIDEDataset
from .datasets.voc2007 import VOC2007Dataset
from .datasets.coco import COCODataset
from .datasets.data_loader import (
    create_dataloader,
    create_multilabel_dataloader,
    DataLoaderFactory
)

# 导入工具模块
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

# 导入脚本模块
from .scripts.train import train_main
from .scripts.test import test_main
from .scripts.inference import inference_main

# 预训练模型
from .models.pretrained import (
    load_resnet_backbone,
    load_tresnet_backbone,
    load_cvt_backbone,
    BACKBONES
)

# 简化的API接口
def create_network(config_path=None, **kwargs):
    """
    创建多尺度上下文感知网络
    
    Args:
        config_path (str, optional): 配置文件路径
        **kwargs: 配置参数
    
    Returns:
        MultiScaleContextAwareNetwork: 初始化的网络
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
    
    # 创建主干网络
    backbone = load_resnet_backbone(config.backbone_name)
    
    # 创建网络
    network = MultiScaleContextAwareNetwork(
        backbone=backbone,
        num_classes=config.num_classes,
        config=config
    )
    
    return network

def create_dataset(dataset_name, config=None):
    """
    创建数据集
    
    Args:
        dataset_name (str): 数据集名称 ('nuswide', 'voc2007', 'coco')
        config: 配置对象
    
    Returns:
        Dataset: 数据集对象
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
    创建训练管理器
    
    Args:
        model: 要训练的模型
        config: 训练配置
    
    Returns:
        TrainingManager: 训练管理器
    """
    return TrainingManager(model=model, config=config)

# 导出常用函数
__all__ = [
    # 创建函数
    'create_network',
    'create_dataset',
    'create_trainer',
    
    # 配置类
    'NetworkConfig',
    'DatasetConfig',
    'TrainingConfig',
    
    # 网络类
    'MultiScaleContextAwareNetwork',
    'MultiLabelClassifier',
    
    # 核心模块
    'ContextAwareKernel',
    'MultiOrderContextAggregator',
    'RandomWalkAttention',
    'MultiScaleFeatureAggregator',
    'DeepKernelMappingNetwork',
    
    # 数据集
    'NUSWIDEDataset',
    'VOC2007Dataset',
    'COCODataset',
    
    # 训练
    'TrainingManager',
    'Evaluator',
    
    # 评估指标
    'compute_mAP',
    'compute_CF1',
    'compute_OF1',
    'compute_all_metrics',
    
    # 工具函数
    'save_checkpoint',
    'load_checkpoint',
    'visualize_attention_maps',
    'setup_logger'
]

# 设置默认日志记录器
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

# 初始化消息
logger.info(f"Multi-Scale Context-Aware Network v{__version__}")
logger.info("Initialized with modules: config, models, training, datasets, utils")

# 环境检查
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