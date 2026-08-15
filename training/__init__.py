"""
Training module initialization file

Contains training, evaluation, and metric calculation functionality.
"""

# Training module
from .trainer import (
    TrainingManager,
    MultiLabelTrainer,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    GradientAccumulator,
    MixedPrecisionTrainer
)

# Evaluation module
from .evaluator import (
    Evaluator,
    PerformanceMetrics,
    ConfusionMatrix,
    ROC_AUC_Calculator,
    PrecisionRecallCalculator
)

# Metrics module
from .metrics import (
    compute_mAP,
    compute_CF1,
    compute_OF1,
    compute_all_metrics,
    MultiLabelMetrics,
    AveragePrecision,
    F1Score,
    PrecisionAtK
)

# Loss functions
from .losses import (
    MultiLabelLoss,
    AsymmetricLoss,
    FocalLoss,
    BCEWithLogitsLoss,
    WeightedBCELoss
)

# Optimizers
from .optimizers import (
    create_optimizer,
    create_scheduler,
    OptimizerFactory,
    SchedulerFactory
)

# Data processing
from .data_utils import (
    create_data_loaders,
    DataSplitter,
    DataAugmentor,
    LabelBalancer
)

# Simplified API
def create_trainer(model, config, device=None):
    """
    Create training manager
    
    Args:
        model: Model
        config: Configuration
        device: Device
        
    Returns:
        Training manager
    """
    from .trainer import TrainingManager
    return TrainingManager(model=model, config=config, device=device)

def create_evaluator(config):
    """
    Create evaluator
    
    Args:
        config: Configuration
        
    Returns:
        Evaluator
    """
    from .evaluator import Evaluator
    return Evaluator(config)

def get_metrics_calculator():
    """
    Get metrics calculator
    
    Returns:
        Metrics calculator
    """
    from .metrics import MultiLabelMetrics
    return MultiLabelMetrics()

# Export list
__all__ = [
    # Training
    'TrainingManager',
    'MultiLabelTrainer',
    'create_trainer',
    
    # Evaluation
    'Evaluator',
    'PerformanceMetrics',
    'create_evaluator',
    
    # Metrics
    'compute_mAP',
    'compute_CF1',
    'compute_OF1',
    'compute_all_metrics',
    'MultiLabelMetrics',
    'get_metrics_calculator',
    
    # Loss functions
    'MultiLabelLoss',
    'AsymmetricLoss',
    'FocalLoss',
    
    # Optimizers
    'create_optimizer',
    'create_scheduler',
    
    # Utilities
    'create_data_loaders',
    'EarlyStopping',
    'ModelCheckpoint'
]
