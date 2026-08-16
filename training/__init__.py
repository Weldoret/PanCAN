"""Training, evaluation, and metric helpers present in this checkout."""

from .evaluator import Evaluator, PerformanceMetrics
from .metrics import MultiLabelMetrics, compute_all_metrics, compute_CF1, compute_mAP, compute_OF1
from .trainer import MultiLabelTrainer, TrainingManager


def create_trainer(model, config, device=None):
    return TrainingManager(model=model, config=config, device=device)


def create_evaluator(config):
    return Evaluator(config)


def get_metrics_calculator():
    return MultiLabelMetrics()


__all__ = [
    "TrainingManager",
    "MultiLabelTrainer",
    "create_trainer",
    "Evaluator",
    "PerformanceMetrics",
    "create_evaluator",
    "MultiLabelMetrics",
    "compute_mAP",
    "compute_CF1",
    "compute_OF1",
    "compute_all_metrics",
    "get_metrics_calculator",
]
