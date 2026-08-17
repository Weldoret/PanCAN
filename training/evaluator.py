"""
Evaluator module

Implements model evaluation functionality, including validation and test set evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import warnings

from .metrics import MultiLabelMetrics, compute_all_metrics


class PerformanceMetrics:
    """Performance metrics storage class"""
    
    def __init__(self):
        self.metrics = {}
        self.best_metrics = {}
        self.epoch_metrics = []
    
    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Update metrics
        
        Args:
            metrics: Metrics dictionary
            epoch: Current epoch
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
        
        if epoch is not None:
            epoch_record = {'epoch': epoch, **metrics}
            self.epoch_metrics.append(epoch_record)
        
        self._update_best_metrics(metrics)
    
    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Update best metrics"""
        for name, value in metrics.items():
            if 'loss' in name.lower():
                if name not in self.best_metrics or value < self.best_metrics[name]['value']:
                    self.best_metrics[name] = {'value': value, 'epoch': len(self.metrics[name]) - 1}
            elif any(keyword in name.lower() for keyword in ['acc', 'f1', 'map', 'precision', 'recall']):
                if name not in self.best_metrics or value > self.best_metrics[name]['value']:
                    self.best_metrics[name] = {'value': value, 'epoch': len(self.metrics[name]) - 1}
    
    def get_best(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get best metric value
        
        Args:
            metric_name: Metric name
            
        Returns:
            Best metric information dictionary, or None
        """
        return self.best_metrics.get(metric_name)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get latest metric value
        
        Args:
            metric_name: Metric name
            
        Returns:
            Latest metric value, or None
        """
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return self.metrics[metric_name][-1]
        return None
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get metric average
        
        Args:
            metric_name: Metric name
            last_n: Last n values, if None use all values
            
        Returns:
            Average value, or None
        """
        if metric_name not in self.metrics:
            return None
        
        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]
        
        if len(values) > 0:
            return np.mean(values)
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary
        
        Returns:
            Summary dictionary
        """
        summary = {
            'best_metrics': self.best_metrics,
            'latest_metrics': {},
            'num_epochs': len(self.epoch_metrics)
        }
        
        for name in self.metrics:
            summary['latest_metrics'][name] = self.get_latest(name)
        
        return summary


class ConfusionMatrix:
    """Multi-label confusion matrix"""
    
    def __init__(self, num_classes: int):
        """
        Initialize confusion matrix
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset confusion matrix"""
        self.tp = np.zeros(self.num_classes, dtype=np.float64)
        self.fp = np.zeros(self.num_classes, dtype=np.float64)
        self.tn = np.zeros(self.num_classes, dtype=np.float64)
        self.fn = np.zeros(self.num_classes, dtype=np.float64)
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update confusion matrix
        
        Args:
            y_true: True labels [n_samples, n_classes]
            y_pred: Predicted labels [n_samples, n_classes]
        """
        y_true_bin = (y_true > 0.5).astype(int)
        y_pred_bin = (y_pred > 0.5).astype(int)
        
        for c in range(self.num_classes):
            true_c = y_true_bin[:, c]
            pred_c = y_pred_bin[:, c]
            
            self.tp[c] += np.sum((true_c == 1) & (pred_c == 1))
            self.fp[c] += np.sum((true_c == 0) & (pred_c == 1))
            self.tn[c] += np.sum((true_c == 0) & (pred_c == 0))
            self.fn[c] += np.sum((true_c == 1) & (pred_c == 0))
    
    def get_class_metrics(self, class_idx: int) -> Dict[str, float]:
        """
        Get metrics for a single class
        
        Args:
            class_idx: Class index
            
        Returns:
            Class metrics dictionary
        """
        tp = self.tp[class_idx]
        fp = self.fp[class_idx]
        tn = self.tn[class_idx]
        fn = self.fn[class_idx]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'support': tp + fn
        }
    
    def get_overall_metrics(self) -> Dict[str, float]:
        """
        Get overall metrics
        
        Returns:
            Overall metrics dictionary
        """
        total_tp = np.sum(self.tp)
        total_fp = np.sum(self.fp)
        total_tn = np.sum(self.tn)
        total_fn = np.sum(self.fn)
        
        class_precisions = []
        class_recalls = []
        
        for c in range(self.num_classes):
            class_metrics = self.get_class_metrics(c)
            class_precisions.append(class_metrics['precision'])
            class_recalls.append(class_metrics['recall'])
        
        macro_precision = np.mean(class_precisions)
        macro_recall = np.mean(class_recalls)
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'accuracy': (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0.0
        }


class ROC_AUC_Calculator:
    """ROC AUC calculator"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset"""
        self.y_true_list = []
        self.y_score_list = []
    
    def update(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Update data
        
        Args:
            y_true: True labels [n_samples, n_classes]
            y_score: Predicted scores [n_samples, n_classes]
        """
        self.y_true_list.append(y_true)
        self.y_score_list.append(y_score)
    
    def compute_auc(self) -> Dict[str, float]:
        """
        Compute AUC
        
        Returns:
            AUC dictionary
        """
        if not self.y_true_list:
            return {}
        
        from sklearn.metrics import roc_auc_score
        
        y_true_all = np.vstack(self.y_true_list)
        y_score_all = np.vstack(self.y_score_list)
        
        auc_scores = {}
        
        try:
            for c in range(self.num_classes):
                if np.sum(y_true_all[:, c]) > 0:
                    auc = roc_auc_score(y_true_all[:, c], y_score_all[:, c])
                    auc_scores[f'auc_class_{c}'] = auc
            
            if auc_scores:
                auc_scores['macro_auc'] = np.mean(list(auc_scores.values()))
            
            try:
                micro_auc = roc_auc_score(y_true_all.ravel(), y_score_all.ravel())
                auc_scores['micro_auc'] = micro_auc
            except:
                pass
            
        except Exception as e:
            warnings.warn(f"Error computing AUC: {e}")
        
        return auc_scores


class PrecisionRecallCalculator:
    """Precision-Recall calculator"""
    
    def __init__(self):
        self.y_true_list = []
        self.y_score_list = []
    
    def reset(self):
        """Reset"""
        self.y_true_list = []
        self.y_score_list = []
    
    def update(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Update data
        
        Args:
            y_true: True labels
            y_score: Predicted scores
        """
        self.y_true_list.append(y_true)
        self.y_score_list.append(y_score)
    
    def compute_pr_curves(self):
        """
        Compute PR curves
        
        Returns:
            PR curve data
        """
        if not self.y_true_list:
            return {}
        
        from sklearn.metrics import precision_recall_curve
        
        y_true_all = np.vstack(self.y_true_list)
        y_score_all = np.vstack(self.y_score_list)
        
        n_classes = y_true_all.shape[1]
        pr_data = {}
        
        for c in range(n_classes):
            if np.sum(y_true_all[:, c]) > 0:
                precision, recall, thresholds = precision_recall_curve(
                    y_true_all[:, c], y_score_all[:, c]
                )
                pr_data[f'class_{c}'] = {
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds
                }
        
        return pr_data


class Evaluator:
    """Model evaluator"""
    
    def __init__(self, config):
        """
        Initialize evaluator
        
        Args:
            config: Configuration object
        """
        self.config = config
        dataset = getattr(config, 'dataset', None)
        self.dataset_name = getattr(dataset, 'dataset_name', '')
        self.metrics_calculator = MultiLabelMetrics(
            threshold=0.5,
            top_k=None
        )
        self.top3_metrics_calculator = MultiLabelMetrics(
            threshold=0.5,
            top_k=None
        )
        self.performance_metrics = PerformanceMetrics()
        self.device = torch.device(config.training.device if hasattr(config, 'training') else 'cuda')
    
    def evaluate(self,
                model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: Optional[nn.Module] = None,
                mode: str = 'val') -> Dict[str, Any]:
        """
        Evaluate model
        
        Args:
            model: Model
            dataloader: Data loader
            criterion: Loss function
            mode: Evaluation mode ('val', 'test')
            
        Returns:
            Evaluation results dictionary
        """
        model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_scores = []
        
        num_classes = self._get_num_classes(model)
        confusion_matrix = ConfusionMatrix(num_classes)
        auc_calculator = ROC_AUC_Calculator(num_classes)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Evaluating ({mode})")):
                if isinstance(batch_data, (list, tuple)):
                    inputs, labels = batch_data[0], batch_data[1]
                else:
                    inputs, labels = batch_data['image'], batch_data['label']
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                
                scores = torch.sigmoid(outputs)

                predictions = (scores > self.metrics_calculator.threshold).float()
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
        
        if all_predictions:
            all_predictions = np.vstack(all_predictions)
            all_labels = np.vstack(all_labels)
            all_scores = np.vstack(all_scores)
        else:
            all_predictions = np.array([])
            all_labels = np.array([])
            all_scores = np.array([])
        
        metrics = {}
        
        if criterion is not None:
            metrics['loss'] = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
        
        ml_metrics = self.metrics_calculator.compute_all(all_labels, all_predictions, all_scores)
        metrics.update(ml_metrics)

        if self.dataset_name == 'coco':
            metrics.update({f'all_{name}': value for name, value in ml_metrics.items()})

            top3_predictions = self.top3_metrics_calculator.get_topk_predictions(
                all_scores, 3
            )
            top3_metrics = self.top3_metrics_calculator.compute_all(
                all_labels, top3_predictions, all_scores
            )
            metrics.update({f'top3_{name}': value for name, value in top3_metrics.items()})
        
        confusion_matrix.update(all_labels, all_predictions)
        confusion_metrics = confusion_matrix.get_overall_metrics()
        metrics.update({f'cm_{k}': v for k, v in confusion_metrics.items()})
        
        auc_calculator.update(all_labels, all_scores)
        auc_metrics = auc_calculator.compute_auc()
        metrics.update(auc_metrics)
        
        self.performance_metrics.update(metrics)
        
        results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'scores': all_scores,
            'num_samples': len(all_labels)
        }
        
        return results
    
    def evaluate_with_thresholds(self,
                                model: nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                thresholds: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model with different thresholds
        
        Args:
            model: Model
            dataloader: Data loader
            thresholds: List of thresholds
            
        Returns:
            Metrics dictionary for each threshold
        """
        model.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Collecting scores"):
                if isinstance(batch_data, (list, tuple)):
                    inputs, labels = batch_data[0], batch_data[1]
                else:
                    inputs, labels = batch_data['image'], batch_data['label']
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                scores = torch.sigmoid(outputs)
                
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_scores = np.vstack(all_scores) if all_scores else np.array([])
        all_labels = np.vstack(all_labels) if all_labels else np.array([])
        
        threshold_metrics = {}
        
        for threshold in thresholds:
            predictions = (all_scores > threshold).astype(float)
            metrics = self.metrics_calculator.compute_all(all_labels, predictions, all_scores)
            threshold_metrics[f'threshold_{threshold:.3f}'] = metrics
        
        return threshold_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Performance summary
        """
        return self.performance_metrics.get_summary()
    
    def get_best_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get best metric
        
        Args:
            metric_name: Metric name
            
        Returns:
            Best metric information
        """
        return self.performance_metrics.get_best(metric_name)
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = PerformanceMetrics()
    
    def _get_num_classes(self, model: nn.Module) -> int:
        """Get model number of classes"""
        if hasattr(model, 'num_classes'):
            return model.num_classes
        elif hasattr(model, 'classifier'):
            if hasattr(model.classifier, 'num_classes'):
                return model.classifier.num_classes
        return 81
