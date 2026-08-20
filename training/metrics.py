
"""
Metrics module

Implements multi-label classification evaluation metrics.
"""

import torch
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss,
    multilabel_confusion_matrix
)
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings


class MultiLabelMetrics:
    """Multi-label classification metrics calculator"""
    
    def __init__(self, 
                 average_method: str = 'macro',
                 threshold: float = 0.5,
                 top_k: Optional[int] = None):
        """
        Initialize multi-label metrics calculator
        
        Args:
            average_method: Average method for metrics
            threshold: Threshold for binary classification
            top_k: Top-K value for evaluation
        """
        self.average_method = average_method
        self.threshold = threshold
        self.top_k = top_k
        self.metric_history = {}
    
    def compute_all(self, 
                   y_true: Union[np.ndarray, torch.Tensor],
                   y_pred: Union[np.ndarray, torch.Tensor],
                   y_score: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Compute all metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_score: Predicted scores
            
        Returns:
            Metrics dictionary
        """
        y_true_np = self._to_numpy(y_true)
        y_pred_np = self._to_numpy(y_pred)
        
        if y_score is not None:
            y_score_np = self._to_numpy(y_score)
        else:
            y_score_np = y_pred_np
        
        metrics = {}
        
        metrics['accuracy'] = self.compute_accuracy(y_true_np, y_pred_np)
        metrics['hamming_loss'] = self.compute_hamming_loss(y_true_np, y_pred_np)
        
        metrics['micro_f1'] = self.compute_f1_score(y_true_np, y_pred_np, average='micro')
        metrics['macro_f1'] = self.compute_f1_score(y_true_np, y_pred_np, average='macro')
        metrics['samples_f1'] = self.compute_f1_score(y_true_np, y_pred_np, average='samples')
        
        metrics['precision'] = self.compute_precision(y_true_np, y_pred_np)
        metrics['recall'] = self.compute_recall(y_true_np, y_pred_np)
        
        if y_score_np is not None:
            metrics['map'] = self.compute_map(y_true_np, y_score_np)
        
        metrics['cf1'] = self.compute_cf1(y_true_np, y_pred_np)
        metrics['of1'] = self.compute_of1(y_true_np, y_pred_np)
        
        if self.top_k is not None:
            y_pred_topk = self.get_topk_predictions(y_score_np, self.top_k)
            metrics[f'precision_at_{self.top_k}'] = self.compute_precision_at_k(y_true_np, y_pred_topk)
            metrics[f'recall_at_{self.top_k}'] = self.compute_recall_at_k(y_true_np, y_pred_topk)
            metrics[f'f1_at_{self.top_k}'] = self.compute_f1_at_k(y_true_np, y_pred_topk)
        
        self._update_history(metrics)
        
        return metrics
    
    def compute_map(self, 
                   y_true: np.ndarray,
                   y_score: np.ndarray) -> float:
        """
        Compute mean average precision
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            
        Returns:
            Mean average precision
        """
        try:
            n_classes = y_true.shape[1]
            ap_scores = []
            
            for i in range(n_classes):
                if np.sum(y_true[:, i]) > 0:
                    ap = average_precision_score(y_true[:, i], y_score[:, i])
                    ap_scores.append(ap)
            
            if len(ap_scores) > 0:
                return np.mean(ap_scores)
            else:
                return 0.0
        except Exception as e:
            warnings.warn(f"Error computing mAP: {e}")
            return 0.0
    
    def compute_cf1(self, 
                   y_true: np.ndarray,
                   y_pred: np.ndarray) -> float:
        """Compute F1 from the mean per-class precision and recall."""
        y_true_bin = y_true > 0.5
        y_pred_bin = y_pred > 0.5
        true_positives = np.sum(y_true_bin & y_pred_bin, axis=0)
        predicted = np.sum(y_pred_bin, axis=0)
        positives = np.sum(y_true_bin, axis=0)
        precision = np.divide(
            true_positives, predicted, out=np.zeros_like(true_positives, dtype=float),
            where=predicted != 0,
        ).mean()
        recall = np.divide(
            true_positives, positives, out=np.zeros_like(true_positives, dtype=float),
            where=positives != 0,
        ).mean()
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    def compute_of1(self, 
                   y_true: np.ndarray,
                   y_pred: np.ndarray) -> float:
        """Compute F1 from precision and recall aggregated over all labels."""
        y_true_bin = y_true > 0.5
        y_pred_bin = y_pred > 0.5
        true_positives = np.sum(y_true_bin & y_pred_bin)
        predicted = np.sum(y_pred_bin)
        positives = np.sum(y_true_bin)
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / positives if positives else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    def compute_f1_score(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        average: str = 'macro') -> float:
        """
        Compute F1 score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Average method
            
        Returns:
            F1 score
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            return f1_score(y_true_bin, y_pred_bin, average=average, zero_division=0)
        except Exception as e:
            warnings.warn(f"Error computing F1 score: {e}")
            return 0.0
    
    def compute_precision(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> float:
        """
        Compute precision
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Precision
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            return precision_score(y_true_bin, y_pred_bin, average=self.average_method, zero_division=0)
        except Exception as e:
            warnings.warn(f"Error computing precision: {e}")
            return 0.0
    
    def compute_recall(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray) -> float:
        """
        Compute recall
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Recall
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            return recall_score(y_true_bin, y_pred_bin, average=self.average_method, zero_division=0)
        except Exception as e:
            warnings.warn(f"Error computing recall: {e}")
            return 0.0
    
    def compute_accuracy(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray) -> float:
        """
        Compute accuracy
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            return accuracy_score(y_true_bin, y_pred_bin)
        except Exception as e:
            warnings.warn(f"Error computing accuracy: {e}")
            return 0.0
    
    def compute_hamming_loss(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray) -> float:
        """
        Compute hamming loss
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Hamming loss
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            return hamming_loss(y_true_bin, y_pred_bin)
        except Exception as e:
            warnings.warn(f"Error computing hamming loss: {e}")
            return 1.0
    
    def compute_precision_at_k(self,
                             y_true: np.ndarray,
                             y_pred_topk: np.ndarray) -> float:
        """
        Compute precision at K
        
        Args:
            y_true: True labels
            y_pred_topk: Top-K predictions
            
        Returns:
            Precision at K
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            
            intersection = np.sum(y_true_bin * y_pred_topk, axis=1)
            precision_per_sample = intersection / np.sum(y_pred_topk, axis=1)
            
            valid_samples = np.sum(y_pred_topk, axis=1) > 0
            if np.any(valid_samples):
                return np.mean(precision_per_sample[valid_samples])
            else:
                return 0.0
        except Exception as e:
            warnings.warn(f"Error computing precision at K: {e}")
            return 0.0
    
    def compute_recall_at_k(self,
                          y_true: np.ndarray,
                          y_pred_topk: np.ndarray) -> float:
        """
        Compute recall at K
        
        Args:
            y_true: True labels
            y_pred_topk: Top-K predictions
            
        Returns:
            Recall at K
        """
        try:
            y_true_bin = (y_true > 0.5).astype(int)
            
            intersection = np.sum(y_true_bin * y_pred_topk, axis=1)
            recall_per_sample = intersection / np.maximum(np.sum(y_true_bin, axis=1), 1.0)
            
            return np.mean(recall_per_sample)
        except Exception as e:
            warnings.warn(f"Error computing recall at K: {e}")
            return 0.0
    
    def compute_f1_at_k(self,
                       y_true: np.ndarray,
                       y_pred_topk: np.ndarray) -> float:
        """
        Compute F1 at K
        
        Args:
            y_true: True labels
            y_pred_topk: Top-K predictions
            
        Returns:
            F1 at K
        """
        try:
            precision = self.compute_precision_at_k(y_true, y_pred_topk)
            recall = self.compute_recall_at_k(y_true, y_pred_topk)
            
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
            else:
                return 0.0
        except Exception as e:
            warnings.warn(f"Error computing F1 at K: {e}")
            return 0.0
    
    def get_topk_predictions(self,
                           y_score: np.ndarray,
                           k: int) -> np.ndarray:
        """
        Get top-K predictions
        
        Args:
            y_score: Predicted scores
            k: Top-K value
            
        Returns:
            Top-K predictions
        """
        n_samples, n_labels = y_score.shape
        topk_predictions = np.zeros((n_samples, n_labels), dtype=int)
        
        for i in range(n_samples):
            topk_indices = np.argsort(y_score[i])[-k:][::-1]
            topk_predictions[i, topk_indices] = 1
        
        return topk_predictions
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def _update_history(self, metrics: Dict[str, float]):
        """Update metric history"""
        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
            self.metric_history[name].append(value)
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get metric history"""
        return self.metric_history.get(metric_name, [])
    
    def reset_history(self):
        """Reset history"""
        self.metric_history = {}


def compute_mAP(y_true: Union[np.ndarray, torch.Tensor],
               y_score: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute mean average precision
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        
    Returns:
        Mean average precision
    """
    calculator = MultiLabelMetrics()
    return calculator.compute_map(
        calculator._to_numpy(y_true),
        calculator._to_numpy(y_score)
    )


def compute_CF1(y_true: Union[np.ndarray, torch.Tensor],
               y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute class-wise F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Class-wise F1 score
    """
    calculator = MultiLabelMetrics()
    return calculator.compute_cf1(
        calculator._to_numpy(y_true),
        calculator._to_numpy(y_pred)
    )


def compute_OF1(y_true: Union[np.ndarray, torch.Tensor],
               y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute instance-wise F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Instance-wise F1 score
    """
    calculator = MultiLabelMetrics()
    return calculator.compute_of1(
        calculator._to_numpy(y_true),
        calculator._to_numpy(y_pred)
    )


def compute_all_metrics(y_true: Union[np.ndarray, torch.Tensor],
                       y_pred: Union[np.ndarray, torch.Tensor],
                       y_score: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
    """
    Compute all metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Predicted scores
        
    Returns:
        Metrics dictionary
    """
    calculator = MultiLabelMetrics()
    return calculator.compute_all(
        calculator._to_numpy(y_true),
        calculator._to_numpy(y_pred),
        calculator._to_numpy(y_score) if y_score is not None else None
    )


class AveragePrecision:
    """Average precision metric"""
    
    def __init__(self, average: str = 'macro'):
        """
        Initialize average precision metric
        
        Args:
            average: Average method
        """
        self.average = average
    
    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Compute average precision
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            
        Returns:
            Average precision
        """
        return average_precision_score(y_true, y_score, average=self.average)


class F1Score:
    """F1 score metric"""
    
    def __init__(self, average: str = 'macro'):
        """
        Initialize F1 score metric
        
        Args:
            average: Average method
        """
        self.average = average
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute F1 score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F1 score
        """
        return f1_score(y_true, y_pred, average=self.average, zero_division=0)


class PrecisionAtK:
    """Precision at K metric"""
    
    def __init__(self, k: int = 3):
        """
        Initialize precision at K metric
        
        Args:
            k: Top-K value
        """
        self.k = k
    
    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Compute precision at K
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            
        Returns:
            Precision at K
        """
        calculator = MultiLabelMetrics(top_k=self.k)
        y_pred_topk = calculator.get_topk_predictions(y_score, self.k)
        return calculator.compute_precision_at_k(y_true, y_pred_topk)
