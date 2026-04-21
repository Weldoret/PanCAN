"""
Trainer module

Implements training and evaluation functionality for multi-label classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import os
import warnings
from datetime import datetime
import json

from .evaluator import Evaluator, PerformanceMetrics
from .metrics import MultiLabelMetrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logger import setup_logger


class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self,
                 patience: int = 10,
                 delta: float = 0.0,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs with no improvement before stopping
            delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing loss, 'max' for maximizing metric
            verbose: Whether to print messages
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = -1
        
        if mode == 'min':
            self.compare = lambda x, y: x < y - self.delta
            self.best_score = np.inf
        elif mode == 'max':
            self.compare = lambda x, y: x > y + self.delta
            self.best_score = -np.inf
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if early stopping should be triggered
        
        Args:
            score: Current score
            epoch: Current epoch
            
        Returns:
            True if early stopping should be triggered
        """
        if self.best_score is None or self.compare(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'EarlyStopping: Improved to {score:.6f}')
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement for {self.counter}/{self.patience} epochs')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stopping at epoch {epoch}')
            
            return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = -1


class LearningRateScheduler:
    """Learning rate scheduler"""
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 scheduler_type: str = 'step',
                 step_size: int = 30,
                 gamma: float = 0.1,
                 min_lr: float = 1e-6,
                 patience: int = 5,
                 factor: float = 0.5,
                 verbose: bool = True):
        """
        Initialize learning rate scheduler
        
        Args:
            optimizer: Optimizer
            scheduler_type: Scheduler type ('step', 'plateau', 'cosine')
            step_size: Step size for StepLR
            gamma: Decay factor
            min_lr: Minimum learning rate
            patience: Patience for ReduceLROnPlateau
            factor: Decay factor for ReduceLROnPlateau
            verbose: Whether to print messages
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.min_lr = min_lr
        self.verbose = verbose
        self.current_lr = self._get_current_lr()
        
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience,
                min_lr=min_lr, verbose=verbose
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=step_size, eta_min=min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """
        Step the scheduler
        
        Args:
            metric: Metric for ReduceLROnPlateau
        """
        old_lr = self.current_lr
        
        if self.scheduler_type == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        self.current_lr = self._get_current_lr()
        
        if self.verbose and abs(self.current_lr - old_lr) > 1e-9:
            print(f'Learning rate changed: {old_lr:.6f} -> {self.current_lr:.6f}')
    
    def _get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr
    
    def state_dict(self):
        """Get state dict"""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.scheduler.load_state_dict(state_dict)
        self.current_lr = self._get_current_lr()


class ModelCheckpoint:
    """Model checkpoint manager"""
    
    def __init__(self,
                 save_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_frequency: int = 1,
                 verbose: bool = True):
        """
        Initialize model checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' for minimizing, 'max' for maximizing
            save_best_only: Whether to save only the best model
            save_frequency: Save frequency in epochs
            verbose: Whether to print messages
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.verbose = verbose
        
        os.makedirs(save_dir, exist_ok=True)
        
        if mode == 'min':
            self.best_score = np.inf
            self.compare = lambda x, y: x < y
        else:
            self.best_score = -np.inf
            self.compare = lambda x, y: x > y
        
        self.best_epoch = -1
        self.checkpoint_history = []
    
    def save(self,
            epoch: int,
            model: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: Optional[LearningRateScheduler],
            score: float,
            config: Dict[str, Any],
            is_best: bool = False,
            additional_info: Optional[Dict[str, Any]] = None):
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            model: Model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            score: Current score
            config: Configuration
            is_best: Whether this is the best model
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        filename = f'checkpoint_epoch_{epoch:04d}.pth'
        filepath = os.path.join(self.save_dir, filename)
        
        torch.save(checkpoint, filepath)
        
        self.checkpoint_history.append({
            'epoch': epoch,
            'filename': filename,
            'score': score,
            'is_best': is_best
        })
        
        if self.verbose:
            print(f'Checkpoint saved: {filepath} (score: {score:.6f})')
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            if os.path.exists(best_path):
                os.remove(best_path)
            os.symlink(filename, best_path)
            
            if self.verbose:
                print(f'Best model updated: {best_path}')
    
    def check_and_save(self,
                      epoch: int,
                      model: nn.Module,
                      optimizer: optim.Optimizer,
                      scheduler: Optional[LearningRateScheduler],
                      metrics: Dict[str, float],
                      config: Dict[str, Any],
                      additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if checkpoint should be saved and save if needed
        
        Args:
            epoch: Current epoch
            model: Model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            metrics: Metrics dictionary
            config: Configuration
            additional_info: Additional information to save
            
        Returns:
            True if checkpoint was saved
        """
        should_save = False
        is_best = False
        
        if epoch % self.save_frequency == 0:
            should_save = True
        
        if self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            if self.compare(current_score, self.best_score):
                self.best_score = current_score
                self.best_epoch = epoch
                is_best = True
                
                if not self.save_best_only:
                    should_save = True
            elif self.save_best_only:
                should_save = False
        
        if should_save:
            self.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                score=current_score if self.monitor in metrics else 0.0,
                config=config,
                is_best=is_best,
                additional_info=additional_info
            )
            return True
        
        return False
    
    def load_best(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """
        Load best model
        
        Args:
            model: Model to load
            device: Device to load on
            
        Returns:
            Checkpoint information
        """
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Best model not found: {best_path}")
        
        actual_path = os.path.realpath(best_path)
        
        return load_checkpoint(model, actual_path, device)
    
    def load_latest(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """
        Load latest model
        
        Args:
            model: Model to load
            device: Device to load on
            
        Returns:
            Checkpoint information
        """
        if not self.checkpoint_history:
            checkpoint_files = [f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_epoch_')]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in {self.save_dir}")
            
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_file = checkpoint_files[-1]
        else:
            latest_file = self.checkpoint_history[-1]['filename']
        
        filepath = os.path.join(self.save_dir, latest_file)
        return load_checkpoint(model, filepath, device)


class GradientAccumulator:
    """Gradient accumulator"""
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.reset()
    
    def reset(self):
        """Reset accumulator"""
        self.step_counter = 0
    
    def step(self, optimizer: optim.Optimizer, model: nn.Module):
        """
        Perform gradient accumulation step
        
        Args:
            optimizer: Optimizer
            model: Model
            
        Returns:
            True if parameters were updated
        """
        self.step_counter += 1
        
        if self.step_counter % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            self.step_counter = 0
            return True
        
        return False
    
    def should_update(self) -> bool:
        """Check if parameters should be updated"""
        return self.step_counter % self.accumulation_steps == 0


class MixedPrecisionTrainer:
    """Mixed precision trainer"""
    
    def __init__(self, enabled: bool = True, device: str = 'cuda'):
        """
        Initialize mixed precision trainer
        
        Args:
            enabled: Whether to enable mixed precision
            device: Device
        """
        self.enabled = enabled and (device == 'cuda')
        self.scaler = amp.GradScaler(enabled=self.enabled)
    
    def backward(self, loss: torch.Tensor, optimizer: optim.Optimizer):
        """
        Perform backward pass with mixed precision
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
        """
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: optim.Optimizer):
        """
        Perform optimizer step with mixed precision
        
        Args:
            optimizer: Optimizer
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def state_dict(self):
        """Get state dict"""
        return self.scaler.state_dict() if self.enabled else {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict"""
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)


class TrainingManager:
    """Training manager"""
    
    def __init__(self,
                 model: nn.Module,
                 config,
                 device: Optional[torch.device] = None):
        """
        Initialize training manager
        
        Args:
            model: Model to train
            config: Configuration
            device: Device to use
        """
        self.model = model
        self.config = config
        self.device = device if device else torch.device(
            config.training.device if hasattr(config, 'training') else 'cuda'
        )
        
        self.model.to(self.device)
        
        self.optimizer = self._create_optimizer()
        
        self.criterion = self._create_criterion()
        
        self.scheduler = self._create_scheduler()
        
        self.evaluator = Evaluator(config)
        
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            delta=config.training.early_stopping_delta,
            mode='min' if 'loss' in config.training.monitor else 'max',
            verbose=True
        ) if config.training.use_early_stopping else None
        
        self.checkpoint = ModelCheckpoint(
            save_dir=config.training.save_dir,
            monitor=config.training.monitor,
            mode='min' if 'loss' in config.training.monitor else 'max',
            save_best_only=config.training.save_best_only,
            save_frequency=config.training.save_frequency,
            verbose=True
        )
        
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=config.training.accumulation_steps
        )
        
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=config.training.use_amp,
            device=str(self.device)
        )
        
        self.current_epoch = 0
        self.train_losses = []
        self.val_metrics = {}
        self.best_score = None
        self.best_epoch = -1
        
        self.logger = setup_logger(
            name='trainer',
            log_dir=config.training.log_dir,
            level='INFO'
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_type = self.config.training.optimizer.lower()
        
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_type == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = getattr(self.config.training, 'momentum', 0.9)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        loss_type = self.config.training.loss_function.lower()
        
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            from .losses import FocalLoss
            alpha = getattr(self.config.training, 'focal_alpha', 0.25)
            gamma = getattr(self.config.training, 'focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'asymmetric':
            from .losses import AsymmetricLoss
            gamma_neg = getattr(self.config.training, 'asymmetric_gamma_neg', 4.0)
            gamma_pos = getattr(self.config.training, 'asymmetric_gamma_pos', 1.0)
            return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        else:
            warnings.warn(f"Unknown loss function: {loss_type}, using BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss()
    
    def _create_scheduler(self) -> Optional[LearningRateScheduler]:
        """Create learning rate scheduler"""
        if not hasattr(self.config.training, 'scheduler'):
            return None
        
        return LearningRateScheduler(
            optimizer=self.optimizer,
            scheduler_type=self.config.training.scheduler,
            step_size=getattr(self.config.training, 'step_size', 30),
            gamma=getattr(self.config.training, 'gamma', 0.1),
            min_lr=getattr(self.config.training, 'min_lr', 1e-6),
            patience=getattr(self.config.training, 'scheduler_patience', 5),
            factor=getattr(self.config.training, 'scheduler_factor', 0.5),
            verbose=True
        )
    
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics dictionary
        """
        self.model.train()
        
        epoch_loss = 0.0
        num_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch_data in enumerate(pbar):
            if isinstance(batch_data, (list, tuple)):
                inputs, labels = batch_data[0], batch_data[1]
            else:
                inputs, labels = batch_data['image'], batch_data['label']
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with amp.autocast(enabled=self.mixed_precision.enabled):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.mixed_precision.backward(loss / self.gradient_accumulator.accumulation_steps, self.optimizer)
            
            if self.gradient_accumulator.step(self.optimizer, self.model):
                self.mixed_precision.step(self.optimizer)
            
            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size
            
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        
        self.train_losses.append(avg_loss)
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {'train_loss': avg_loss}
    
    def validate(self,
                val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics dictionary
        """
        results = self.evaluator.evaluate(
            model=self.model,
            dataloader=val_loader,
            criterion=self.criterion,
            mode='val'
        )
        
        return results['metrics']
    
    def train(self,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            
        Returns:
            Training results dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            
            metrics = {**train_metrics, **val_metrics}
            
            self.logger.info(f"Epoch {self.current_epoch}/{num_epochs}: {metrics}")
            
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
            
            self.checkpoint.check_and_save(
                epoch=self.current_epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=metrics,
                config=config_dict,
                additional_info={
                    'train_losses': self.train_losses,
                    'val_metrics': self.evaluator.performance_metrics.metrics
                }
            )
            
            if self.early_stopping is not None:
                monitor_metric = metrics.get(self.config.training.monitor, 0.0)
                
                if self.early_stopping(monitor_metric, self.current_epoch):
                    self.logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                    break
        
        self.logger.info("Training completed")
        
        if self.checkpoint.checkpoint_history:
            self.logger.info("Loading best model")
            checkpoint_info = self.checkpoint.load_best(self.model, self.device)
            self.best_score = checkpoint_info.get('score')
            self.best_epoch = checkpoint_info.get('epoch', -1)
        
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_metrics': self.evaluator.performance_metrics.metrics,
            'checkpoint_history': self.checkpoint.checkpoint_history
        }
    
    def test(self,
            test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Test model
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing model")
        
        results = self.evaluator.evaluate(
            model=self.model,
            dataloader=test_loader,
            criterion=self.criterion,
            mode='test'
        )
        
        self.logger.info(f"Test results: {results['metrics']}")
        
        return results
    
    def save_training_history(self, filepath: str):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_metrics': self.evaluator.performance_metrics.metrics,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'checkpoint_history': self.checkpoint.checkpoint_history,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved to {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint information
        """
        return load_checkpoint(self.model, checkpoint_path, self.device)
    
    def get_lr(self) -> float:
        """
        Get current learning rate
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """
        Set learning rate
        
        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class MultiLabelTrainer(TrainingManager):
    """Multi-label trainer"""
    
    def __init__(self, model, config, device=None):
        """
        Initialize multi-label trainer
        
        Args:
            model: Model
            config: Configuration
            device: Device
        """
        super().__init__(model, config, device)
    
    def predict(self, dataloader, threshold=0.5, top_k=None):
        """
        Make predictions
        
        Args:
            dataloader: Data loader
            threshold: Threshold for binary classification
            top_k: Top-K value for predictions
            
        Returns:
            Predictions dictionary
        """
        self.model.eval()
        
        all_outputs = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Predicting"):
                if isinstance(batch_data, (list, tuple)):
                    inputs, labels = batch_data[0], batch_data[1]
                else:
                    inputs, labels = batch_data['image'], batch_data['label']
                
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)
                
                if top_k is not None:
                    _, top_indices = probs.topk(top_k, dim=1)
                    predictions = torch.zeros_like(probs).scatter_(1, top_indices, 1)
                else:
                    predictions = (probs > threshold).float()
                
                all_outputs.append(outputs.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0) if all_outputs else torch.tensor([])
        all_probs = torch.cat(all_probs, dim=0) if all_probs else torch.tensor([])
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        
        return {
            'outputs': all_outputs,
            'probs': all_probs,
            'predictions': predictions.cpu() if 'predictions' in locals() else None,
            'labels': all_labels
        }
