#!/usr/bin/env python

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig, NetworkConfig, DatasetConfig, TrainingConfig
from models import create_model, create_custom_network
from training import create_trainer, create_evaluator
from utils import setup_experiment, create_data_loaders, CheckpointManager
from utils.visualization import plot_metrics


def parse_args():
    
    parser = argparse.ArgumentParser(description='Train a multi-scale context-aware deep kernel mapping network')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='nuswide',
                       choices=['nuswide', 'voc2007', 'coco'],
                       help='Dataset name (default: nuswide)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Dataset root directory (default: ./data)')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet101',
                       choices=['resnet34', 'resnet50', 'resnet101', 'tresnet_l', 'cvt_w24'],
                       help='Backbone network (default: resnet101)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (default: set automatically from the dataset)')
    parser.add_argument('--grid_rows', type=int, default=8,
                       help='Number of grid rows (default: 8)')
    parser.add_argument('--grid_cols', type=int, default=10,
                       help='Number of grid columns (default: 10)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay (default: 0.0001)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer (default: adamw)')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: generated automatically)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to the configuration file (optional)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint path for resuming training (optional)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Evaluation-only mode')
    parser.add_argument('--use_features', action='store_true', default=True,
                       help='Use pre-extracted features (default: True)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device (default: detected automatically)')
    
    # Saving and logging
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='Experiment save directory (default: ./experiments)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory (default: ./logs)')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output (default: True)')
    
    return parser.parse_args()


def load_config(args):
    """Load configuration."""
    if args.config_file:
      
        print(f"Loading configuration from file: {args.config_file}")
        config = ExperimentConfig.from_file(args.config_file)
    else:
      
        print("Creating configuration from command-line arguments")
        
       
        dataset_classes = {
            'nuswide': 81,
            'voc2007': 20,
            'coco': 80
        }
        
        num_classes = args.num_classes if args.num_classes else dataset_classes.get(args.dataset, 81)
        
        # Create network configuration
        network_config = NetworkConfig(
            backbone_name=args.backbone,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            num_classes=num_classes
        )
        
        # Create dataset configuration
        dataset_config = DatasetConfig(
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            device=args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
            seed=args.seed,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
        
        # Create experiment configuration
        config = ExperimentConfig(
            network=network_config,
            dataset=dataset_config,
            training=training_config
        )
    
    return config


def create_experiment_directories(config, experiment_name):
    exp_root = os.path.join(config.training.save_dir, experiment_name)

    dirs = {
        'root': exp_root,
        'checkpoints': os.path.join(exp_root, 'checkpoints'),
        'logs': os.path.join(exp_root, 'logs'),
        'visualizations': os.path.join(exp_root, 'visualizations'),
        'results': os.path.join(exp_root, 'results'),
        'configs': os.path.join(exp_root, 'configs')
    }
    

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
   
    config_path = os.path.join(dirs['configs'], 'config.yaml')
    config.save(config_path)
    
    args_path = os.path.join(dirs['configs'], 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Experiment directory created at: {exp_root}")
    print(f"Configuration saved to: {config_path}")
    
    return dirs


def setup_model(config, device, experiment_dirs):
    print("Creating model...")
    
    # Select model creation method
    use_custom_network = False
    
    if config.network.model_name == 'CustomNetwork' or use_custom_network:
        # Create custom network (compatible with the original code)
        print("Using custom network architecture")
        
        # Generate adjacency matrix
        from models.neighborhood import generate_adjacency_index_matrix
        connect_idx, weights_flag = generate_adjacency_index_matrix(
            config.network.grid_rows,
            config.network.grid_cols
        )
        
        # Create model
        model = create_custom_network(
            config=config,
            device=device
        )
    else:
        # Create standard network
        print(f"Using standard network architecture: {config.network.model_name}")
        model = create_model(
            config=config,
            device=device
        )
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total model parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print("Model architecture:")
    print(model)
    
    # Save model architecture
    model_arch_path = os.path.join(experiment_dirs['configs'], 'model_architecture.txt')
    with open(model_arch_path, 'w') as f:
        f.write(str(model))
    
    return model


def resume_training(model, checkpoint_path, device, trainer):
    """Resume training."""
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager(save_dir=os.path.dirname(checkpoint_path))
    checkpoint_info = checkpoint_manager.load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler
    )
    
    # Restore training state
    if 'epoch' in checkpoint_info:
        print(f"Resuming training from epoch {checkpoint_info['epoch']}")
    
    return checkpoint_info


def train_model(model, train_loader, val_loader, config, experiment_dirs, device, args):
    print("Starting model training...")
    
    # Create trainer
    trainer = create_trainer(model=model, config=config, device=device)
    
    # Resume training if specified
    if args.resume:
        resume_training(model, args.resume, device, trainer)
    
    # Train
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Save training history
    history_path = os.path.join(experiment_dirs['results'], 'training_history.json')
    trainer.save_training_history(history_path)
    
    # Plot training metrics
    if hasattr(trainer, 'train_losses') and hasattr(trainer.evaluator.performance_metrics, 'metrics'):
        metrics_history = {
            'train_loss': trainer.train_losses,
            **trainer.evaluator.performance_metrics.metrics
        }
        
        metrics_plot_path = os.path.join(experiment_dirs['visualizations'], 'training_metrics.png')
        plot_metrics(
            metrics_history=metrics_history,
            save_path=metrics_plot_path,
            title='Training Metrics'
        )
    
    return training_results, trainer


def evaluate_model(model, test_loader, trainer, experiment_dirs, device):
    print("Evaluating model...")
    
    if trainer is not None:
        evaluator = trainer.evaluator
    else:
        evaluator = create_evaluator(config)
    
    # Evaluate on the test set
    test_results = evaluator.evaluate(
        model=model,
        dataloader=test_loader,
        mode='test'
    )
    
    # Save test results
    test_results_path = os.path.join(experiment_dirs['results'], 'test_results.json')
    with open(test_results_path, 'w') as f:
        # Convert NumPy arrays to lists for JSON serialization
        results_to_save = {
            'metrics': test_results['metrics'],
            'num_samples': test_results['num_samples']
        }
        json.dump(results_to_save, f, indent=2)
    
    # Print test results
    print("Test results:")
    for metric_name, metric_value in test_results['metrics'].items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return test_results


def main():
   
    args = parse_args()
    
  
    config = load_config(args)
    

    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config.network.model_name}_{config.dataset.dataset_name}_{timestamp}"
    
    # Set up experiment environment
    experiment_setup = setup_experiment(config, experiment_name)
    experiment_dirs = create_experiment_directories(config, experiment_name)
    
    # Set device
    device = torch.device(config.training.device)
    print(f"Using device: {device}")
    
    
    print("Creating data loaders...")
    dataloaders = create_data_loaders(config, use_features=args.use_features)
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    test_loader = dataloaders.get('test')
    
    if train_loader is None:
        raise ValueError("Failed to create training data loader")
    
    print(f"Training set: {len(train_loader.dataset)} samples")
    if val_loader:
        print(f"Validation set: {len(val_loader.dataset)} samples")
    if test_loader:
        print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Create model
    model = setup_model(config, device, experiment_dirs)
    
    if args.eval_only:
        # Evaluation-only mode
        print("Evaluation-only mode")
        
        # Load model if needed
        if args.resume:
            checkpoint_manager = CheckpointManager(save_dir=os.path.dirname(args.resume))
            checkpoint_info = checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path=args.resume,
                device=device
            )
            print(f"Loaded model from epoch {checkpoint_info.get('epoch', 'unknown')}")
        
        # Evaluate model
        test_results = evaluate_model(model, test_loader, None, experiment_dirs, device)
    else:
        # Training mode
        # Train model
        training_results, trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            experiment_dirs=experiment_dirs,
            device=device,
            args=args
        )
        
        # Evaluate model
        test_results = evaluate_model(model, test_loader, trainer, experiment_dirs, device)
        
        # Save final model
        final_model_path = os.path.join(experiment_dirs['checkpoints'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict() if hasattr(config, 'to_dict') else config,
            'test_metrics': test_results['metrics']
        }, final_model_path)
        
        print(f"Final model saved to: {final_model_path}")
    
    if 'test_results' in locals():
        print(f"  Test set size: {test_results['num_samples']}")
        best_metric = None
        for metric_name in ['mAP', 'CF1', 'OF1']:
            if metric_name in test_results['metrics']:
                metric_value = test_results['metrics'][metric_name]
                print(f"  {metric_name}: {metric_value:.4f}")
                if metric_name == 'mAP' or best_metric is None:
                    best_metric = metric_value
        
        if best_metric is not None:
            print(f"  Best metric: {best_metric:.4f}")
    
    print(f"  Experiment directory: {experiment_dirs['root']}")
    print("="*60)
    
    summary_path = os.path.join(experiment_dirs['results'], 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"See {experiment_dirs['root']} for details")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
