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
    
    parser = argparse.ArgumentParser(description='训练多尺度上下文感知深度核映射网络')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='nuswide',
                       choices=['nuswide', 'voc2007', 'coco'],
                       help='数据集名称 (默认: nuswide)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='数据根目录 (默认: ./data)')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet101',
                       choices=['resnet34', 'resnet50', 'resnet101', 'tresnet_l', 'cvt_w24'],
                       help='主干网络 (默认: resnet101)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='类别数 (默认: 根据数据集自动设置)')
    parser.add_argument('--grid_rows', type=int, default=8,
                       help='网格行数 (默认: 8)')
    parser.add_argument('--grid_cols', type=int, default=10,
                       help='网格列数 (默认: 10)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数 (默认: 200)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率 (默认: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='权重衰减 (默认: 0.0001)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='优化器 (默认: adamw)')
    
    # 实验设置
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称 (默认: 自动生成)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径 (可选)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径 (可选)')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--use_features', action='store_true', default=True,
                       help='使用预提取的特征 (默认: True)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='设备 (默认: 自动检测)')
    
    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='实验保存目录 (默认: ./experiments)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志目录 (默认: ./logs)')
    
    # 杂项
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数 (默认: 4)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='详细输出 (默认: True)')
    
    return parser.parse_args()


def load_config(args):
    """加载配置"""
    if args.config_file:
      
        print(f"从配置文件加载配置: {args.config_file}")
        config = ExperimentConfig.from_file(args.config_file)
    else:
      
        print("从命令行参数创建配置")
        
       
        dataset_classes = {
            'nuswide': 81,
            'voc2007': 20,
            'coco': 80
        }
        
        num_classes = args.num_classes if args.num_classes else dataset_classes.get(args.dataset, 81)
        
        # 创建网络配置
        network_config = NetworkConfig(
            backbone_name=args.backbone,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            num_classes=num_classes
        )
        
        # 创建数据集配置
        dataset_config = DatasetConfig(
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 创建训练配置
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
        
        # 创建实验配置
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
    
    print(f"实验目录创建在: {exp_root}")
    print(f"配置保存到: {config_path}")
    
    return dirs


def setup_model(config, device, experiment_dirs):
    print("创建模型...")
    
    # 选择模型创建方式
    use_custom_network = False
    
    if config.network.model_name == 'CustomNetwork' or use_custom_network:
        # 创建自定义网络（兼容原始代码）
        print("使用自定义网络架构")
        
        # 生成邻接矩阵
        from models.neighborhood import generate_adjacency_index_matrix
        connect_idx, weights_flag = generate_adjacency_index_matrix(
            config.network.grid_rows,
            config.network.grid_cols
        )
        
        # 创建模型
        model = create_custom_network(
            config=config,
            device=device
        )
    else:
        # 创建标准网络
        print(f"使用标准网络架构: {config.network.model_name}")
        model = create_model(
            config=config,
            device=device
        )
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数总数: {num_params:,}")
    print(f"可训练参数数: {num_trainable:,}")
    print(f"模型结构:")
    print(model)
    
    # 保存模型结构
    model_arch_path = os.path.join(experiment_dirs['configs'], 'model_architecture.txt')
    with open(model_arch_path, 'w') as f:
        f.write(str(model))
    
    return model


def resume_training(model, checkpoint_path, device, trainer):
    """恢复训练"""
    print(f"从检查点恢复训练: {checkpoint_path}")
    
    # 加载检查点
    checkpoint_manager = CheckpointManager(save_dir=os.path.dirname(checkpoint_path))
    checkpoint_info = checkpoint_manager.load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler
    )
    
    # 恢复训练状态
    if 'epoch' in checkpoint_info:
        print(f"恢复训练到 epoch {checkpoint_info['epoch']}")
    
    return checkpoint_info


def train_model(model, train_loader, val_loader, config, experiment_dirs, device, args):
    print("开始训练模型...")
    
    # 创建训练器
    trainer = create_trainer(model=model, config=config, device=device)
    
    # 恢复训练（如果指定）
    if args.resume:
        resume_training(model, args.resume, device, trainer)
    
    # 训练
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 保存训练历史
    history_path = os.path.join(experiment_dirs['results'], 'training_history.json')
    trainer.save_training_history(history_path)
    
    # 绘制训练指标
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
    print("评估模型...")
    
    if trainer is not None:
        evaluator = trainer.evaluator
    else:
        evaluator = create_evaluator(config)
    
    # 在测试集上评估
    test_results = evaluator.evaluate(
        model=model,
        dataloader=test_loader,
        mode='test'
    )
    
    # 保存测试结果
    test_results_path = os.path.join(experiment_dirs['results'], 'test_results.json')
    with open(test_results_path, 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {
            'metrics': test_results['metrics'],
            'num_samples': test_results['num_samples']
        }
        json.dump(results_to_save, f, indent=2)
    
    # 打印测试结果
    print("测试结果:")
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
    
    # 设置实验环境
    experiment_setup = setup_experiment(config, experiment_name)
    experiment_dirs = create_experiment_directories(config, experiment_name)
    
    # 设置设备
    device = torch.device(config.training.device)
    print(f"使用设备: {device}")
    
    
    print("创建数据加载器...")
    dataloaders = create_data_loaders(config, use_features=args.use_features)
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    test_loader = dataloaders.get('test')
    
    if train_loader is None:
        raise ValueError("训练数据加载器创建失败")
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    if val_loader:
        print(f"验证集: {len(val_loader.dataset)} 样本")
    if test_loader:
        print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    model = setup_model(config, device, experiment_dirs)
    
    if args.eval_only:
        # 仅评估模式
        print("仅评估模式")
        
        # 加载模型（如果需要）
        if args.resume:
            checkpoint_manager = CheckpointManager(save_dir=os.path.dirname(args.resume))
            checkpoint_info = checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path=args.resume,
                device=device
            )
            print(f"加载模型 epoch {checkpoint_info.get('epoch', 'unknown')}")
        
        # 评估模型
        test_results = evaluate_model(model, test_loader, None, experiment_dirs, device)
    else:
        # 训练模式
        # 训练模型
        training_results, trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            experiment_dirs=experiment_dirs,
            device=device,
            args=args
        )
        
        # 评估模型
        test_results = evaluate_model(model, test_loader, trainer, experiment_dirs, device)
        
        # 保存最终模型
        final_model_path = os.path.join(experiment_dirs['checkpoints'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict() if hasattr(config, 'to_dict') else config,
            'test_metrics': test_results['metrics']
        }, final_model_path)
        
        print(f"最终模型保存到: {final_model_path}")
    
    if 'test_results' in locals():
        print(f"  测试集大小: {test_results['num_samples']}")
        best_metric = None
        for metric_name in ['mAP', 'CF1', 'OF1']:
            if metric_name in test_results['metrics']:
                metric_value = test_results['metrics'][metric_name]
                print(f"  {metric_name}: {metric_value:.4f}")
                if metric_name == 'mAP' or best_metric is None:
                    best_metric = metric_value
        
        if best_metric is not None:
            print(f"  最佳指标: {best_metric:.4f}")
    
    print(f"  实验目录: {experiment_dirs['root']}")
    print("="*60)
    
    summary_path = os.path.join(experiment_dirs['results'], 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2
    print(f"详细信息请查看: {experiment_dirs['root']}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)