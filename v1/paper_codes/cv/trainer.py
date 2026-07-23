#!/usr/bin/env python3
"""
Computer Vision Benchmark - Main Training Script
=================================================

This script implements the computer vision benchmark comparing quantum-inspired
optimizers (SPSA, QNG) against classical baselines (Adam, AdamW, SGD) on 
CIFAR-10, CIFAR-100, and Fashion-MNIST datasets using SimpleCNN and ResNet18 architectures.

Usage:
    python trainer.py --dataset cifar10 --model SimpleCNN --optimizer Adam --seed 42 --epochs 50

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from models import SimpleCNN, ResNet18
from optimizers import create_optimizer
from utils import (
    set_seed, 
    save_results, 
    evaluate_detailed,
    create_visualizations,
    log_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, data_dir: str, batch_size: int = 32):
    """
    Load and prepare dataset.
    
    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'fashion_mnist'
        data_dir: Directory to store/load datasets
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader, test_loader, num_classes, input_channels
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir / 'cifar10', 
            train=True, 
            download=True, 
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir / 'cifar10', 
            train=False, 
            download=True, 
            transform=transform
        )
        num_classes = 10
        input_channels = 3
        
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir / 'cifar100', 
            train=True, 
            download=True, 
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir / 'cifar100', 
            train=False, 
            download=True, 
            transform=transform
        )
        num_classes = 100
        input_channels = 3
        
    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir / 'fashion_mnist', 
            train=True, 
            download=True, 
            transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir / 'fashion_mnist', 
            train=False, 
            download=True, 
            transform=transform
        )
        num_classes = 10
        input_channels = 1
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader, num_classes, input_channels


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
        
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Computer Vision Benchmark - Quantum-Inspired Optimizers'
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'fashion_mnist'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['SimpleCNN', 'ResNet18'],
                        help='Model architecture')
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['SGD', 'Adam', 'AdamW', 'SPSA', 'QNG'],
                        help='Optimizer to use')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory for datasets (default: ./datasets)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for results (default: ./results)')
    
    # Optimizer-specific hyperparameters
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW weight decay (default: 0.01)')
    parser.add_argument('--spsa_a', type=float, default=0.1,
                        help='SPSA parameter a (default: 0.1)')
    parser.add_argument('--spsa_c', type=float, default=0.01,
                        help='SPSA parameter c (default: 0.01)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create run identifier
    run_id = f"{args.dataset}_{args.model}_{args.optimizer}_{args.seed}"
    
    logger.info(f"🚀 Starting experiment: {run_id}")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Optimizer: {args.optimizer}")
    logger.info(f"   Seed: {args.seed}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Learning Rate: {args.lr}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Device: {device}")
    
    start_time = time.time()
    
    try:
        # Load dataset
        train_loader, test_loader, num_classes, input_channels = load_dataset(
            args.dataset, 
            args.data_dir, 
            args.batch_size
        )
        logger.info(f"✅ Dataset loaded: {num_classes} classes, {input_channels} channels")
        
        # Create model
        if args.model == 'SimpleCNN':
            model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
        elif args.model == 'ResNet18':
            model = ResNet18(num_classes=num_classes, input_channels=input_channels)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        model = model.to(device)
        logger.info(f"✅ Model created: {args.model}")
        
        # Create optimizer
        optimizer_kwargs = {
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'a': args.spsa_a,
            'c': args.spsa_c
        }
        optimizer = create_optimizer(
            args.optimizer, 
            model, 
            args.lr, 
            **optimizer_kwargs
        )
        
        criterion = nn.CrossEntropyLoss()
        logger.info(f"✅ Optimizer created: {args.optimizer} (lr={args.lr})")
        
        # Training loop
        best_val_acc = 0.0
        training_history = []
        
        for epoch in range(args.epochs):
            logger.info(f"📊 Epoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Evaluate
            val_loss, val_acc = evaluate(
                model, test_loader, criterion, device
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Store training history
            training_history.append({
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            logger.info(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Final detailed evaluation
        logger.info("🔍 Performing detailed final evaluation...")
        test_loss, test_acc, y_pred, y_true, y_prob = evaluate_detailed(
            model, test_loader, criterion, device, num_classes
        )
        
        training_time = time.time() - start_time
        
        # Prepare results
        results = {
            "run_id": run_id,
            "dataset": args.dataset,
            "model": args.model,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "hyperparameters": {
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "momentum": args.momentum if args.optimizer == 'SGD' else None,
                "weight_decay": args.weight_decay if args.optimizer == 'AdamW' else None,
                "spsa_a": args.spsa_a if args.optimizer == 'SPSA' else None,
                "spsa_c": args.spsa_c if args.optimizer == 'SPSA' else None
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "best_val_accuracy": best_val_acc,
                "training_time_seconds": training_time
            },
            "training_history": training_history
        }
        
        # Save results
        results_dir = Path(args.results_dir) / run_id
        save_results(results_dir, results, model, y_true, y_pred, y_prob, num_classes)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"   Final Test Accuracy: {test_acc:.2f}%")
        logger.info(f"   Best Val Accuracy: {best_val_acc:.2f}%")
        logger.info(f"   Training Time: {training_time:.1f}s")
        logger.info(f"   Results saved to: {results_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
