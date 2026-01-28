#!/usr/bin/env python3
"""
Natural Language Processing Benchmark - Main Training Script
============================================================

This script implements the NLP benchmark comparing quantum-inspired optimizers
(SPSA, QNG, QPSO, COBYLA) against classical baselines (Adam, AdamW, SGD, RMSprop, Adagrad)
on AG News, IMDB, and SST-2 datasets using DistilBERT, RoBERTa, and SimpleLSTM architectures.

Usage:
    python trainer.py --dataset ag_news --model distilbert --optimizer adam --seed 42 --epochs 3

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import os
import sys
import time
import json
import random
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import build_model, SimpleLSTM
from optimizers import build_optimizer
from utils import (
    set_seed,
    load_nlp_dataset,
    save_results,
    train_epoch,
    evaluate
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='NLP Benchmark - Quantum-Inspired Optimizers'
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ag_news', 'imdb', 'sst2'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['distilbert', 'roberta', 'lstm'],
                        help='Model architecture')
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 
                                'spsa', 'qng', 'qpso', 'cobyla'],
                        help='Optimizer to use')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (default: 256)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory for datasets (default: ./datasets)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for results (default: ./results)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create run identifier
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{args.dataset}_{args.model}_{args.optimizer}_{args.seed}_{timestamp}"
    
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
        # Build model and tokenizer
        if args.dataset == 'ag_news':
            num_classes = 4
        else:  # imdb, sst2
            num_classes = 2
        
        model, tokenizer = build_model(args.model, num_classes)
        model = model.to(device)
        is_transformer = args.model in ['distilbert', 'roberta']
        
        logger.info(f"✅ Model loaded: {args.model}")
        
        # Load dataset
        train_dataset, test_dataset, _ = load_nlp_dataset(
            args.dataset,
            tokenizer,
            args.data_dir,
            max_length=args.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        logger.info(f"✅ Dataset loaded: {args.dataset}")
        
        # Build optimizer
        optimizer = build_optimizer(
            args.optimizer,
            model.parameters(),
            lr=args.lr
        )
        
        logger.info(f"✅ Optimizer created: {args.optimizer} (lr={args.lr})")
        
        # Training loop
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        for epoch in range(1, args.epochs + 1):
            logger.info(f"📊 Epoch {epoch}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, is_transformer
            )
            
            # Validate
            val_loss, val_acc, val_f1, labels, preds, probs = evaluate(
                model, test_loader, device, is_transformer, num_classes
            )
            
            # Record history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            logger.info(f"   Train: Loss={train_loss:.4f} Acc={train_acc:.4f}")
            logger.info(f"   Val: Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
        
        # Final evaluation
        test_loss, test_acc, test_f1, test_labels, test_preds, test_probs = evaluate(
            model, test_loader, device, is_transformer, num_classes
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
                "max_length": args.max_length
            },
            "metrics": {
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "test_loss": test_loss,
                "training_time_seconds": training_time
            },
            "training_history": history
        }
        
        # Save results
        results_dir = Path(args.results_dir) / run_id
        save_results(
            results_dir, 
            results, 
            model, 
            test_labels, 
            test_preds, 
            test_probs, 
            num_classes
        )
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"   Test Accuracy: {test_acc:.4f}")
        logger.info(f"   Test F1: {test_f1:.4f}")
        logger.info(f"   Training Time: {training_time/60:.1f} minutes")
        logger.info(f"   Results saved to: {results_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
