#!/usr/bin/env python3
"""
Tabular Data Benchmark - Main Training Script
==============================================

This script implements the tabular data benchmark comparing quantum-inspired optimizers
against classical baselines on Breast Cancer and Wine Quality datasets using MLP,
XGBoost, LightGBM, and CatBoost models.

Usage:
    python trainer.py --dataset breast_cancer --model mlp --optimizer adam --seed 42

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

import numpy as np
import pandas as pd

from models import build_model
from optimizers import build_optimizer
from utils import (
    set_seed,
    load_tabular_dataset,
    train_mlp,
    evaluate_model,
    save_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Tabular Data Benchmark - Quantum-Inspired Optimizers'
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['breast_cancer', 'wine_quality'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['mlp', 'xgboost', 'lightgbm', 'catboost'],
                        help='Model to use')
    parser.add_argument('--optimizer', type=str, required=True,
                        help='Optimizer to use (for MLP: adam, sgd, spsa, qng; for tree models: default)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs for MLP (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for MLP (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for MLP (default: 32)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Directory for datasets')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for results')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create run identifier
    run_id = f"{args.dataset}_{args.model}_{args.optimizer}_{args.seed}"
    
    logger.info(f"🚀 Starting experiment: {run_id}")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Optimizer: {args.optimizer}")
    logger.info(f"   Seed: {args.seed}")
    
    start_time = time.time()
    
    try:
        # Load dataset
        X_train, X_val, X_test, y_train, y_val, y_test, task_type = load_tabular_dataset(
            args.dataset,
            args.seed
        )
        logger.info(f"✅ Dataset loaded: {args.dataset} ({task_type})")
        logger.info(f"   Train samples: {len(X_train)}")
        logger.info(f"   Val samples: {len(X_val)}")
        logger.info(f"   Test samples: {len(X_test)}")
        
        # Build model
        input_dim = X_train.shape[1]
        if task_type == 'classification':
            output_dim = len(np.unique(y_train))
        else:
            output_dim = 1
        
        model = build_model(
            args.model,
            task_type=task_type,
            input_dim=input_dim,
            output_dim=output_dim
        )
        logger.info(f"✅ Model created: {args.model}")
        
        # Train model
        if args.model == 'mlp':
            # Train MLP with specified optimizer
            history, test_metrics = train_mlp(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                optimizer_name=args.optimizer,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                task_type=task_type
            )
        else:
            # Train tree-based model
            model.fit(X_train, y_train)
            test_metrics = evaluate_model(model, X_test, y_test, task_type)
            history = None
        
        training_time = time.time() - start_time
        
        # Prepare results
        results = {
            "run_id": run_id,
            "dataset": args.dataset,
            "model": args.model,
            "optimizer": args.optimizer if args.model == 'mlp' else 'default',
            "seed": args.seed,
            "task_type": task_type,
            "hyperparameters": {
                "lr": args.lr if args.model == 'mlp' else None,
                "epochs": args.epochs if args.model == 'mlp' else None,
                "batch_size": args.batch_size if args.model == 'mlp' else None
            },
            "metrics": {
                **test_metrics,
                "training_time_seconds": training_time
            },
            "training_history": history
        }
        
        # Save results
        results_dir = Path(args.results_dir) / run_id
        save_results(results_dir, results, model)
        
        logger.info(f"🎉 Training completed!")
        logger.info(f"   Test Accuracy: {test_metrics.get('accuracy', 'N/A')}")
        logger.info(f"   Test F1: {test_metrics.get('f1_macro', 'N/A')}")
        if 'r2' in test_metrics:
            logger.info(f"   Test R2: {test_metrics.get('r2', 'N/A')}")
        logger.info(f"   Training Time: {training_time:.1f}s")
        logger.info(f"   Results saved to: {results_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
