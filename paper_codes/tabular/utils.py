"""
Utility Functions for Tabular Data Benchmark
=============================================

This module provides helper functions for:
- Setting random seeds
- Loading datasets
- Training MLP models
- Evaluation and metrics
- Saving results

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score
)

from torch.utils.data import DataLoader, TensorDataset

from optimizers import build_optimizer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tabular_dataset(dataset_name, seed=42, test_size=0.2, val_size=0.1):
    """
    Load and prepare tabular dataset.
    
    Args:
        dataset_name: 'breast_cancer' or 'wine_quality'
        seed: Random seed
        test_size: Test set size
        val_size: Validation set size
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, task_type
    """
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target
        task_type = 'classification'
    
    elif dataset_name == 'wine_quality':
        # Load wine quality dataset from OpenML
        data = fetch_openml('wine-quality-red', version=1, as_frame=False, parser='auto')
        X = data.data
        y = data.target.astype(int)
        task_type = 'classification'
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if task_type == 'classification' else None
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=seed,
        stratify=y_temp if task_type == 'classification' else None
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, task_type


def train_mlp(model, X_train, y_train, X_val, y_val, X_test, y_test,
              optimizer_name, lr, batch_size, epochs, task_type):
    """
    Train MLP model.
    
    Args:
        model: MLP model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        optimizer_name: Name of optimizer
        lr: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        task_type: 'classification' or 'regression'
        
    Returns:
        history, test_metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train) if task_type == 'classification' else torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val) if task_type == 'classification' else torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer and criterion
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)
    
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Training loop
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            if task_type == 'classification':
                loss = criterion(outputs, y_batch)
            else:
                loss = criterion(outputs.squeeze(), y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                
                if task_type == 'classification':
                    loss = criterion(outputs, y_batch)
                    _, preds = torch.max(outputs, 1)
                else:
                    loss = criterion(outputs.squeeze(), y_batch)
                    preds = outputs.squeeze()
                
                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        if task_type == 'classification':
            val_acc = accuracy_score(val_labels, val_preds)
        else:
            val_acc = r2_score(val_labels, val_preds)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, task_type)
    
    return history, test_metrics


def evaluate_model(model, X_test, y_test, task_type):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    if isinstance(model, torch.nn.Module):
        # MLP model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            
            if task_type == 'classification':
                _, y_pred = torch.max(outputs, 1)
                y_pred = y_pred.cpu().numpy()
            else:
                y_pred = outputs.squeeze().cpu().numpy()
    else:
        # Tree-based model
        y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0)
        }
    else:
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    return metrics


def save_results(results_dir, results, model):
    """
    Save experiment results.
    
    Args:
        results_dir: Directory to save results
        results: Dictionary of results
        model: Trained model
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run summary
    with open(results_dir / "run_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history if available
    if results.get('training_history'):
        history_df = pd.DataFrame(results['training_history'])
        history_df.to_csv(results_dir / 'history.csv', index=False)
    
    # Save model
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), results_dir / "model.pth")
    
    print(f"✅ Results saved to {results_dir}")
