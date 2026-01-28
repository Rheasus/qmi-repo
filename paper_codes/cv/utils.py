"""
Utility Functions for Computer Vision Benchmark
================================================

This module provides helper functions for:
- Setting random seeds
- Saving experiment results
- Creating visualizations
- Detailed model evaluation

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import os
import json
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        classification_report, 
        confusion_matrix,
        f1_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics will be unavailable.")


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_detailed(model, test_loader, criterion, device, num_classes):
    """
    Perform detailed evaluation with predictions and probabilities.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
        num_classes: Number of classes
        
    Returns:
        avg_loss, accuracy, predictions, true_labels, probabilities
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    return avg_loss, accuracy, all_predictions, all_targets, all_probabilities


def save_results(results_dir: Path, results: dict, model, y_true, y_pred, y_prob, num_classes):
    """
    Save all experiment results including metrics, visualizations, and model.
    
    Args:
        results_dir: Directory to save results
        results: Dictionary containing experiment results
        model: Trained model
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        num_classes: Number of classes
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save run summary
    with open(results_dir / "run_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 2. Save training history
    if 'training_history' in results:
        history_df = pd.DataFrame(results['training_history'])
        history_df.to_csv(results_dir / 'history.csv', index=False)
        
        # Create training curves
        create_training_curves(results['training_history'], results_dir / 'loss_curve.png')
    
    # 3. Save model
    torch.save(model.state_dict(), results_dir / "model.pth")
    
    # 4. Save detailed metrics if sklearn is available
    if SKLEARN_AVAILABLE:
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        with open(results_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.savetxt(results_dir / "confusion_matrix.csv", cm, delimiter=',', fmt='%d')
        with open(results_dir / "confusion_matrix.json", 'w') as f:
            json.dump(cm.tolist(), f)
        
        # F1 scores
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        with open(results_dir / "f1_scores.json", 'w') as f:
            json.dump({
                'macro_f1': f1_macro,
                'weighted_f1': f1_weighted
            }, f, indent=2)
    
    print(f"✅ Results saved to {results_dir}")


def create_training_curves(history, save_path):
    """
    Create training and validation loss/accuracy curves.
    
    Args:
        history: List of dictionaries containing training history
        save_path: Path to save the figure
    """
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', marker='o')
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, train_loss, 'b-', label='Train Loss', marker='o')
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', marker='s')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
