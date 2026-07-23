"""
Utility Functions for NLP Benchmark
====================================

This module provides helper functions for:
- Setting random seeds
- Loading datasets
- Training and evaluation
- Saving results
- Creating visualizations

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
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score
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


class TextDataset(Dataset):
    """
    Custom dataset for text classification.
    
    Args:
        texts: List of text samples
        labels: List of labels
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_nlp_dataset(dataset_name, tokenizer, data_dir, max_length=256, max_samples=None):
    """
    Load and prepare NLP dataset.
    
    Args:
        dataset_name: One of 'ag_news', 'imdb', 'sst2'
        tokenizer: Tokenizer instance
        data_dir: Directory to cache datasets
        max_length: Maximum sequence length
        max_samples: Maximum number of samples (None for all)
        
    Returns:
        train_dataset, test_dataset, num_classes
    """
    cache_dir = str(Path(data_dir))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'ag_news':
        dataset = hf_load_dataset('ag_news', cache_dir=cache_dir)
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        num_classes = 4
    
    elif dataset_name == 'imdb':
        dataset = hf_load_dataset('imdb', cache_dir=cache_dir)
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        num_classes = 2
    
    elif dataset_name == 'sst2':
        dataset = hf_load_dataset('glue', 'sst2', cache_dir=cache_dir)
        train_texts = dataset['train']['sentence']
        train_labels = dataset['train']['label']
        test_texts = dataset['validation']['sentence']
        test_labels = dataset['validation']['label']
        num_classes = 2
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit samples if specified
    if max_samples:
        train_texts = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        test_texts = test_texts[:min(max_samples//5, len(test_texts))]
        test_labels = test_labels[:min(max_samples//5, len(test_labels))]
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
    
    return train_dataset, test_dataset, num_classes


def train_epoch(model, dataloader, optimizer, device, is_transformer=True):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        is_transformer: Whether model is a transformer
        
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Check if optimizer requires closure (e.g., SPSA)
    from optimizers import SPSA
    is_spsa = isinstance(optimizer, SPSA)
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if is_spsa:
            # SPSA requires closure
            def closure():
                if is_transformer:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    return outputs.loss
                else:
                    logits = model(input_ids, attention_mask)
                    return nn.CrossEntropyLoss()(logits, labels)
            
            loss = optimizer.step(closure)
            
            # Get logits for accuracy
            if is_transformer:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
            else:
                logits = model(input_ids, attention_mask)
        else:
            # Standard gradient-based optimizers
            if is_transformer:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device, is_transformer=True, num_classes=2):
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        dataloader: Test data loader
        device: Device to use (cuda/cpu)
        is_transformer: Whether model is a transformer
        num_classes: Number of classes
        
    Returns:
        avg_loss, accuracy, f1_score, labels, predictions, probabilities
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if is_transformer:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = 0.0
    
    return avg_loss, accuracy, f1, all_labels, all_preds, all_probs


def save_results(results_dir, results, model, y_true, y_pred, y_prob, num_classes):
    """
    Save all experiment results.
    
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
    
    # 3. Save model state dict
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
    
    print(f"✅ Results saved to {results_dir}")


def create_training_curves(history, save_path):
    """
    Create training and validation curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
    """
    epochs = history['epoch']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', marker='o')
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
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
