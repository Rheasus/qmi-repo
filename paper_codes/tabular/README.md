# Tabular Data Benchmark - Quantum-Inspired Optimizers

This directory contains code for benchmarking quantum-inspired optimizers on tabular data tasks.

## Overview

### Optimizers
- **Classical:** Adam, AdamW, SGD, RMSprop
- **Quantum-Inspired:** SPSA, QNG

### Datasets
- **Breast Cancer:** Binary classification (569 samples, 30 features)
- **Wine Quality:** Multi-class classification (1599 samples, 11 features)

### Models
- **MLP:** Multi-layer Perceptron (128→64→output)
- **XGBoost:** Gradient Boosting with XGBoost
- **LightGBM:** Gradient Boosting with LightGBM
- **CatBoost:** Gradient Boosting with CatBoost

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python trainer.py \
    --dataset breast_cancer \
    --model mlp \
    --optimizer adam \
    --seed 42
```

### Tree-based Models

```bash
python trainer.py \
    --dataset wine_quality \
    --model xgboost \
    --optimizer default \
    --seed 42
```

## Command-Line Arguments

**Required:**
- `--dataset`: breast_cancer, wine_quality
- `--model`: mlp, xgboost, lightgbm, catboost
- `--optimizer`: For MLP (adam, sgd, spsa, qng); For trees (default)
- `--seed`: Random seed

**Optional:**
- `--epochs`: Epochs for MLP (default: 100)
- `--lr`: Learning rate for MLP (default: 0.001)
- `--batch_size`: Batch size for MLP (default: 32)

## Code Structure

```
tabular/
├── trainer.py        # Main training script
├── models.py         # Model implementations
├── optimizers.py     # Optimizer implementations
├── utils.py          # Utility functions
├── README.md
└── requirements.txt
```

## Author

**Gorkem Yilmaz**  
University of Sussex  
gy74@sussex.ac.uk
