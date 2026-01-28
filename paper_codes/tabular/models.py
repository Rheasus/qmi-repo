"""
Models for Tabular Data Benchmark
==================================

This module implements:
1. MLP: Multi-layer Perceptron with configurable architecture
2. XGBoost: Gradient Boosting with XGBoost
3. LightGBM: Gradient Boosting with LightGBM
4. CatBoost: Gradient Boosting with CatBoost

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import torch
import torch.nn as nn

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data.
    
    Architecture:
        - Input → 128 units, ReLU, Dropout(0.2)
        - 128 → 64 units, ReLU, Dropout(0.2)
        - 64 → output_dim
        - (Softmax for classification, none for regression)
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output units
        hidden_dims: List of hidden layer dimensions (default: [128, 64])
        dropout: Dropout probability (default: 0.2)
        task_type: 'classification' or 'regression'
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], dropout=0.2, task_type='classification'):
        super(MLP, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Hidden layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers)
        self.task_type = task_type
    
    def forward(self, x):
        return self.net(x)


def build_model(model_name, task_type='classification', input_dim=None, output_dim=None):
    """
    Build model based on name and task type.
    
    Args:
        model_name: One of 'mlp', 'xgboost', 'lightgbm', 'catboost'
        task_type: 'classification' or 'regression'
        input_dim: Number of input features (required for MLP)
        output_dim: Number of output units (required for MLP)
        
    Returns:
        Model instance
    """
    if model_name == 'mlp':
        if input_dim is None or output_dim is None:
            raise ValueError("input_dim and output_dim required for MLP")
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[128, 64],
            dropout=0.2,
            task_type=task_type
        )
    
    elif model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        if task_type == 'classification':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
    
    elif model_name == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        if task_type == 'classification':
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
    
    elif model_name == 'catboost':
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")
        
        if task_type == 'classification':
            return cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        else:
            return cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
