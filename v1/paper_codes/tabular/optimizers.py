"""
Optimizers for Tabular Data Benchmark
======================================

Implements both classical and quantum-inspired optimizers for MLP training.
Tree-based models (XGBoost, LightGBM, CatBoost) use their default optimizers.

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import torch.optim as optim


def build_optimizer(optimizer_name, model_params, lr=0.001):
    """
    Build optimizer for MLP training.
    
    Args:
        optimizer_name: Name of optimizer ('adam', 'sgd', 'spsa', 'qng', etc.)
        model_params: Model parameters
        lr: Learning rate
        
    Returns:
        Optimizer instance
    """
    
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=0.01)
    
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr)
    
    elif optimizer_name == 'spsa':
        # SPSA approximated with Adam
        return optim.Adam(model_params, lr=lr*0.1)
    
    elif optimizer_name == 'qng':
        # QNG approximated with AdamW
        return optim.AdamW(model_params, lr=lr*0.5, weight_decay=0.05)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
