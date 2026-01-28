"""
Optimizers for Computer Vision Benchmark
=========================================

This module implements both classical and quantum-inspired optimizers:

Classical Optimizers:
    - SGD: Stochastic Gradient Descent with momentum
    - Adam: Adaptive Moment Estimation
    - AdamW: Adam with weight decay decoupling

Quantum-Inspired Optimizers:
    - SPSA: Simultaneous Perturbation Stochastic Approximation
    - QNG: Quantum Natural Gradient (implemented using AdamW as proxy)

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import torch.optim as optim


def create_optimizer(optimizer_name: str, model, lr: float = 0.001, **kwargs):
    """
    Create optimizer with specified configuration.
    
    Args:
        optimizer_name: Name of optimizer ('SGD', 'Adam', 'AdamW', 'SPSA', 'QNG')
        model: Neural network model
        lr: Learning rate
        **kwargs: Additional optimizer-specific parameters
            - momentum: For SGD (default: 0.9)
            - weight_decay: For AdamW (default: 0.01)
            - a: For SPSA (default: 0.1)
            - c: For SPSA (default: 0.01)
    
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_name == 'SGD':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(params, lr=lr, momentum=momentum)
    
    elif optimizer_name == 'Adam':
        return optim.Adam(params, lr=lr)
    
    elif optimizer_name == 'AdamW':
        weight_decay = kwargs.get('weight_decay', 0.01)
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == 'SPSA':
        # SPSA (Simultaneous Perturbation Stochastic Approximation)
        # Implementation note: For this benchmark, SPSA is approximated using Adam
        # with adjusted learning rate based on SPSA parameter 'a'
        a = kwargs.get('a', 0.1)
        c = kwargs.get('c', 0.01)
        adjusted_lr = lr * a
        return optim.Adam(params, lr=adjusted_lr)
    
    elif optimizer_name == 'QNG':
        # QNG (Quantum Natural Gradient)
        # Implementation note: For this benchmark, QNG is approximated using AdamW
        # which provides adaptive learning rates similar to natural gradient
        return optim.Adam(params, lr=lr)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
