"""
Optimizers for NLP Benchmark
=============================

This module implements both classical and quantum-inspired optimizers:

Classical Optimizers:
    - Adam: Adaptive Moment Estimation
    - AdamW: Adam with weight decay decoupling
    - SGD: Stochastic Gradient Descent with Nesterov momentum
    - RMSprop: Root Mean Square Propagation
    - Adagrad: Adaptive Gradient Algorithm

Quantum-Inspired Optimizers:
    - SPSA: Simultaneous Perturbation Stochastic Approximation (gradient-free)
    - QNG: Quantum Natural Gradient (implemented as AdamW variant)
    - QPSO: Quantum Particle Swarm Optimization (implemented as Adam variant)
    - COBYLA: Constrained Optimization BY Linear Approximations (implemented as SGD variant)

Author: Gorkem Yilmaz
Institution: University of Sussex
Contact: gy74@sussex.ac.uk
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad


class SPSA(torch.optim.Optimizer):
    """
    SPSA: Simultaneous Perturbation Stochastic Approximation.
    
    A gradient-free optimizer that approximates gradients using finite differences
    with simultaneous perturbations. Inspired by quantum measurement principles.
    
    Reference:
        Spall, J. C. (1998). Implementation of the simultaneous perturbation algorithm 
        for stochastic optimization. IEEE Transactions on aerospace and electronic systems.
    
    Args:
        params: Model parameters
        lr: Learning rate (a parameter in SPSA notation)
        c: Perturbation size (default: 0.01)
        alpha: Learning rate decay exponent (default: 0.602)
        gamma: Perturbation decay exponent (default: 0.101)
    """
    
    def __init__(self, params, lr=1e-3, c=0.01, alpha=0.602, gamma=0.101):
        defaults = dict(lr=lr, c=c, alpha=alpha, gamma=gamma)
        super(SPSA, self).__init__(params, defaults)
        
        for group in self.param_groups:
            group['k'] = 0
    
    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                     Required for SPSA as it needs multiple forward passes.
        """
        if closure is None:
            raise RuntimeError('SPSA requires closure for function evaluation')
        
        loss = None
        
        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            alpha = group['alpha']
            gamma = group['gamma']
            k = group['k']
            
            k += 1
            group['k'] = k
            
            # Compute step sizes
            a_k = lr / (k ** alpha)
            c_k = c / (k ** gamma)
            
            # Generate random perturbation and save original parameters
            delta = []
            original_params = []
            params_list = []
            
            for p in group['params']:
                if p.requires_grad:
                    original_params.append(p.data.clone())
                    params_list.append(p)
                    delta_p = torch.randint_like(p, low=0, high=2, dtype=p.dtype) * 2 - 1
                    delta.append(delta_p)
            
            if len(delta) == 0:
                loss = closure()
                continue
            
            # Positive perturbation
            for p, d in zip(params_list, delta):
                p.data.add_(d, alpha=c_k)
            loss_plus = closure()
            
            # Negative perturbation
            for p, orig, d in zip(params_list, original_params, delta):
                p.data.copy_(orig)
                p.data.add_(d, alpha=-c_k)
            loss_minus = closure()
            
            # Approximate gradient
            grad_approx = (loss_plus - loss_minus) / (2 * c_k)
            
            # Apply update
            for p, orig, d in zip(params_list, original_params, delta):
                p.data.copy_(orig)
                p.data.add_(d * grad_approx, alpha=-a_k)
            
            loss = loss_plus
        
        return loss


def build_optimizer(optimizer_name, model_params, lr=1e-4):
    """
    Build optimizer with specified configuration.
    
    Args:
        optimizer_name: Name of optimizer
        model_params: Model parameters
        lr: Learning rate
        
    Returns:
        Optimizer instance
    """
    
    # Classical Baselines
    if optimizer_name == 'adam':
        return Adam(model_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    elif optimizer_name == 'adamw':
        return AdamW(model_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    elif optimizer_name == 'sgd':
        return SGD(model_params, lr=lr, momentum=0.9, nesterov=True)
    
    elif optimizer_name == 'rmsprop':
        return RMSprop(model_params, lr=lr, alpha=0.99, eps=1e-8)
    
    elif optimizer_name == 'adagrad':
        return Adagrad(model_params, lr=lr, eps=1e-10)
    
    # Quantum-Inspired Optimizers
    elif optimizer_name == 'spsa':
        return SPSA(model_params, lr=lr, c=0.01, alpha=0.602, gamma=0.101)
    
    elif optimizer_name == 'qng':
        # QNG: Quantum Natural Gradient
        # Approximated using AdamW with different weight decay
        return AdamW(model_params, lr=lr*0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    
    elif optimizer_name == 'qpso':
        # QPSO: Quantum Particle Swarm Optimization
        # Approximated using Adam with adjusted learning rate and beta parameters
        return Adam(model_params, lr=lr*1.5, betas=(0.95, 0.999), eps=1e-7)
    
    elif optimizer_name == 'cobyla':
        # COBYLA: Constrained Optimization BY Linear Approximations
        # Approximated using SGD without momentum
        return SGD(model_params, lr=lr*0.1, momentum=0.0)
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
