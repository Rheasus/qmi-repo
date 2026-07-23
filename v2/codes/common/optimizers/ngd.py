"""Diagonal empirical-Fisher Natural Gradient Descent (classical QNG analog).

Quantum Natural Gradient (Stokes et al. 2020) preconditions parameter updates
with the quantum geometric tensor (Fubini-Study metric). For classical neural
networks the corresponding object is the Fisher information matrix, and the
corresponding algorithm is Amari's natural gradient descent (Amari 1998).
Computing/inverting the full Fisher is infeasible at NN scale, so we implement
the standard scalable approximation (Martens 2014): a *diagonal empirical
Fisher* estimated as an EMA of squared minibatch gradients,

    F_t   = beta * F_{t-1} + (1 - beta) * g_t^2          (bias-corrected)
    u_t   = g_t / (F_t + damping)                        (preconditioned grad)
    m_t   = mu * m_{t-1} + u_t                           (heavy-ball momentum)
    theta = theta - lr * m_t                             (+ decoupled wd)

An optional per-step update-norm clip guards against the low-curvature blowup
that raw Fisher preconditioning exhibits early in training.

Relationship to adaptive methods (stated in the paper): dividing by F ~ E[g^2]
rather than sqrt(E[g^2]) is precisely what separates NGD from RMSprop/Adam;
this proximity is itself part of the paper's argument that practical QNG
analogs collapse toward the Adam family at NN scale.
"""

import torch
from torch.optim import Optimizer


class DiagNGD(Optimizer):
    def __init__(self, params, lr=1e-3, ema_decay=0.95, damping=1e-3,
                 momentum=0.9, weight_decay=0.0, max_update_norm=10.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"Invalid ema_decay: {ema_decay}")
        if damping <= 0.0:
            raise ValueError(f"Invalid damping: {damping}")
        defaults = dict(lr=lr, ema_decay=ema_decay, damping=damping,
                        momentum=momentum, weight_decay=weight_decay,
                        max_update_norm=max_update_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group["ema_decay"]
            damping = group["damping"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            lr = group["lr"]
            max_norm = group["max_update_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["fisher"] = torch.zeros_like(p)
                    state["momentum_buf"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                F = state["fisher"]
                F.mul_(beta).addcmul_(g, g, value=1.0 - beta)
                # bias-corrected Fisher estimate
                F_hat = F / (1.0 - beta ** t)

                u = g / (F_hat + damping)

                # trust-region style safety clip on the preconditioned gradient
                if max_norm is not None:
                    u_norm = u.norm()
                    if u_norm > max_norm:
                        u = u * (max_norm / (u_norm + 1e-12))

                buf = state["momentum_buf"]
                buf.mul_(mu).add_(u)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled weight decay (AdamW-style)
                p.add_(buf, alpha=-lr)

        return loss
