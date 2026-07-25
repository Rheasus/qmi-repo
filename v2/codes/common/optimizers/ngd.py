"""Diagonal second-moment NGD-style optimizer (classical QNG analog).

Quantum Natural Gradient (Stokes et al. 2020) preconditions parameter updates
with the quantum geometric tensor (Fubini-Study metric). For classical neural
networks the corresponding object is the Fisher information matrix, and the
corresponding algorithm is Amari's natural gradient descent (Amari 1998).
Computing/inverting the full Fisher is infeasible at NN scale, so this
configured analog uses an EMA of squared minibatch-mean gradients,

    F_t   = beta * F_{t-1} + (1 - beta) * g_t^2          (bias-corrected)
    u_t   = g_t / (F_t + damping)                        (preconditioned grad)
    m_t   = mu * m_{t-1} + u_t                           (heavy-ball momentum)
    theta = theta - lr * m_t                             (+ decoupled wd)

This is a diagonal second-moment / empirical-Fisher proxy, not a directly
computed empirical Fisher: square(mean per-example gradient) is not generally
equal to mean(square per-example gradient). An optional per-step update-norm
clip guards against early large preconditioned updates.

The configured arm also has optimizer-specific momentum, damping, clipping,
weight decay, and denominator scaling. Comparisons with AdamW therefore do
not isolate the causal contribution of the preconditioner.
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
                    # Historical key retained for checkpoint compatibility.
                    state["fisher"] = torch.zeros_like(p)
                    state["momentum_buf"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                F = state["fisher"]
                F.mul_(beta).addcmul_(g, g, value=1.0 - beta)
                # Bias-corrected squared-minibatch-gradient second moment.
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
