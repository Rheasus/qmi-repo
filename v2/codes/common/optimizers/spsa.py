"""Canonical SPSA optimizer (Spall 1992/1998).

Genuine gradient-free SPSA: two loss evaluations per update step, Rademacher
(+/-1) simultaneous perturbations, Spall gain schedules

    a_k = lr / (A + k)^alpha        (step size)
    c_k = c  / k^gamma              (perturbation size)

    g_hat_i = (L(theta + c_k*Delta) - L(theta - c_k*Delta)) / (2 c_k Delta_i)

For Delta_i in {+1,-1}, 1/Delta_i == Delta_i, so the update is a scalar times
the perturbation vector. No backpropagation is used at any point.

This is a faithful port of the v1 NLP implementation (the only direct
algorithm implementation in v1), with two fixes:
  * a single global perturbation vector spans ALL parameter groups (the v1
    version perturbed groups sequentially, which is only equivalent when the
    model has exactly one param group);
  * an optional Spall stability constant A (default 0 keeps v1 behaviour).

Note on BatchNorm: SPSA's two forward passes both update BN running statistics.
This is inherent to applying SPSA to BN networks and is documented in the paper.
"""

import torch
from torch.optim import Optimizer


class SPSA(Optimizer):
    def __init__(self, params, lr=1e-3, c=0.01, alpha=0.602, gamma=0.101, A=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if c <= 0.0:
            raise ValueError(f"Invalid perturbation size c: {c}")
        defaults = dict(lr=lr, c=c, alpha=alpha, gamma=gamma, A=A)
        super().__init__(params, defaults)
        self.state.setdefault("global", {})["k"] = 0

    def state_dict(self):
        sd = super().state_dict()
        sd["spsa_k"] = self.state["global"]["k"]
        return sd

    def load_state_dict(self, state_dict):
        k = state_dict.pop("spsa_k", 0)
        super().load_state_dict(state_dict)
        self.state.setdefault("global", {})["k"] = k

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("SPSA requires a closure that returns the loss "
                               "(forward pass only, no backward)")

        self.state["global"]["k"] += 1
        k = self.state["global"]["k"]

        # Per-group schedules; perturbation applied coherently across all groups.
        group_sched = []
        for group in self.param_groups:
            a_k = group["lr"] / ((group["A"] + k) ** group["alpha"])
            c_k = group["c"] / (k ** group["gamma"])
            group_sched.append((a_k, c_k))

        # Draw one Rademacher perturbation spanning every trainable parameter.
        deltas = []  # list of (param, delta, c_k, a_k)
        for group, (a_k, c_k) in zip(self.param_groups, group_sched):
            for p in group["params"]:
                if p.requires_grad:
                    delta = torch.randint_like(p, low=0, high=2, dtype=p.dtype) * 2 - 1
                    deltas.append((p, delta, c_k, a_k))

        if not deltas:
            return closure()

        # theta + c_k * Delta
        for p, delta, c_k, _ in deltas:
            p.add_(delta, alpha=c_k)
        loss_plus = closure()

        # theta - c_k * Delta  (subtract 2 c_k Delta from the perturbed value)
        for p, delta, c_k, _ in deltas:
            p.add_(delta, alpha=-2.0 * c_k)
        loss_minus = closure()

        # restore theta, then apply the SPSA update
        for p, delta, c_k, a_k in deltas:
            p.add_(delta, alpha=c_k)
            g_scale = (loss_plus - loss_minus) / (2.0 * c_k)
            p.add_(delta, alpha=-a_k * float(g_scale))

        return loss_plus
