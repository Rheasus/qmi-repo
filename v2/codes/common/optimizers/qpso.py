"""Canonical Quantum-behaved Particle Swarm Optimization (Sun et al. 2004).

Genuine population-based QPSO for neural-network training. Each particle is a
full flattened copy of the model parameters. Per optimization step, every
particle is evaluated (forward pass only) on the SAME minibatch, personal and
global bests are updated, and positions are resampled around per-particle
attractors using the quantum delta-potential-well rule:

    p_i   = phi * P_i + (1 - phi) * G,        phi ~ U(0,1)  (per dimension)
    C     = mean_i(P_i)                                    (mean best)
    X_i   = p_i +/- beta * |C - X_i| * ln(1/u), u ~ U(0,1) (per dimension)

with the contraction-expansion coefficient beta annealed linearly from
beta_start to beta_end over the training horizon.

Scale limits (documented in the paper): memory is n_particles * n_params and
compute is n_particles forward passes per step, which is why genuine QPSO is
evaluated on the small/medium models (MLP, SimpleCNN, LSTM) and is infeasible
for 100M+ parameter transformers -- that infeasibility is itself a finding.

Stochastic-objective caveat (documented): personal-best losses are compared
across minibatches, the standard practical compromise when applying swarm
methods to stochastic NN objectives.
"""

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class QPSO:
    """Population-based optimizer; not a torch.optim.Optimizer subclass.

    Usage per step:
        loss = qpso.step(closure)   # closure() -> loss on the CURRENT minibatch
    The model is left loaded with the global-best parameters after each step,
    so evaluation/checkpointing code can treat the model normally.
    """

    def __init__(self, model, n_particles=10, beta_start=1.0, beta_end=0.5,
                 total_steps=10000, init_spread=0.01, seed=None):
        if n_particles < 2:
            raise ValueError("QPSO needs at least 2 particles")
        self.model = model
        self.n = n_particles
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = max(1, total_steps)
        self._step = 0

        theta0 = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach()
        self.device = theta0.device
        self.dim = theta0.numel()
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)

        # Particles initialized around the model's init point.
        noise = torch.randn((self.n, self.dim), generator=gen) * init_spread
        self.X = theta0.unsqueeze(0).cpu() + noise           # positions (CPU master copy)
        self.X[0] = theta0.cpu()                             # particle 0 = exact init
        self.P = self.X.clone()                              # personal bests
        self.pbest_loss = torch.full((self.n,), float("inf"))
        self.G = self.X[0].clone()                           # global best
        self.gbest_loss = float("inf")

    def _beta(self):
        frac = min(1.0, self._step / self.total_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * frac

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @torch.no_grad()
    def step(self, closure):
        self._step += 1
        params = self._trainable_params()

        # 1) evaluate every particle on the current minibatch
        losses = torch.empty(self.n)
        for i in range(self.n):
            vector_to_parameters(self.X[i].to(self.device), params)
            losses[i] = float(closure())

        # 2) update personal / global bests
        improved = losses < self.pbest_loss
        self.pbest_loss[improved] = losses[improved]
        self.P[improved] = self.X[improved]
        best_i = int(torch.argmin(self.pbest_loss))
        if float(self.pbest_loss[best_i]) < self.gbest_loss:
            self.gbest_loss = float(self.pbest_loss[best_i])
            self.G = self.P[best_i].clone()

        # 3) resample positions around attractors (quantum delta-potential well)
        beta = self._beta()
        C = self.P.mean(dim=0)                               # mean best
        phi = torch.rand((self.n, self.dim))
        attractor = phi * self.P + (1.0 - phi) * self.G.unsqueeze(0)
        u = torch.rand((self.n, self.dim)).clamp_min(1e-12)
        sign = torch.where(torch.rand((self.n, self.dim)) < 0.5, -1.0, 1.0)
        self.X = attractor + sign * beta * (C.unsqueeze(0) - self.X).abs() * torch.log(1.0 / u)

        # 4) leave the model holding the global best
        vector_to_parameters(self.G.to(self.device), params)
        return float(losses.min())

    # --- checkpointing -------------------------------------------------
    def state_dict(self):
        return {"step": self._step, "X": self.X, "P": self.P,
                "pbest_loss": self.pbest_loss, "G": self.G,
                "gbest_loss": self.gbest_loss,
                "config": {"n": self.n, "beta_start": self.beta_start,
                           "beta_end": self.beta_end, "total_steps": self.total_steps}}

    def load_state_dict(self, sd):
        self._step = sd["step"]
        # master copies live on CPU; a ckpt loaded with map_location=cuda would
        # otherwise mix devices with the CPU-side resampling randomness
        self.X = sd["X"].cpu(); self.P = sd["P"].cpu()
        self.pbest_loss = sd["pbest_loss"].cpu()
        self.G = sd["G"].cpu(); self.gbest_loss = sd["gbest_loss"]
        vector_to_parameters(self.G.to(self.device), self._trainable_params())
