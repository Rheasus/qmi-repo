"""Approximate Hessian curvature analysis at trained checkpoints.

For a trained checkpoint, computes on one data subset:
  * dominant-direction Rayleigh estimates (power iteration with deflation,
    HVP via double backprop),
  * Hessian trace (Hutchinson estimator, Rademacher probes),
  * a first-to-k returned-estimate ratio.

Power iteration targets large-magnitude directions; the returned values are
not guaranteed to be positive or algebraically ordered. They do not certify
that the checkpoint is a local minimum or provide a Hessian condition number.

BatchNorm/Dropout are in eval mode, so the loss is deterministic in the
parameters (standard practice for landscape/curvature analysis).
"""

import json
from pathlib import Path

import torch


def _flat_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _hvp_fn(model, loss, params):
    """Returns v -> Hv with the graph kept alive across calls."""
    grads = torch.autograd.grad(loss, params, create_graph=True)

    def hvp(vecs):
        dot = sum((g * v).sum() for g, v in zip(grads, vecs))
        return torch.autograd.grad(dot, params, retain_graph=True)
    return hvp


def _rand_like_params(params, generator):
    return [torch.randn(p.shape, generator=generator, device=p.device)
            for p in params]


def _dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def _scale(vecs, s):
    return [v * s for v in vecs]


def _norm(vecs):
    return _dot(vecs, vecs) ** 0.5


def top_eigenvalues(model, loss, k=5, iters=25, seed=0):
    """Backward-compatible name: return dominant-direction Rayleigh estimates."""
    params = _flat_params(model)
    hvp = _hvp_fn(model, loss, params)
    gen = torch.Generator(device=params[0].device).manual_seed(seed)

    eigs, eigvecs = [], []
    for _ in range(k):
        v = _rand_like_params(params, gen)
        v = _scale(v, 1.0 / _norm(v))
        lam = 0.0
        for _ in range(iters):
            hv = [h.detach() for h in hvp(v)]
            # deflate previously found directions
            for mu, u in zip(eigs, eigvecs):
                proj = _dot(hv, u)
                hv = [x - proj * y for x, y in zip(hv, u)]
            lam = _dot(v, hv)
            n = _norm(hv)
            if n < 1e-12:
                break
            v = _scale(hv, 1.0 / n)
        eigs.append(lam)
        eigvecs.append(v)
    return eigs


def hutchinson_trace(model, loss, probes=30, seed=0):
    params = _flat_params(model)
    hvp = _hvp_fn(model, loss, params)
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    est = 0.0
    for _ in range(probes):
        v = [(torch.randint(0, 2, p.shape, generator=gen, device=p.device,
                            dtype=p.dtype) * 2 - 1) for p in params]
        hv = hvp(v)
        est += _dot(v, hv)
    return est / probes


def analyze_checkpoint(model, data_batch, criterion, k=5, probes=30):
    """model: loaded+eval-mode; data_batch: (inputs, targets) on device."""
    model.eval()
    inputs, targets = data_batch
    loss = criterion(model(inputs), targets)
    eigs = top_eigenvalues(model, loss, k=k)
    # rebuild graph for trace (power iteration consumed retain_graph budget safely,
    # but a fresh loss keeps memory bounded)
    loss2 = criterion(model(inputs), targets)
    trace = hutchinson_trace(model, loss2, probes=probes)
    n_params = sum(p.numel() for p in _flat_params(model))
    return {
        "dominant_rayleigh_estimates": eigs,
        "dominant_rayleigh_first": eigs[0],
        # Legacy keys retained so existing artifacts/table scripts still load.
        "top_eigenvalues": eigs,
        "lambda_max": eigs[0],
        "spectral_ratio_1_to_k": eigs[0] / eigs[-1] if eigs[-1] != 0 else None,
        "trace": trace,
        "trace_per_param": trace / n_params,
        "n_params": n_params,
        "batch_size": int(targets.shape[0]),
    }


def write_result(run_dir: Path, payload: dict):
    (run_dir / "hessian.json").write_text(json.dumps(payload, indent=2))
