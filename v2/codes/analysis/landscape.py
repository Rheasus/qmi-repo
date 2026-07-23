"""Loss-landscape probes: 1D minima interpolation and 2D filter-normalized
surfaces (Li et al. 2018), computed on fixed seeded data subsets.

Outputs JSON (curves/grids); figures are rendered locally for the paper.
"""

import json
from pathlib import Path

import torch


@torch.no_grad()
def _loss_on(model, batches, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        total += float(criterion(model(x), y)) * len(y)
        n += len(y)
    return total / n


def _state_vec(sd):
    return {k: v.clone().float() for k, v in sd.items()}


def _load_mix(model, a, b, alpha):
    mixed = {k: (1 - alpha) * a[k] + alpha * b[k] for k in a}
    model.load_state_dict(mixed)


def interpolate_1d(model, sd_a, sd_b, batches, criterion, device,
                   alphas=None):
    """Loss along the line between two trained solutions (same architecture)."""
    if alphas is None:
        alphas = [i / 40 * 1.5 - 0.25 for i in range(41)]  # [-0.25, 1.25]
    a, b = _state_vec(sd_a), _state_vec(sd_b)
    curve = []
    for al in alphas:
        _load_mix(model, a, b, al)
        curve.append({"alpha": al,
                      "loss": _loss_on(model, batches, criterion, device)})
    model.load_state_dict(sd_a)
    return curve


def _filter_normalized_direction(sd, generator):
    """Random direction with per-filter norm matched to the weights
    (Li et al. 2018); BN/bias/1D tensors get zero direction."""
    d = {}
    for k, w in sd.items():
        w = w.float()
        if w.dim() <= 1:
            d[k] = torch.zeros_like(w)
            continue
        r = torch.randn(w.shape, generator=generator, device=w.device)
        r = r.view(r.shape[0], -1)
        wf = w.view(w.shape[0], -1)
        r = r * (wf.norm(dim=1, keepdim=True) / (r.norm(dim=1, keepdim=True) + 1e-10))
        d[k] = r.view_as(w)
    return d


def surface_2d(model, sd_center, batches, criterion, device,
               span=1.0, steps=21, seed=0):
    """Loss grid over a filter-normalized random 2D slice around a solution."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    center = _state_vec(sd_center)
    cpu_center = {k: v.cpu() for k, v in center.items()}
    d1 = _filter_normalized_direction(cpu_center, gen)
    d2 = _filter_normalized_direction(cpu_center, gen)
    coords = [(-span + 2 * span * i / (steps - 1)) for i in range(steps)]
    grid = []
    for ax in coords:
        row = []
        for ay in coords:
            mixed = {k: center[k] + ax * d1[k].to(device) + ay * d2[k].to(device)
                     for k in center}
            model.load_state_dict(mixed)
            row.append(_loss_on(model, batches, criterion, device))
        grid.append(row)
    model.load_state_dict(sd_center)
    return {"coords": coords, "loss_grid": grid, "span": span, "seed": seed}


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload))
