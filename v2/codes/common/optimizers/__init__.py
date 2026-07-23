"""Unified optimizer registry for the v2 benchmark.

Every arm in the revised paper is built through build_optimizer(), which
returns (optimizer, mode):

    mode == "grad"        standard loop: backward() then optimizer.step()
    mode == "closure"     gradient-free: optimizer.step(closure), no backward
    mode == "population"  population-based: optimizer.step(closure), no backward

Arms
----
Classical baselines:   sgd, adam, adamw, rmsprop, adagrad
Genuine QI-family:     spsa (Spall), ngd (diag-Fisher natural gradient,
                       classical QNG analog), qpso (Sun 2004)
Hyperparameter ctrls:  ctrl_qng_hp, ctrl_qpso_hp, ctrl_cobyla_hp
                       (the v1 'QI-inspired' configurations, kept as controls)

lr/wd-matched ablations are expressed through explicit kwargs on the classical
arms (e.g. name='adamw', lr=1e-5, weight_decay=0.05), driven by the run queue.
"""

from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad

from .spsa import SPSA
from .ngd import DiagNGD
from .qpso import QPSO
from .proxies import ctrl_qng_hp, ctrl_qpso_hp, ctrl_cobyla_hp

GRAD, CLOSURE, POPULATION = "grad", "closure", "population"

# name -> interaction mode (needed by the trainers)
MODES = {
    "sgd": GRAD, "adam": GRAD, "adamw": GRAD, "rmsprop": GRAD, "adagrad": GRAD,
    "ngd": GRAD,
    "spsa": CLOSURE,
    "qpso": POPULATION,
    "ctrl_qng_hp": GRAD, "ctrl_qpso_hp": GRAD, "ctrl_cobyla_hp": GRAD,
}


def build_optimizer(name, model, lr, **kw):
    """Build optimizer by arm name. `model` is the nn.Module (QPSO needs it);
    gradient-based arms receive model.parameters()."""
    name = name.lower()
    if name not in MODES:
        raise ValueError(f"Unknown optimizer arm: {name}. Known: {sorted(MODES)}")
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return SGD(params, lr=lr, momentum=kw.get("momentum", 0.9),
                   nesterov=kw.get("nesterov", False)), GRAD
    if name == "adam":
        return Adam(params, lr=lr, betas=kw.get("betas", (0.9, 0.999)),
                    eps=kw.get("eps", 1e-8)), GRAD
    if name == "adamw":
        return AdamW(params, lr=lr, betas=kw.get("betas", (0.9, 0.999)),
                     eps=kw.get("eps", 1e-8),
                     weight_decay=kw.get("weight_decay", 0.01)), GRAD
    if name == "rmsprop":
        return RMSprop(params, lr=lr, alpha=kw.get("alpha", 0.99),
                       eps=kw.get("eps", 1e-8)), GRAD
    if name == "adagrad":
        return Adagrad(params, lr=lr, eps=kw.get("eps", 1e-10)), GRAD

    if name == "ngd":
        return DiagNGD(params, lr=lr,
                       ema_decay=kw.get("ema_decay", 0.95),
                       damping=kw.get("damping", 1e-3),
                       momentum=kw.get("momentum", 0.9),
                       weight_decay=kw.get("weight_decay", 0.0),
                       max_update_norm=kw.get("max_update_norm", 10.0)), GRAD

    if name == "spsa":
        return SPSA(params, lr=lr, c=kw.get("c", 0.01),
                    alpha=kw.get("alpha", 0.602), gamma=kw.get("gamma", 0.101),
                    A=kw.get("A", 0.0)), CLOSURE

    if name == "qpso":
        return QPSO(model, n_particles=kw.get("n_particles", 10),
                    beta_start=kw.get("beta_start", 1.0),
                    beta_end=kw.get("beta_end", 0.5),
                    total_steps=kw["total_steps"],
                    init_spread=kw.get("init_spread", 0.01),
                    seed=kw.get("seed")), POPULATION

    if name == "ctrl_qng_hp":
        return ctrl_qng_hp(params, lr), GRAD
    if name == "ctrl_qpso_hp":
        return ctrl_qpso_hp(params, lr), GRAD
    if name == "ctrl_cobyla_hp":
        return ctrl_cobyla_hp(params, lr), GRAD
