"""Hyperparameter-control arms (the v1 'QI-inspired' configurations).

These are the EXACT configurations that ran in v1 under the labels QNG / QPSO /
COBYLA (verified against the v1 run logs and result CSVs). In v2 they are kept
as *controls*: classical optimizers with alternative hyperparameters, used to
decompose "QI-inspired" performance differences into hyperparameter effects vs
algorithm effects. They are reported under honest ctrl_* names in the paper.

Provenance (v1 nlp/optimizers.py, confirmed by logged learning rates):
    'qng'    -> AdamW(lr*0.5, betas=(0.9,0.999), eps=1e-8, weight_decay=0.05)
    'qpso'   -> Adam(lr*1.5,  betas=(0.95,0.999), eps=1e-7)
    'cobyla' -> SGD(lr*0.1, momentum=0.0)
"""

from torch.optim import Adam, AdamW, SGD


def ctrl_qng_hp(params, lr):
    """v1 'QNG-inspired' arm: conservative AdamW (half lr, 5x weight decay)."""
    return AdamW(params, lr=lr * 0.5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)


def ctrl_qpso_hp(params, lr):
    """v1 'QPSO-inspired' arm: exploratory Adam (1.5x lr, beta1=0.95, eps=1e-7)."""
    return Adam(params, lr=lr * 1.5, betas=(0.95, 0.999), eps=1e-7)


def ctrl_cobyla_hp(params, lr):
    """v1 'COBYLA-inspired' arm: momentum-free SGD at 0.1x lr."""
    return SGD(params, lr=lr * 0.1, momentum=0.0)
