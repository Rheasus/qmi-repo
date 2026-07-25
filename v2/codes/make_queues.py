#!/usr/bin/env python3
"""Generate experiment queues (JSONL) for the v2 revision.

Queues are ordered so that the scientifically decisive runs execute first.
Run ids are deterministic: <dataset>_<model>_<optimizer>_<seed>_<tag>.
"""

import json
from pathlib import Path

QUEUE_DIR = Path(__file__).resolve().parents[1] / "infra" / "queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1337, 2025]
EXTRA_SEEDS = [7, 123]


def rid(dataset, model, opt, seed, tag):
    return f"{dataset}_{model}_{opt}_{seed}_{tag}"


def write(name, specs):
    p = QUEUE_DIR / f"{name}.jsonl"
    with open(p, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    print(f"{p.name:<28} {len(specs):>4} runs")


# ---------------------------------------------------------------- CV rerun
# Clean, single-codebase rerun of the whole CV benchmark.
# Arms: 3 classical + real SPSA + real NGD (QNG analog) + the v1 hp-control.
def cv_rerun():
    specs = []
    tag = "v2cv"
    arms = [
        ("sgd", {"momentum": 0.9}),
        ("adam", {}),
        ("adamw", {"weight_decay": 0.01}),
        ("spsa", {"c": 0.01}),          # lr acts as Spall's 'a'
        ("ngd", {"damping": 1e-3}),
        ("ctrl_qng_hp", {}),
    ]
    # order: cheap configs first so early results validate the pipeline
    configs = [
        ("fashion_mnist", "SimpleCNN"), ("cifar10", "SimpleCNN"),
        ("cifar100", "SimpleCNN"), ("fashion_mnist", "ResNet18"),
        ("cifar10", "ResNet18"), ("cifar100", "ResNet18"),
    ]
    for dataset, model in configs:
        for opt, okw in arms:
            for seed in SEEDS:
                lr = 0.01 if opt == "spsa" else 0.001  # spsa a=0.01: from smoke grid (0.1 diverges)
                specs.append({
                    "run_id": rid(dataset, model, opt, seed, tag), "domain": "cv",
                    "tag": tag, "dataset": dataset, "model": model,
                    "optimizer": opt, "seed": seed, "lr": lr,
                    "epochs": 50, "batch_size": 32, "opt_kwargs": okw,
                })
    write("cv_rerun", specs)


# ------------------------------------------------- NLP lr/wd-matched controls
# 2x2 factorial decomposing the v1 'QNG beats AdamW on RoBERTa' finding into
# lr vs wd effects, entirely within the v2 codebase (so no dependence on the
# lost v1 run code):
#   (2e-5, 0.01) = v1 adamw baseline (replication)
#   (1e-5, 0.05) = v1 'QNG-inspired' arm (replication)
#   (1e-5, 0.01) = lr effect only
#   (2e-5, 0.05) = wd effect only
def nlp_controls():
    specs = []
    tag = "v2ctl"
    settings = [
        ("adamw_lr2e5_wd001", 2e-5, 0.01),
        ("adamw_lr1e5_wd005", 1e-5, 0.05),
        ("adamw_lr1e5_wd001", 1e-5, 0.01),
        ("adamw_lr2e5_wd005", 2e-5, 0.05),
    ]
    for dataset in ["sst2", "imdb", "ag_news"]:        # cheapest RoBERTa first
        for label, lr, wd in settings:
            for seed in SEEDS:
                specs.append({
                    "run_id": rid(dataset, "roberta", label, seed, tag),
                    "domain": "nlp", "tag": tag, "dataset": dataset,
                    "model": "roberta", "optimizer": "adamw", "seed": seed,
                    "lr": lr, "epochs": 3, "batch_size": 16,
                    "opt_kwargs": {"weight_decay": wd}, "arm_label": label,
                })
    write("nlp_controls", specs)


# ------------------------------------------ NLP direct-algorithm / analog arms
def nlp_real_qi():
    specs = []
    tag = "v2qi"
    # Configured NGD-style classical analog on transformers.
    for dataset in ["sst2", "imdb", "ag_news"]:
        for model, epochs in [("roberta", 3), ("distilbert", 3)]:
            for seed in SEEDS:
                specs.append({
                    "run_id": rid(dataset, model, "ngd", seed, tag),
                    "domain": "nlp", "tag": tag, "dataset": dataset,
                    "model": model, "optimizer": "ngd", "seed": seed,
                    "lr": 2e-5, "epochs": epochs, "batch_size": 16,
                    "opt_kwargs": {"damping": 1e-3},
                })
    # real QPSO on the small model (LSTM); transformers documented as infeasible
    for dataset in ["sst2", "imdb", "ag_news"]:
        for seed in SEEDS:
            specs.append({
                "run_id": rid(dataset, "lstm", "qpso", seed, tag),
                "domain": "nlp", "tag": tag, "dataset": dataset,
                "model": "lstm", "optimizer": "qpso", "seed": seed,
                "lr": 0.0, "epochs": 8, "batch_size": 16,
                "opt_kwargs": {"n_particles": 10},
            })
    # missing v1 seed for real SPSA (n=2 -> n=3)
    for dataset in ["sst2", "imdb", "ag_news"]:
        for model, epochs in [("distilbert", 3), ("roberta", 3), ("lstm", 8)]:
            specs.append({
                "run_id": rid(dataset, model, "spsa", 42, tag),
                "domain": "nlp", "tag": tag, "dataset": dataset,
                "model": model, "optimizer": "spsa", "seed": 42,
                "lr": 1e-3, "epochs": epochs, "batch_size": 16,
                "opt_kwargs": {"c": 0.01},
            })
    write("nlp_real_qi", specs)


# ------------------------------------------------------------- tabular MLP
# The genuine neural optimizer benchmark for the tabular domain.
def tabular_mlp():
    specs = []
    tag = "v2tab"
    arms = ["sgd", "adam", "adamw", "spsa", "ngd", "ctrl_qng_hp", "qpso"]
    for dataset in ["adult", "higgs", "california_housing"]:
        for opt in arms:
            for seed in SEEDS + EXTRA_SEEDS:
                lr = 0.01 if opt == "spsa" else 0.001
                specs.append({
                    "run_id": rid(dataset, "MLP", opt, seed, tag),
                    "domain": "tabular", "tag": tag, "dataset": dataset,
                    "model": "MLP", "optimizer": opt, "seed": seed, "lr": lr,
                    "epochs": 50, "batch_size": 32,
                    "opt_kwargs": {"n_particles": 10} if opt == "qpso" else {},
                })
    write("tabular_mlp", specs)


# ------------------------------------------- NLP extra seeds (headline arms)
def nlp_extra_seeds():
    specs = []
    tag = "v2seed"
    for dataset in ["sst2", "imdb", "ag_news"]:
        for opt, lr, okw in [("adamw", 2e-5, {"weight_decay": 0.01}),
                             ("adam", 2e-5, {}),
                             ("ctrl_qng_hp", 2e-5, {})]:
            for seed in EXTRA_SEEDS:
                specs.append({
                    "run_id": rid(dataset, "roberta", opt, seed, tag),
                    "domain": "nlp", "tag": tag, "dataset": dataset,
                    "model": "roberta", "optimizer": opt, "seed": seed,
                    "lr": lr, "epochs": 3, "batch_size": 16, "opt_kwargs": okw,
                })
    write("nlp_extra_seeds", specs)


# ------------------------------------------------ CV extra seeds (L package)
# Extends every CV arm from 3 to 5 seeds for stronger statistics.
def cv_extra_seeds():
    specs = []
    tag = "v2cvx"
    arms = [
        ("sgd", {"momentum": 0.9}), ("adam", {}),
        ("adamw", {"weight_decay": 0.01}), ("spsa", {"c": 0.01}),
        ("ngd", {"damping": 1e-3}), ("ctrl_qng_hp", {}),
    ]
    configs = [
        ("fashion_mnist", "SimpleCNN"), ("cifar10", "SimpleCNN"),
        ("cifar100", "SimpleCNN"), ("fashion_mnist", "ResNet18"),
        ("cifar10", "ResNet18"), ("cifar100", "ResNet18"),
    ]
    for dataset, model in configs:
        for opt, okw in arms:
            for seed in EXTRA_SEEDS:
                lr = 0.01 if opt == "spsa" else 0.001
                specs.append({
                    "run_id": rid(dataset, model, opt, seed, tag), "domain": "cv",
                    "tag": tag, "dataset": dataset, "model": model,
                    "optimizer": opt, "seed": seed, "lr": lr,
                    "epochs": 50, "batch_size": 32, "opt_kwargs": okw,
                })
    write("cv_extra_seeds", specs)


# ---------------------------------------------------- L top-ups & offloads
# Factorial cells to uniform n=5: the two pure-effect cells get seeds 7,123
# (baseline and v1-QNG cells reach n=5 by pooling with the v2seed runs).
def factorial_topup():
    cells = [("adamw_lr1e5_wd001", 1e-5, 0.01), ("adamw_lr2e5_wd005", 2e-5, 0.05)]
    for name, datasets in [("factorial_topup_sst2imdb", ["sst2", "imdb"]),
                           ("factorial_topup_agnews", ["ag_news"])]:
        specs = []
        for dataset in datasets:
            for label, lr, wd in cells:
                for seed in EXTRA_SEEDS:
                    specs.append({
                        "run_id": rid(dataset, "roberta", label, seed, "v2ctl"),
                        "domain": "nlp", "tag": "v2ctl", "dataset": dataset,
                        "model": "roberta", "optimizer": "adamw", "seed": seed,
                        "lr": lr, "epochs": 3, "batch_size": 16,
                        "opt_kwargs": {"weight_decay": wd}, "arm_label": label,
                    })
        write(name, specs)


def ngd_topup():
    for name, model in [("ngd_topup_roberta", "roberta"),
                        ("ngd_topup_distilbert", "distilbert")]:
        specs = []
        for dataset in ["sst2", "imdb", "ag_news"]:
            for seed in EXTRA_SEEDS:
                specs.append({
                    "run_id": rid(dataset, model, "ngd", seed, "v2qi"),
                    "domain": "nlp", "tag": "v2qi", "dataset": dataset,
                    "model": model, "optimizer": "ngd", "seed": seed,
                    "lr": 2e-5, "epochs": 3, "batch_size": 16,
                    "opt_kwargs": {"damping": 1e-3},
                })
        write(name, specs)


# The stochastic-minibatch QPSO adaptation measured at ~37x the per-run cost
# of Adam on LSTM (15.1h vs
# 0.4h on SST-2). ag_news QPSO (~38h/run) is dropped and reported as a
# measured-cost infeasibility; imdb seeds 1337/2025 are offloaded to qmi-d.
def qpso_offload():
    specs = []
    for seed in [1337, 2025]:
        specs.append({
            "run_id": rid("imdb", "lstm", "qpso", seed, "v2qi"),
            "domain": "nlp", "tag": "v2qi", "dataset": "imdb",
            "model": "lstm", "optimizer": "qpso", "seed": seed,
            "lr": 0.0, "epochs": 8, "batch_size": 16,
            "opt_kwargs": {"n_particles": 10},
        })
    write("qpso_offload", specs)


def nlp_real_qi_trimmed():
    """Replacement queue for qmi-c: original nlp_real_qi minus the dropped
    ag_news QPSO runs and the offloaded imdb QPSO seeds. run_ids unchanged,
    so completed runs are skipped on relaunch."""
    src = QUEUE_DIR / "nlp_real_qi.jsonl"
    specs = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    keep = []
    for s in specs:
        if s["optimizer"] == "qpso":
            if s["dataset"] == "ag_news":
                continue
            if s["dataset"] == "imdb" and s["seed"] in (1337, 2025):
                continue
        keep.append(s)
    write("nlp_real_qi_trimmed", keep)


# Within-v2 DistilBERT classical baseline: needed both for the NGD comparison
# table (v2-internal) and as Hessian counterparts for the curvature analysis.
def nlp_distilbert_baseline():
    specs = []
    for dataset, epochs in [("sst2", 3), ("imdb", 3), ("ag_news", 3)]:
        for seed in SEEDS:
            specs.append({
                "run_id": rid(dataset, "distilbert", "adamw", seed, "v2ctl"),
                "domain": "nlp", "tag": "v2ctl", "dataset": dataset,
                "model": "distilbert", "optimizer": "adamw", "seed": seed,
                "lr": 2e-5, "epochs": epochs, "batch_size": 16,
                "opt_kwargs": {"weight_decay": 0.01},
            })
    write("nlp_distilbert_baseline", specs)


def qpso_simplecnn():
    # Trimmed after measuring the QPSO adaptation at ~5.4h/run on SimpleCNN with
    # chance-level accuracy: fashion keeps n=5, cifar10 n=3, cifar100 dropped
    # (reported as measured-cost infeasibility alongside the LSTM numbers).
    specs = []
    tag = "v2cvq"
    for dataset, seeds in [("fashion_mnist", SEEDS + EXTRA_SEEDS),
                           ("cifar10", SEEDS)]:
        for seed in seeds:
            specs.append({
                "run_id": rid(dataset, "SimpleCNN", "qpso", seed, tag),
                "domain": "cv", "tag": tag, "dataset": dataset,
                "model": "SimpleCNN", "optimizer": "qpso", "seed": seed,
                "lr": 0.0, "epochs": 50, "batch_size": 32,
                "opt_kwargs": {"n_particles": 10},
            })
    write("qpso_simplecnn", specs)


if __name__ == "__main__":
    cv_rerun()
    nlp_controls()
    nlp_real_qi()
    tabular_mlp()
    nlp_extra_seeds()
    cv_extra_seeds()
    factorial_topup()
    ngd_topup()
    qpso_offload()
    nlp_real_qi_trimmed()
    qpso_simplecnn()
    nlp_distilbert_baseline()
