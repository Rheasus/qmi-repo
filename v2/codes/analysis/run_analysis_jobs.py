#!/usr/bin/env python3
"""Run Hessian / landscape analyses over trained v2 checkpoints (on a GPU VM).

Usage:
  python analysis/run_analysis_jobs.py --results-dir ~/qmi-v2/results \
         --data-dir ~/qmi-v2/datasets --what hessian
  python analysis/run_analysis_jobs.py ... --what landscape

hessian:   every CV v2cv run of arms {adam, adamw, sgd, ngd, ctrl_qng_hp}
           (seeds 42/1337/2025) -> hessian.json next to result.json.
landscape: for each CV (dataset, model): 1D interpolation adam<->ngd and
           adam<->sgd (seed 42) + 2D surface around adam and ngd minima
           -> landscape_*.json in an analysis/ folder under results.
Idempotent: existing outputs are skipped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv.models import build_model            # noqa: E402
from cv.data import load_cv_data             # noqa: E402
from analysis.hessian import analyze_checkpoint, write_result   # noqa: E402
from analysis import landscape as lsc        # noqa: E402

CV_ARMS = ["adam", "adamw", "sgd", "ngd", "ctrl_qng_hp"]
CONFIGS = [("fashion_mnist", "SimpleCNN"), ("cifar10", "SimpleCNN"),
           ("cifar100", "SimpleCNN"), ("fashion_mnist", "ResNet18"),
           ("cifar10", "ResNet18"), ("cifar100", "ResNet18")]
SEEDS = [42, 1337, 2025]


def fixed_batch(dataset, data_dir, n=512, seed=1234):
    train_loader, _, _, num_classes, channels = load_cv_data(
        dataset, data_dir, batch_size=n, seed=seed, num_workers=0)
    x, y = next(iter(train_loader))
    return (x, y), num_classes, channels


def eval_batches(dataset, data_dir, n_total=2048, seed=1234):
    train_loader, _, _, _, _ = load_cv_data(
        dataset, data_dir, batch_size=256, seed=seed, num_workers=0)
    batches, got = [], 0
    for x, y in train_loader:
        batches.append((x, y))
        got += len(y)
        if got >= n_total:
            break
    return batches


def load_model(results, run_id, model_name, num_classes, channels, device):
    p = results / run_id / "final_model.pt"
    if not p.exists():
        return None
    m = build_model(model_name, num_classes, channels).to(device)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    return m


def do_hessian(results, data_dir, device):
    criterion = nn.CrossEntropyLoss()
    for dataset, model_name in CONFIGS:
        (x, y), num_classes, channels = fixed_batch(dataset, data_dir)
        x, y = x.to(device), y.to(device)
        for arm in CV_ARMS:
            for seed in SEEDS:
                run_id = f"{dataset}_{model_name}_{arm}_{seed}_v2cv"
                run_dir = results / run_id
                if (run_dir / "hessian.json").exists():
                    continue
                model = load_model(results, run_id, model_name,
                                   num_classes, channels, device)
                if model is None:
                    print(f"skip (no ckpt): {run_id}", flush=True)
                    continue
                t0 = time.time()
                try:
                    payload = analyze_checkpoint(model, (x, y), criterion)
                    payload.update({"run_id": run_id, "arm": arm,
                                    "dataset": dataset, "model": model_name,
                                    "seed": seed})
                    write_result(run_dir, payload)
                    print(f"hessian {run_id}: lmax={payload['lambda_max']:.2f} "
                          f"trace/n={payload['trace_per_param']:.3e} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                except RuntimeError as e:
                    print(f"FAILED {run_id}: {e}", flush=True)
                del model
                torch.cuda.empty_cache()


def do_landscape(results, data_dir, device):
    criterion = nn.CrossEntropyLoss()
    out_dir = results / "analysis"
    out_dir.mkdir(exist_ok=True)
    for dataset, model_name in CONFIGS:
        batches = eval_batches(dataset, data_dir)
        (_, _), num_classes, channels = fixed_batch(dataset, data_dir, n=8)
        sds = {}
        for arm in ["adam", "ngd", "sgd", "ctrl_qng_hp"]:
            p = results / f"{dataset}_{model_name}_{arm}_42_v2cv" / "final_model.pt"
            if p.exists():
                sds[arm] = torch.load(p, map_location=device, weights_only=True)
        if "adam" not in sds:
            continue
        model = build_model(model_name, num_classes, channels).to(device)
        for other in ["ngd", "sgd", "ctrl_qng_hp"]:
            out = out_dir / f"interp_{dataset}_{model_name}_adam_{other}.json"
            if other in sds and not out.exists():
                t0 = time.time()
                model.load_state_dict(sds["adam"])
                curve = lsc.interpolate_1d(model, sds["adam"], sds[other],
                                           batches, criterion, device)
                lsc.write_json(out, {"dataset": dataset, "model": model_name,
                                     "pair": ["adam", other], "curve": curve})
                print(f"interp {dataset}/{model_name} adam<->{other} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        for arm in ["adam", "ngd"]:
            out = out_dir / f"surface_{dataset}_{model_name}_{arm}.json"
            if arm in sds and not out.exists():
                t0 = time.time()
                model.load_state_dict(sds[arm])
                surf = lsc.surface_2d(model, sds[arm], batches, criterion, device)
                surf.update({"dataset": dataset, "model": model_name, "arm": arm})
                lsc.write_json(out, surf)
                print(f"surface {dataset}/{model_name}/{arm} "
                      f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--what", choices=["hessian", "landscape", "both"],
                    default="both")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = Path(args.results_dir).expanduser()
    data_dir = str(Path(args.data_dir).expanduser())
    if args.what in ("hessian", "both"):
        do_hessian(results, data_dir, device)
    if args.what in ("landscape", "both"):
        do_landscape(results, data_dir, device)
    print("analysis jobs complete", flush=True)
