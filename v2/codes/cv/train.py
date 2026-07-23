"""CV training with checkpoint/resume, supporting grad / closure / population arms.

Protocol matches v1 (50 epochs, batch 32, CE loss, lr constant, no augmentation,
no early stopping) with the paper-stated 90/10 train/val split. Every epoch a
checkpoint is written; if the process dies (VM restart, etc.) the runner resumes
from the last completed epoch with RNG states restored.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, precision_score, recall_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from common.optimizers import build_optimizer, GRAD
from cv.models import build_model
from cv.data import load_cv_data


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    # ckpt is loaded with map_location=device, which moves RNG tensors to CUDA;
    # torch's RNG setters require CPU ByteTensors
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def _epoch_pass(model, loader, criterion, optimizer, mode, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if mode == GRAD:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:  # closure / population arms: forward-only evaluations
            def closure():
                with torch.no_grad():
                    return criterion(model(inputs), targets)
            loss = optimizer.step(closure)
            loss = torch.as_tensor(loss)
            with torch.no_grad():
                outputs = model(inputs)
        total_loss += float(loss.detach())
        pred = outputs.argmax(1)
        total += targets.size(0)
        correct += int((pred == targets).sum())
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device, detailed=False):
    model.eval()
    total_loss, n_batches = 0.0, 0
    ys, ps = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += float(criterion(outputs, targets))
        n_batches += 1
        ps.append(outputs.argmax(1).cpu())
        ys.append(targets.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    acc = float((y == p).mean())
    out = {"loss": total_loss / n_batches, "accuracy": acc}
    if detailed:
        out.update({
            "macro_f1": float(f1_score(y, p, average="macro")),
            "weighted_f1": float(f1_score(y, p, average="weighted")),
            "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y, p, average="macro", zero_division=0)),
        })
    return out


def run(spec: dict, results_dir: str, data_dir: str) -> dict:
    """Execute one CV run described by `spec`; resumable via epoch checkpoints.

    spec: {run_id, dataset, model, optimizer, seed, lr, epochs, batch_size,
           opt_kwargs{}, tag}
    """
    run_dir = Path(results_dir) / spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(spec["seed"])
    train_loader, val_loader, test_loader, num_classes, channels = load_cv_data(
        spec["dataset"], data_dir, spec["batch_size"], spec["seed"])

    model = build_model(spec["model"], num_classes, channels).to(device)
    opt_kwargs = dict(spec.get("opt_kwargs", {}))
    if spec["optimizer"] == "qpso":
        opt_kwargs.setdefault("total_steps", spec["epochs"] * len(train_loader))
        opt_kwargs.setdefault("seed", spec["seed"])
    optimizer, mode = build_optimizer(spec["optimizer"], model, spec["lr"], **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    start_epoch, history, elapsed = 0, [], 0.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer"])
        _load_rng_state(ckpt["rng"])
        start_epoch = ckpt["epoch"]
        history = ckpt["history"]
        elapsed = ckpt["elapsed"]
        print(f"[resume] {spec['run_id']} from epoch {start_epoch}", flush=True)

    collapsed_at = None
    for epoch in range(start_epoch, spec["epochs"]):
        t0 = time.time()
        train_loss, train_acc = _epoch_pass(model, train_loader, criterion,
                                            optimizer, mode, device)
        val = _evaluate(model, val_loader, criterion, device)
        elapsed += time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val["loss"],
                        "val_acc": 100.0 * val["accuracy"]})
        # collapse early-abort: 3 consecutive non-finite train losses means the
        # run is dead; stop burning budget, record it honestly as collapsed
        recent = [h["train_loss"] for h in history[-3:]]
        if len(recent) == 3 and all(not np.isfinite(l) for l in recent):
            collapsed_at = epoch + 1
            print(f"[{spec['run_id']}] collapsed (NaN loss 3 epochs) at epoch "
                  f"{collapsed_at}; aborting remaining epochs", flush=True)
            break
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
            "rng": _rng_state(), "epoch": epoch + 1,
            "history": history, "elapsed": elapsed,
        }, ckpt_path)
        print(f"[{spec['run_id']}] epoch {epoch+1}/{spec['epochs']} "
              f"train_acc={train_acc:.2f} val_acc={history[-1]['val_acc']:.2f}", flush=True)

    test = _evaluate(model, test_loader, criterion, device, detailed=True)
    result = {
        "run_id": spec["run_id"], "domain": "cv", "tag": spec.get("tag", ""),
        "dataset": spec["dataset"], "model": spec["model"],
        "optimizer": spec["optimizer"], "seed": spec["seed"],
        "lr": spec["lr"], "epochs": spec["epochs"], "batch_size": spec["batch_size"],
        "opt_kwargs": spec.get("opt_kwargs", {}),
        "test_accuracy": test["accuracy"], "test_loss": test["loss"],
        "macro_f1": test["macro_f1"], "weighted_f1": test["weighted_f1"],
        "macro_precision": test["macro_precision"], "macro_recall": test["macro_recall"],
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_acc": history[-1]["val_acc"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "collapsed": collapsed_at is not None,
        "collapsed_epoch": collapsed_at,
        "completed_epochs": len(history),
        "training_time_seconds": elapsed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history))
    # final model weights for Hessian / landscape analysis
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    ckpt_path.unlink(missing_ok=True)
    return result
