"""NLP training with checkpoint/resume, supporting grad / closure / population arms.

Protocol matches v1: batch 16, max_length 256, 3 epochs (transformers) /
8 epochs (LSTM), constant learning rate, no scheduler, no early stopping,
per-epoch curves on the evaluation set (no model selection — final-epoch model
is the reported model). Adds: epoch checkpointing/resume, NaN-collapse abort.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from common.optimizers import build_optimizer, GRAD
from nlp.models import build_model
from nlp.data import load_nlp_data


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


def _forward_loss(model, batch, is_transformer, criterion):
    if is_transformer:
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        return out.loss, out.logits
    logits = model(batch["input_ids"], batch["attention_mask"])
    return criterion(logits, batch["labels"]), logits


def _epoch_pass(model, loader, optimizer, mode, is_transformer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if mode == GRAD:
            optimizer.zero_grad()
            loss, logits = _forward_loss(model, batch, is_transformer, criterion)
            loss.backward()
            optimizer.step()
        else:  # closure / population: forward-only evaluations
            def closure():
                with torch.no_grad():
                    l, _ = _forward_loss(model, batch, is_transformer, criterion)
                    return l
            loss = optimizer.step(closure)
            loss = torch.as_tensor(loss)
            with torch.no_grad():
                _, logits = _forward_loss(model, batch, is_transformer, criterion)
        total_loss += float(torch.as_tensor(loss).detach())
        pred = logits.argmax(1)
        total += batch["labels"].size(0)
        correct += int((pred == batch["labels"]).sum())
    return total_loss / len(loader), correct / total


@torch.no_grad()
def _evaluate(model, loader, is_transformer, criterion, device, detailed=False):
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, logits = _forward_loss(model, batch, is_transformer, criterion)
        total_loss += float(loss)
        n += 1
        ps.append(logits.argmax(1).cpu())
        ys.append(batch["labels"].cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    out = {"loss": total_loss / n, "accuracy": float((y == p).mean()),
           "macro_f1": float(f1_score(y, p, average="macro"))}
    if detailed:
        out.update({
            "weighted_f1": float(f1_score(y, p, average="weighted")),
            "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y, p, average="macro", zero_division=0)),
        })
    return out


def run(spec: dict, results_dir: str, data_dir: str) -> dict:
    run_dir = Path(results_dir) / spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(spec["seed"])
    model, tokenizer = build_model(spec["model"], _num_classes(spec["dataset"]))
    model = model.to(device)
    is_transformer = spec["model"] in ("distilbert", "roberta")

    train_ds, test_ds, num_classes = load_nlp_data(
        spec["dataset"], tokenizer, data_dir,
        max_length=spec.get("max_length", 256),
        max_samples=spec.get("max_samples"))
    train_loader = DataLoader(train_ds, batch_size=spec["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=2, pin_memory=True)

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
        start_epoch, history, elapsed = ckpt["epoch"], ckpt["history"], ckpt["elapsed"]
        print(f"[resume] {spec['run_id']} from epoch {start_epoch}", flush=True)

    collapsed_at = None
    for epoch in range(start_epoch, spec["epochs"]):
        t0 = time.time()
        train_loss, train_acc = _epoch_pass(model, train_loader, optimizer, mode,
                                            is_transformer, criterion, device)
        ev = _evaluate(model, test_loader, is_transformer, criterion, device)
        elapsed += time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": ev["loss"],
                        "val_acc": ev["accuracy"], "val_f1": ev["macro_f1"]})
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
            "rng": _rng_state(), "epoch": epoch + 1,
            "history": history, "elapsed": elapsed,
        }, ckpt_path)
        print(f"[{spec['run_id']}] epoch {epoch+1}/{spec['epochs']} "
              f"train_acc={train_acc:.4f} eval_acc={ev['accuracy']:.4f}", flush=True)
        recent = [h["train_loss"] for h in history[-3:]]
        if len(recent) == 3 and all(not np.isfinite(l) for l in recent):
            collapsed_at = epoch + 1
            print(f"[{spec['run_id']}] collapsed (NaN loss 3 epochs); aborting",
                  flush=True)
            break

    test = _evaluate(model, test_loader, is_transformer, criterion, device, detailed=True)
    result = {
        "run_id": spec["run_id"], "domain": "nlp", "tag": spec.get("tag", ""),
        "dataset": spec["dataset"], "model": spec["model"],
        "optimizer": spec["optimizer"], "arm_label": spec.get("arm_label", spec["optimizer"]),
        "seed": spec["seed"], "lr": spec["lr"], "epochs": spec["epochs"],
        "batch_size": spec["batch_size"], "opt_kwargs": spec.get("opt_kwargs", {}),
        "test_accuracy": test["accuracy"], "test_loss": test["loss"],
        "macro_f1": test["macro_f1"], "weighted_f1": test["weighted_f1"],
        "macro_precision": test["macro_precision"], "macro_recall": test["macro_recall"],
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "best_eval_acc": max(h["val_acc"] for h in history),
        "collapsed": collapsed_at is not None, "collapsed_epoch": collapsed_at,
        "completed_epochs": len(history),
        "training_time_seconds": elapsed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history))
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    ckpt_path.unlink(missing_ok=True)
    return result


def _num_classes(dataset):
    return {"ag_news": 4, "imdb": 2, "sst2": 2}[dataset]
