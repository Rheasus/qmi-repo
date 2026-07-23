"""Tabular MLP benchmark — the genuine neural optimizer study for tabular data.

v1 reported only GBDT results (where 'optimizer' labels were learning-rate
conditions); the revised paper separates those as a control study and adds
this MLP benchmark as the real tabular neural-optimization arm.

Protocol (matches the v1 tabular MLP protocol notes): 70/15/15 train/val/test,
stratified for classification; StandardScaler on numeric and
OneHotEncoder(handle_unknown='ignore') on categorical features, fit on train
only; MLP [128, 64] with dropout 0.2; batch 32; max 50 epochs with early
stopping on validation loss (patience 5). Runs are cheap (minutes), so no
mid-run checkpointing — the runner simply restarts an interrupted run.

Datasets: adult (OpenML v2), higgs (OpenML data_id 23512, 98k rows, 28
features), california_housing (sklearn).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.seed import set_seed
from common.optimizers import build_optimizer, GRAD


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _load_raw(dataset, data_dir):
    cache = str(Path(data_dir))
    if dataset == "adult":
        ds = fetch_openml("adult", version=2, as_frame=True, data_home=cache)
        X = ds.data
        y = (ds.target == ">50K").astype(int).values
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]
        return X, y, num_cols, cat_cols, "classification"
    if dataset == "higgs":
        ds = fetch_openml(data_id=23512, as_frame=True, data_home=cache)
        X = ds.data
        y = ds.target.astype(int).values
        # drop rows with missing values (openml higgs has a handful)
        mask = ~X.isna().any(axis=1)
        X, y = X[mask], y[mask.values]
        return X, y, X.columns.tolist(), [], "classification"
    if dataset == "california_housing":
        ds = fetch_california_housing(as_frame=True, data_home=cache)
        return ds.data, ds.target.values, ds.data.columns.tolist(), [], "regression"
    raise ValueError(f"Unknown tabular dataset: {dataset}")


def _split_and_preprocess(X, y, num_cols, cat_cols, task, seed):
    strat = y if task == "classification" else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=strat)
    strat_tmp = y_tmp if task == "classification" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=strat_tmp)

    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False), cat_cols))
    ct = ColumnTransformer(transformers)
    X_train = ct.fit_transform(X_train)          # fit on train only
    X_val, X_test = ct.transform(X_val), ct.transform(X_test)
    return (X_train.astype(np.float32), y_train), (X_val.astype(np.float32), y_val), \
           (X_test.astype(np.float32), y_test)


def _loader(X, y, task, batch_size, shuffle):
    ty = torch.long if task == "classification" else torch.float32
    ds = TensorDataset(torch.from_numpy(np.asarray(X)),
                       torch.as_tensor(np.asarray(y), dtype=ty))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _epoch_pass(model, loader, optimizer, mode, criterion, device, task):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        if mode == GRAD:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out.squeeze(-1) if task == "regression" else out, yb)
            loss.backward()
            optimizer.step()
        else:
            def closure():
                with torch.no_grad():
                    out = model(Xb)
                    return criterion(out.squeeze(-1) if task == "regression" else out, yb)
            loss = optimizer.step(closure)
        total_loss += float(torch.as_tensor(loss).detach())
    return total_loss / len(loader)


@torch.no_grad()
def _evaluate(model, loader, criterion, device, task):
    model.eval()
    total, n = 0.0, 0
    outs, ys = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model(Xb)
        out = out.squeeze(-1) if task == "regression" else out
        total += float(criterion(out, yb))
        n += 1
        outs.append(out.cpu())
        ys.append(yb.cpu())
    out = torch.cat(outs)
    y = torch.cat(ys)
    m = {"loss": total / n}
    if task == "classification":
        prob = torch.softmax(out, dim=1).numpy()
        pred = out.argmax(1).numpy()
        yn = y.numpy()
        m["accuracy"] = float((pred == yn).mean())
        m["macro_f1"] = float(f1_score(yn, pred, average="macro"))
        try:
            m["roc_auc"] = float(roc_auc_score(yn, prob[:, 1]))
        except ValueError:
            m["roc_auc"] = None
    else:
        err = (out - y).numpy()
        m["rmse"] = float(np.sqrt((err ** 2).mean()))
        m["mae"] = float(np.abs(err).mean())
        ss_res = float((err ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        m["r2"] = 1.0 - ss_res / ss_tot
    return m


def run(spec: dict, results_dir: str, data_dir: str) -> dict:
    run_dir = Path(results_dir) / spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(spec["seed"])
    X, y, num_cols, cat_cols, task = _load_raw(spec["dataset"], data_dir)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _split_and_preprocess(
        X, y, num_cols, cat_cols, task, spec["seed"])

    output_dim = len(np.unique(ytr)) if task == "classification" else 1
    model = MLP(Xtr.shape[1], output_dim).to(device)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    train_loader = _loader(Xtr, ytr, task, spec["batch_size"], shuffle=True)
    val_loader = _loader(Xva, yva, task, 512, shuffle=False)
    test_loader = _loader(Xte, yte, task, 512, shuffle=False)

    opt_kwargs = dict(spec.get("opt_kwargs", {}))
    if spec["optimizer"] == "qpso":
        opt_kwargs.setdefault("total_steps", spec["epochs"] * len(train_loader))
        opt_kwargs.setdefault("seed", spec["seed"])
    optimizer, mode = build_optimizer(spec["optimizer"], model, spec["lr"], **opt_kwargs)

    t0 = time.time()
    history, best_val, best_state, patience_left = [], float("inf"), None, 5
    for epoch in range(spec["epochs"]):
        train_loss = _epoch_pass(model, train_loader, optimizer, mode,
                                 criterion, device, task)
        val = _evaluate(model, val_loader, criterion, device, task)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val["loss"]})
        if val["loss"] < best_val - 1e-6:
            best_val, patience_left = val["loss"], 5
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:   # early stopping restores best-val model
        model.load_state_dict(best_state)
    test = _evaluate(model, test_loader, criterion, device, task)

    result = {
        "run_id": spec["run_id"], "domain": "tabular", "tag": spec.get("tag", ""),
        "dataset": spec["dataset"], "model": "MLP", "task": task,
        "optimizer": spec["optimizer"], "seed": spec["seed"], "lr": spec["lr"],
        "epochs_max": spec["epochs"], "completed_epochs": len(history),
        "batch_size": spec["batch_size"], "opt_kwargs": spec.get("opt_kwargs", {}),
        "final_train_loss": history[-1]["train_loss"],
        "best_val_loss": best_val,
        "training_time_seconds": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    result.update({f"test_{k}": v for k, v in test.items()})
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history))
    return result
