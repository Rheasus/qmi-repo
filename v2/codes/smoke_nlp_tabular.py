#!/usr/bin/env python3
"""CPU smoke for the NLP and tabular pipelines (tiny subsets, 1 epoch).

Covers: transformer + LSTM forward paths, grad + closure modes, tokenizer/
dataset caching, tabular preprocessing (numeric-only and one-hot), regression
and classification metrics, result.json schemas.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nlp.train import run as nlp_run          # noqa: E402
from tabular.train import run as tab_run      # noqa: E402

BASE = Path(__file__).resolve().parents[1] / "runs"


def main():
    nlp_specs = [
        {"run_id": "smoke_sst2_lstm_adam", "domain": "nlp", "tag": "smoke",
         "dataset": "sst2", "model": "lstm", "optimizer": "adam", "seed": 42,
         "lr": 1e-3, "epochs": 1, "batch_size": 16, "max_samples": 200,
         "opt_kwargs": {}},
        {"run_id": "smoke_sst2_distilbert_adamw", "domain": "nlp", "tag": "smoke",
         "dataset": "sst2", "model": "distilbert", "optimizer": "adamw",
         "seed": 42, "lr": 2e-5, "epochs": 1, "batch_size": 16,
         "max_samples": 200, "opt_kwargs": {"weight_decay": 0.01}},
        {"run_id": "smoke_sst2_lstm_qpso", "domain": "nlp", "tag": "smoke",
         "dataset": "sst2", "model": "lstm", "optimizer": "qpso", "seed": 42,
         "lr": 0.0, "epochs": 1, "batch_size": 16, "max_samples": 200,
         "opt_kwargs": {"n_particles": 4}},
    ]
    for spec in nlp_specs:
        res = nlp_run(spec, str(BASE / "smoke"), str(BASE / "datasets"))
        print(json.dumps({"run": spec["run_id"],
                          "acc": res["test_accuracy"],
                          "t": round(res["training_time_seconds"], 1)}))

    tab_specs = [
        {"run_id": "smoke_calif_MLP_adam", "domain": "tabular", "tag": "smoke",
         "dataset": "california_housing", "model": "MLP", "optimizer": "adam",
         "seed": 42, "lr": 1e-3, "epochs": 3, "batch_size": 32, "opt_kwargs": {}},
        {"run_id": "smoke_adult_MLP_spsa", "domain": "tabular", "tag": "smoke",
         "dataset": "adult", "model": "MLP", "optimizer": "spsa", "seed": 42,
         "lr": 0.01, "epochs": 3, "batch_size": 32, "opt_kwargs": {"c": 0.01}},
    ]
    for spec in tab_specs:
        res = tab_run(spec, str(BASE / "smoke"), str(BASE / "datasets"))
        key = "test_accuracy" if "test_accuracy" in res else "test_rmse"
        print(json.dumps({"run": spec["run_id"], key: res[key],
                          "t": round(res["training_time_seconds"], 1)}))


if __name__ == "__main__":
    main()
