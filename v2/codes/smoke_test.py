#!/usr/bin/env python3
"""End-to-end CPU smoke test of the CV pipeline (1 epoch, small batches).

Validates: data loading + split, all optimizer modes (grad/closure), epoch
checkpointing, resume, metrics, result.json schema. Run before any VM queue.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv.train import run  # noqa: E402


def main():
    base = Path(__file__).resolve().parents[1] / "runs"
    for opt, lr, okw in [("spsa", 0.1, {"c": 0.01}),
                         ("ngd", 0.001, {"damping": 1e-3}),
                         ("adam", 0.001, {})]:
        spec = {
            "run_id": f"smoke_fashion_SimpleCNN_{opt}", "domain": "cv",
            "tag": "smoke", "dataset": "fashion_mnist", "model": "SimpleCNN",
            "optimizer": opt, "seed": 42, "lr": lr, "epochs": 1,
            "batch_size": 256, "opt_kwargs": okw,
        }
        res = run(spec, str(base / "smoke"), str(base / "datasets"))
        print(json.dumps({k: res[k] for k in
                          ["run_id", "test_accuracy", "final_train_acc",
                           "training_time_seconds"]}))


if __name__ == "__main__":
    main()
