#!/usr/bin/env python3
"""SPSA step-size (Spall 'a') grid smoke: pick the a that avoids divergence.

1-epoch CPU runs on Fashion-MNIST/SimpleCNN. a=0.1 already shown to diverge
(NaN); this probes {0.01, 0.001, 0.0001}. The chosen value goes into the CV
queue and is documented in the paper as selected-by-validation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv.train import run  # noqa: E402


def main():
    base = Path(__file__).resolve().parents[1] / "runs"
    for a in [0.01, 0.001, 0.0001]:
        spec = {
            "run_id": f"smoke_spsa_a{a}", "domain": "cv", "tag": "smoke",
            "dataset": "fashion_mnist", "model": "SimpleCNN",
            "optimizer": "spsa", "seed": 42, "lr": a, "epochs": 1,
            "batch_size": 256, "opt_kwargs": {"c": 0.01},
        }
        res = run(spec, str(base / "smoke"), str(base / "datasets"))
        print(json.dumps({"a": a, "test_acc": res["test_accuracy"],
                          "train_loss": res["final_train_loss"],
                          "collapsed": res["collapsed"]}))


if __name__ == "__main__":
    main()
