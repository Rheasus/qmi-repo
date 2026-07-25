# QMI Benchmark — Quantum-Inspired vs Classical Optimizers

Code and complete run artifacts for:

> **A Cross-Domain Empirical Benchmark of Quantum-Inspired and Classical
> Optimization Algorithms for Machine Learning** — G. Yilmaz (under review,
> major revision).

The study benchmarks classical optimizers (SGD, Adam, AdamW, RMSprop, Adagrad)
against the QI-family under an explicit **implementation taxonomy**:

| Class | Arms | Meaning |
|---|---|---|
| Direct algorithm | SPSA (Spall 1992), QPSO position update (Sun et al. 2004) | published update rules; QPSO adapted to stochastic minibatches |
| Classical analog | NGD-style diagonal squared-minibatch-gradient preconditioner | empirical-Fisher proxy, not a directly computed per-example Fisher |
| Hyperparameter control | HP-QNG, HP-QPSO, HP-COBYLA | the original "-inspired" configurations, kept as controls |

Domains: CV (CIFAR-10/100, Fashion-MNIST × ResNet18/SimpleCNN), NLP (AG News,
IMDb, SST-2 × RoBERTa/DistilBERT/LSTM), tabular (Adult, HIGGS, California
Housing × MLP). 413 training runs, five seeds per headline comparison, single
NVIDIA T4 per run.

## Repository layout

```
v2/                      revised campaign (the version under review)
  codes/
    common/optimizers/   SPSA, QPSO, DiagNGD, HP-* controls, classical arms
    cv/  nlp/  tabular/  domain trainers (checkpoint/resume, collapse handling)
    runner.py            queue-driven executor
    make_queues.py       generates every run specification
    analysis/            aggregate.py, stats.py (paired tests, Holm,
                         modified Hartung-Knapp, exact sign-flip,
                         Friedman/Nemenyi + Bonferroni-Dunn), hessian.py,
                         landscape.py, make_tables.py, make_figures.py
  infra/queue/           the exact JSONL run specifications executed
  runs/
    qmi-{a,b,c,d}/       per-run artifacts: result.json, history.json,
                         hessian.json; qmi-a/analysis/ holds loss-landscape
                         surfaces and interpolations
    stats_summary.json   output of analysis/stats.py
v1/                      original submission's code and consolidated CSVs
                         (retained for provenance; superseded by v2)
```

Model weights are not stored in git (≈54 GB). Training can be rerun from the
specifications (`infra/queue/*.jsonl`) with the seeds recorded in each
`result.json`. The archived `hessian.json` files regenerate the reported
curvature table, but exact checkpoint-level recomputation requires retraining.
The original curvature analysis used one shuffled training batch shared across
arms and seeds within each dataset; its exact sample-index realization was not
stored. Future analysis runs now seed that DataLoader explicitly.

## Reproducing the paper's numbers

The main-text result tables and the four generated figures are produced from `v2/runs/` by the scripts below, with the exceptions the manuscript states: the appendix tables retain v1 values verbatim, and in the QPSO cost table the LSTM Adam reference times are v1 measurements while the AG News row is a projection. The two architecture diagrams are legacy v1 assets and are not generated here:

```bash
cd v2/codes
pip install -r requirements.txt          # torch/torchvision per your platform
python analysis/aggregate.py             # per-configuration summaries
python analysis/stats.py --json ../runs/stats_summary.json
python analysis/make_tables.py           # LaTeX tables
python analysis/make_figures.py          # PDF figures
```

The global reference ledger in `stats_summary.json` contains all 84 reported
inferential quantities (45 uncorrected, 39 under global BH, and 18 under global
Holm). The manuscript's primary multiplicity strategy remains correction
within predeclared families.

## Implementation boundaries

- QPSO historical personal/global-best losses can come from different
  minibatches and dropout draws. They are noisy, non-commensurate evaluations;
  the reported result is scoped to this stochastic-minibatch adaptation.
- NGD tracks an EMA of squared minibatch-mean gradients. Because
  `square(mean(per-example gradients))` is not generally
  `mean(square(per-example gradients))`, it is a second-moment /
  empirical-Fisher proxy.
- Hessian power iteration returns dominant-direction Rayleigh estimates. The
  outputs do not certify positive, algebraically ordered eigenvalues, local
  minima, or a Hessian condition number. Legacy JSON keys are retained for
  artifact compatibility.

Re-running the experiments themselves:

```bash
python runner.py --queue ../infra/queue/cv_rerun.jsonl \
                 --results-dir <out> --data-dir <data>
```

Datasets download automatically (torchvision / Hugging Face / OpenML). CIFAR
archives are fetched from the `uoft-cs/*` Hugging Face mirrors of the original
tarballs (lossless-identical; the origin server throttles downloads).

## Notes on v1

The v1 directory preserves the original submission's pipeline and logs.
Known v1 issues — a CV "SPSA" arm that mixed an Adam-based stand-in with the
genuine algorithm across runs, and manuscript/config mismatches — are
documented and corrected in the revised manuscript; v2 supersedes v1
throughout. Where protocols coincide, v2 reproduces the v1 numbers (see the
paper's replication section).

## License and citation

Code: MIT. If you use this benchmark, please cite the paper above.
