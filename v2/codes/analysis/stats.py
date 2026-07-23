#!/usr/bin/env python3
"""Statistical analysis for the v2 revision.

Produces, from the synced result.json files:
  1. RoBERTa factorial: lr / wd / interaction effect estimates per dataset,
     with seed-paired t-tests and bootstrap 95% CIs.
  2. Genuine-NGD comparisons vs the v1-QNG cell and vs baseline AdamW,
     per dataset (paired t) and pooled across datasets (Wilcoxon, n=15 pairs).
  3. CV: every arm vs Adam per configuration (paired by seed), with Holm
     correction across the family of comparisons.
  4. CV: Friedman test + average ranks + Nemenyi critical difference
     across the 6 gradient arms over the 6 configurations.
  5. Tabular MLP: arms vs Adam.

Note on small n: with 5 seeds the two-sided sign-rank floor is p=0.0625, so
per-config inference uses paired t-tests (with CIs and effect sizes); Wilcoxon
is used where pooling gives n >= 15 pairs. This is stated in the paper.

Usage: python analysis/stats.py [--json out.json]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

RUNS = Path(__file__).resolve().parents[2] / "runs"
SEEDS5 = [42, 1337, 2025, 7, 123]


def load():
    rows = []
    for p in RUNS.glob("*/**/result.json"):
        if "smoke" in str(p):
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("tag") == "smoke":
            continue
        rows.append(r)
    return rows


def by_seed(rows, domain, dataset, model, pred):
    """dict seed -> metric for rows matching pred (latest wins on dup)."""
    out = {}
    for r in rows:
        if r["domain"] != domain or r["dataset"] != dataset or r.get("model") != model:
            continue
        if not pred(r):
            continue
        v = r.get("test_accuracy")
        if v is not None and math.isfinite(v):
            out[r["seed"]] = v
    return out


def paired(a: dict, b: dict):
    seeds = sorted(set(a) & set(b))
    return (np.array([a[s] for s in seeds]),
            np.array([b[s] for s in seeds]), seeds)


def boot_ci(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    m = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def paired_t(a, b):
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    return float(st.ttest_rel(a, b).pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


# ------------------------------------------------------------------ factorial
def factorial(rows, report, out):
    def cell(ds, lr, wd):
        def pred(r):
            if r["optimizer"] == "adamw" and abs(r["lr"] - lr) < 1e-12 \
               and abs(r.get("opt_kwargs", {}).get("weight_decay", 0.01) - wd) < 1e-9:
                return True
            if r["optimizer"] == "ctrl_qng_hp" and abs(lr - 1e-5) < 1e-12 and wd == 0.05:
                return True
            return False
        return by_seed(rows, "nlp", ds, "roberta", pred)

    report.append("\n== RoBERTa 2x2 factorial (lr x wd), seed-paired ==")
    out["factorial"] = {}
    for ds in ["sst2", "imdb", "ag_news"]:
        c = {(lr, wd): cell(ds, lr, wd)
             for lr in (2e-5, 1e-5) for wd in (0.01, 0.05)}
        seeds = sorted(set.intersection(*[set(v) for v in c.values()]))
        A = {k: np.array([v[s] for s in seeds]) for k, v in c.items()}
        lr_eff = ((A[(1e-5, 0.01)] + A[(1e-5, 0.05)]) -
                  (A[(2e-5, 0.01)] + A[(2e-5, 0.05)])) / 2
        wd_eff = ((A[(2e-5, 0.05)] + A[(1e-5, 0.05)]) -
                  (A[(2e-5, 0.01)] + A[(1e-5, 0.01)])) / 2
        inter = ((A[(1e-5, 0.05)] - A[(1e-5, 0.01)]) -
                 (A[(2e-5, 0.05)] - A[(2e-5, 0.01)]))
        res = {}
        for name, eff in [("lr", lr_eff), ("wd", wd_eff), ("interaction", inter)]:
            lo, hi = boot_ci(eff)
            p = 1.0 if np.allclose(eff, 0) else float(st.ttest_1samp(eff, 0).pvalue)
            res[name] = {"mean": float(eff.mean()), "ci": [lo, hi], "p": p,
                         "n_seeds": len(seeds)}
            report.append(f"  {ds:<8} {name:<12} {eff.mean()*100:+.2f}pt "
                          f"CI[{lo*100:+.2f},{hi*100:+.2f}] p={p:.3f} (n={len(seeds)})")
        out["factorial"][ds] = res


# ------------------------------------------------------------------ NGD
def ngd_comparisons(rows, report, out):
    report.append("\n== Genuine NGD vs AdamW cells (RoBERTa) ==")
    out["ngd"] = {}
    pooled = {"vs_v1qng": ([], []), "vs_baseline": ([], [])}
    for ds in ["sst2", "imdb", "ag_news"]:
        ngd = by_seed(rows, "nlp", ds, "roberta", lambda r: r["optimizer"] == "ngd")
        v1qng = by_seed(rows, "nlp", ds, "roberta",
                        lambda r: (r["optimizer"] == "ctrl_qng_hp") or
                        (r["optimizer"] == "adamw" and abs(r["lr"] - 1e-5) < 1e-12 and
                         abs(r.get("opt_kwargs", {}).get("weight_decay", 0) - 0.05) < 1e-9))
        base = by_seed(rows, "nlp", ds, "roberta",
                       lambda r: r["optimizer"] == "adamw" and abs(r["lr"] - 2e-5) < 1e-12 and
                       abs(r.get("opt_kwargs", {}).get("weight_decay", 0) - 0.01) < 1e-9)
        res = {}
        for label, ref in [("vs_v1qng", v1qng), ("vs_baseline", base)]:
            a, b, seeds = paired(ngd, ref)
            d = a - b
            lo, hi = boot_ci(d)
            res[label] = {"mean_diff": float(d.mean()), "ci": [lo, hi],
                          "p_paired_t": paired_t(a, b), "n": len(seeds)}
            pooled[label][0].extend(a.tolist())
            pooled[label][1].extend(b.tolist())
            report.append(f"  {ds:<8} ngd {label:<12} {d.mean()*100:+.2f}pt "
                          f"CI[{lo*100:+.2f},{hi*100:+.2f}] p={res[label]['p_paired_t']:.3f}")
        out["ngd"][ds] = res
    for label, (a, b) in pooled.items():
        a, b = np.array(a), np.array(b)
        w = st.wilcoxon(a, b) if not np.allclose(a, b) else None
        p = float(w.pvalue) if w else 1.0
        report.append(f"  POOLED   ngd {label:<12} {(a-b).mean()*100:+.2f}pt "
                      f"Wilcoxon p={p:.3f} (n={len(a)} pairs)")
        lo, hi = boot_ci(a - b)
        out["ngd"][f"pooled_{label}"] = {"mean_diff": float((a - b).mean()),
                                         "ci": [lo, hi],
                                         "p_wilcoxon": p, "n": int(len(a))}


# ------------------------------------------------------------------ CV
CV_CONFIGS = [("fashion_mnist", "SimpleCNN"), ("cifar10", "SimpleCNN"),
              ("cifar100", "SimpleCNN"), ("fashion_mnist", "ResNet18"),
              ("cifar10", "ResNet18"), ("cifar100", "ResNet18")]
CV_ARMS = ["adamw", "sgd", "ngd", "ctrl_qng_hp", "spsa"]


def cv_vs_adam(rows, report, out):
    report.append("\n== CV: arms vs Adam (seed-paired, Holm-corrected) ==")
    comps, meta = [], []
    for ds, model in CV_CONFIGS:
        adam = by_seed(rows, "cv", ds, model, lambda r: r["optimizer"] == "adam")
        for arm in CV_ARMS:
            other = by_seed(rows, "cv", ds, model,
                            lambda r, a=arm: r["optimizer"] == a)
            a, b, seeds = paired(other, adam)
            if len(seeds) < 3:
                continue
            d = a - b
            comps.append(paired_t(a, b))
            meta.append((ds, model, arm, float(d.mean()), boot_ci(d), len(seeds)))
    adj = holm([p if not math.isnan(p) else 1.0 for p in comps])
    out["cv_vs_adam"] = []
    for (ds, model, arm, md, (lo, hi), n), p_raw, p_adj in zip(meta, comps, adj):
        sig = "*" if p_adj < 0.05 else " "
        report.append(f"  {ds:<14}{model:<10}{arm:<12} {md*100:+.2f}pt "
                      f"CI[{lo*100:+.2f},{hi*100:+.2f}] p={p_raw:.4f} "
                      f"holm={p_adj:.4f}{sig} (n={n})")
        out["cv_vs_adam"].append({"dataset": ds, "model": model, "arm": arm,
                                  "mean_diff": md, "ci": [lo, hi],
                                  "p": p_raw, "p_holm": p_adj, "n": n})


def cv_friedman(rows, report, out):
    report.append("\n== CV: Friedman + average ranks (6 gradient arms x 6 configs) ==")
    arms = ["adam", "adamw", "sgd", "ngd", "ctrl_qng_hp", "spsa"]
    table = []
    for ds, model in CV_CONFIGS:
        row = []
        for arm in arms:
            vals = by_seed(rows, "cv", ds, model, lambda r, a=arm: r["optimizer"] == a)
            row.append(np.mean(list(vals.values())) if vals else np.nan)
        table.append(row)
    T = np.array(table)
    fr = st.friedmanchisquare(*[T[:, j] for j in range(T.shape[1])])
    ranks = (-T).argsort(axis=1).argsort(axis=1) + 1
    avg_ranks = ranks.mean(axis=0)
    k, n = len(arms), T.shape[0]
    cd = 2.850 * math.sqrt(k * (k + 1) / (6 * n))  # Nemenyi q_0.05 for k=6
    report.append(f"  Friedman chi2={fr.statistic:.2f} p={fr.pvalue:.4f}; "
                  f"Nemenyi CD(0.05)={cd:.2f}")
    for arm, r in sorted(zip(arms, avg_ranks), key=lambda x: x[1]):
        report.append(f"    {arm:<14} avg rank {r:.2f}")
    out["cv_friedman"] = {"p": float(fr.pvalue), "cd": cd,
                          "avg_ranks": dict(zip(arms, avg_ranks.tolist()))}
    # Bonferroni-Dunn control comparison vs Adam (higher power than Nemenyi)
    se = math.sqrt(k * (k + 1) / (6 * n))
    ctrl = avg_ranks[arms.index("adam")]
    bd = {}
    for arm, r in zip(arms, avg_ranks):
        if arm == "adam":
            continue
        z = (r - ctrl) / se
        p_raw = 2 * (1 - st.norm.cdf(abs(z)))
        bd[arm] = {"z": float(z), "p_adj": min(1.0, float(p_raw) * (k - 1))}
        report.append(f"    BD vs adam: {arm:<14} z={z:+.2f} adj_p={bd[arm]['p_adj']:.4f}")
    out["cv_friedman"]["bonferroni_dunn_vs_adam"] = bd


# ------------------------------------------------------------------ tabular
def tabular_vs_adam(rows, report, out):
    report.append("\n== Tabular MLP: arms vs Adam ==")
    out["tabular"] = []
    for ds in ["adult", "higgs", "california_housing"]:
        metric = "test_rmse" if ds == "california_housing" else "test_accuracy"
        def get(arm):
            vals = {}
            for r in rows:
                if (r["domain"], r["dataset"], r["optimizer"]) == ("tabular", ds, arm):
                    v = r.get(metric)
                    if v is not None and math.isfinite(v):
                        vals[r["seed"]] = v
            return vals
        adam = get("adam")
        for arm in ["adamw", "sgd", "ngd", "ctrl_qng_hp", "spsa", "qpso"]:
            a, b, seeds = paired(get(arm), adam)
            if len(seeds) < 3:
                continue
            d = a - b
            lo, hi = boot_ci(d)
            p = paired_t(a, b)
            report.append(f"  {ds:<20}{arm:<12} {metric[5:]:<5} d={d.mean():+.4f} "
                          f"CI[{lo:+.4f},{hi:+.4f}] p={p:.3f} (n={len(seeds)})")
            out["tabular"].append({"dataset": ds, "arm": arm, "metric": metric,
                                   "mean_diff": float(d.mean()), "ci": [lo, hi],
                                   "p": p, "n": len(seeds)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = load()
    report = [f"loaded {len(rows)} runs"]
    out = {}
    factorial(rows, report, out)
    ngd_comparisons(rows, report, out)
    cv_vs_adam(rows, report, out)
    cv_friedman(rows, report, out)
    tabular_vs_adam(rows, report, out)
    print("\n".join(report))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
