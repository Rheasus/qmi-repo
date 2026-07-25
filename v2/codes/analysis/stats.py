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


def boot_ci(diffs, n=10000, seed=0, small_n=10):
    """95% CI for the mean of paired differences.

    For small samples (n < small_n, i.e. every per-configuration comparison in
    this study) the percentile bootstrap is badly anticonservative: at n=5 only
    C(9,5)=126 distinct resample multisets exist and simulated coverage of the
    nominal-95% interval is ~83%. We therefore report the Student-t interval
    there, which is exact under normality and agrees by construction with the
    paired t-test used for significance. The percentile bootstrap is retained
    for the pooled comparisons (n>=15), where it is well behaved.
    """
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    if len(d) < small_n:
        if len(d) < 2 or np.allclose(d, d[0]):
            return (float(d.mean()), float(d.mean()))
        se = d.std(ddof=1) / np.sqrt(len(d))
        h = st.t.ppf(0.975, len(d) - 1) * se
        return (float(d.mean() - h), float(d.mean() + h))
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
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


def cluster_summary(per_group):
    """Dataset-level (cluster-aware) summary of a set of per-seed effect arrays.

    The pooled seed-level tests treat 15 seed-dataset pairs as exchangeable;
    they are 3 clusters of 5. This reports the DerSimonian-Laird random-effects
    estimate over the cluster means, Cochran's Q heterogeneity test, and the
    exact cluster-level sign-flip p-value (whose floor at 3 clusters is 0.25).
    """
    means = np.array([g.mean() for g in per_group], dtype=float)
    ses = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in per_group], dtype=float)
    k = len(means)
    w = 1.0 / ses ** 2
    mu_fe = float((w * means).sum() / w.sum())
    Q = float((w * (means - mu_fe) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    w2 = 1.0 / (ses ** 2 + tau2)
    mu = float((w2 * means).sum() / w2.sum())
    se = float(np.sqrt(1.0 / w2.sum()))
    z = mu / se if se > 0 else float("nan")
    p_re = float(2 * (1 - st.norm.cdf(abs(z))))
    import itertools
    obs = abs(means.mean())
    flips = list(itertools.product([-1, 1], repeat=k))
    p_flip = sum(1 for f in flips if abs(np.mean(np.array(f) * means)) >= obs - 1e-12) / len(flips)
    return {"per_dataset_means": means.tolist(),
            "re_mean": mu, "re_se": se, "re_p": p_re,
            "Q": Q, "Q_p": float(1 - st.chi2.cdf(Q, k - 1)),
            "sign_flip_p": p_flip, "n_clusters": k}


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

    # pooled per-seed effects across datasets (descriptive; the cluster-aware
    # dataset-level summary is the inferential statement)
    import numpy as _np
    all_lr, all_wd = [], []
    per_ds_lr, per_ds_wd = [], []
    for ds in ["sst2", "imdb", "ag_news"]:
        c = {(lr, wd): cell(ds, lr, wd) for lr in (2e-5, 1e-5) for wd in (0.01, 0.05)}
        seeds = sorted(set.intersection(*[set(v) for v in c.values()]))
        A = {k: _np.array([v[s] for s in seeds]) for k, v in c.items()}
        _lr = ((A[(1e-5,0.01)]+A[(1e-5,0.05)])-(A[(2e-5,0.01)]+A[(2e-5,0.05)]))/2
        _wd = ((A[(2e-5,0.05)]+A[(1e-5,0.05)])-(A[(2e-5,0.01)]+A[(1e-5,0.01)]))/2
        all_lr.extend(_lr.tolist()); per_ds_lr.append(_lr)
        all_wd.extend(_wd.tolist()); per_ds_wd.append(_wd)
    all_lr, all_wd = _np.array(all_lr), _np.array(all_wd)
    out["factorial"]["pooled"] = {
        "lr": {"mean": float(all_lr.mean()), "p_wilcoxon": float(st.wilcoxon(all_lr).pvalue),
               "n": len(all_lr), **cluster_summary(per_ds_lr)},
        "wd": {"mean": float(all_wd.mean()), "p_wilcoxon": float(st.wilcoxon(all_wd).pvalue),
               "n": len(all_wd), **cluster_summary(per_ds_wd)},
    }
    report.append(f"  POOLED   lr {all_lr.mean()*100:+.2f}pt Wilcoxon p={st.wilcoxon(all_lr).pvalue:.5f}; "
                  f"wd {all_wd.mean()*100:+.2f}pt p={st.wilcoxon(all_wd).pvalue:.3f} (n={len(all_lr)})")


# ------------------------------------------------------------------ NGD
def ngd_comparisons(rows, report, out):
    report.append("\n== Genuine NGD vs AdamW cells (RoBERTa) ==")
    out["ngd"] = {}
    pooled = {"vs_v1qng": ([], []), "vs_baseline": ([], [])}
    per_ds = {"vs_v1qng": [], "vs_baseline": []}
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
            per_ds[label].append(d)
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
                                         "p_wilcoxon": p, "n": int(len(a)),
                                         **cluster_summary(per_ds[label])}


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
    # Within-family Holm over the whole tabular family, reported so that the
    # unadjusted per-dataset values in the text can be read against it.
    _ps = [e["p"] for e in out["tabular"]]
    for e, adj in zip(out["tabular"], holm(_ps)):
        e["p_holm_family"] = adj
    report.append(f"  (tabular family: {len(_ps)} tests; "
                  f"{sum(1 for e in out['tabular'] if e['p'] < 0.05)} raw-significant, "
                  f"{sum(1 for e in out['tabular'] if e['p_holm_family'] < 0.05)} after within-family Holm)")


def multiplicity_summary(out, report):
    """Global multiplicity reference point over every test reported in the main text.

    The paper corrects within families, not globally. This records what a single
    global correction over the whole reported set would give, so the claim is
    reproducible rather than asserted. Membership is explicit below.
    """
    groups = {}
    groups["cv_vs_adam"] = [e["p"] for e in out.get("cv_vs_adam", [])]
    groups["tabular_vs_adam"] = [e["p"] for e in out.get("tabular", [])]
    fac = out.get("factorial", {})
    groups["factorial_contrasts"] = [fac[ds][k]["p"] for ds in ("sst2", "imdb", "ag_news")
                                     for k in ("lr", "wd", "interaction") if ds in fac]
    groups["factorial_pooled"] = [fac["pooled"][k]["p_wilcoxon"] for k in ("lr", "wd")] if "pooled" in fac else []
    ngd = out.get("ngd", {})
    groups["ngd_per_dataset"] = [ngd[ds][k]["p_paired_t"] for ds in ("sst2", "imdb", "ag_news")
                                 for k in ("vs_v1qng", "vs_baseline") if ds in ngd]
    groups["ngd_pooled"] = [ngd[k]["p_wilcoxon"] for k in ("pooled_vs_v1qng", "pooled_vs_baseline") if k in ngd]
    fr = out.get("cv_friedman", {})
    groups["friedman"] = [fr["p"]] if "p" in fr else []
    # Dunn entries store the family-adjusted p; the global reference uses the raw
    # two-sided normal p implied by z.
    groups["bonferroni_dunn"] = [float(2 * (1 - st.norm.cdf(abs(d["z"]))))
                                 for d in (fr.get("bonferroni_dunn_vs_adam") or {}).values()]

    allp = [p for g in groups.values() for p in g]
    m = len(allp)
    raw = sum(1 for p in allp if p < 0.05)
    holm_sig = sum(1 for p in holm(allp) if p < 0.05)
    order = np.argsort(allp)
    bh_sig, thresh = 0, 0
    for rank, idx in enumerate(order, start=1):
        if allp[idx] <= 0.05 * rank / m:
            thresh = rank
    bh_sig = thresh
    out["multiplicity"] = {"per_group": {k: len(v) for k, v in groups.items()},
                           "n_tests": m, "raw_significant": raw,
                           "bh_significant": bh_sig, "global_holm_significant": holm_sig}
    report.append(f"\n== Global multiplicity reference ==\n  {m} reported tests "
                  f"({', '.join(f'{k}:{len(v)}' for k, v in groups.items())}); "
                  f"{raw} significant uncorrected, {bh_sig} under global BH(0.05), "
                  f"{holm_sig} under global Holm")


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
    multiplicity_summary(out, report)
    print("\n".join(report))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
