#!/usr/bin/env python3
"""Aggregate v2 run results into per-configuration summaries.

Reads every result.json under v2/runs/**, groups by (domain, dataset, model,
arm) and prints mean±std tables per domain. The `arm` is arm_label when
present (NLP factorial cells) else the optimizer name.

Usage: python aggregate.py [--domain cv|nlp|tabular] [--csv out.csv]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

RUNS = Path(__file__).resolve().parents[2] / "runs"

PRIMARY = {"cv": "test_accuracy", "nlp": "test_accuracy",
           "tabular": None}  # tabular: accuracy or rmse by task


def load_results():
    rows = []
    for p in RUNS.glob("*/**/result.json"):
        if "smoke" in str(p):
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("tag", "").startswith("smoke") or r.get("tag") == "smoke":
            continue
        rows.append(r)
    return rows


def arm_of(r):
    """Canonical arm label. NLP RoBERTa AdamW-family runs are pooled by their
    EFFECTIVE (lr, wd): the v2seed 'adamw' runs equal the factorial baseline
    cell, and 'ctrl_qng_hp' (AdamW at lr*0.5, wd 0.05) equals the v1-QNG cell."""
    if r["domain"] == "nlp" and r.get("model") == "roberta":
        opt = r["optimizer"]
        if opt == "adamw":
            lr = r["lr"]
            wd = r.get("opt_kwargs", {}).get("weight_decay", 0.01)
            return f"adamw_lr{lr:g}_wd{wd:g}"
        if opt == "ctrl_qng_hp":
            return f"adamw_lr{r['lr'] * 0.5:g}_wd0.05"
    return r.get("arm_label") or r["optimizer"]


def key_metric(r):
    if r["domain"] == "tabular" and r.get("task") == "regression":
        return "test_rmse", r.get("test_rmse")
    return "test_accuracy", r.get("test_accuracy")


def summarize(rows, domain=None):
    groups = defaultdict(list)
    for r in rows:
        if domain and r["domain"] != domain:
            continue
        groups[(r["domain"], r["dataset"], r.get("model", ""), arm_of(r))].append(r)

    out = []
    for (dom, ds, model, arm), rs in sorted(groups.items()):
        import math
        mname, _ = key_metric(rs[0])
        vals = [key_metric(r)[1] for r in rs
                if key_metric(r)[1] is not None and math.isfinite(key_metric(r)[1])]
        seeds = sorted({r["seed"] for r in rs})
        times = [r.get("training_time_seconds", 0) for r in rs]
        collapsed = sum(1 for r in rs if r.get("collapsed"))
        out.append({
            "domain": dom, "dataset": ds, "model": model, "arm": arm,
            "n": len(rs), "seeds": ",".join(map(str, seeds)),
            "metric": mname,
            "mean": mean(vals) if vals else float("nan"),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "mean_time_s": mean(times) if times else 0,
            "collapsed": collapsed,
        })
    return out


def print_table(summary):
    cur = None
    for s in summary:
        block = (s["domain"], s["dataset"], s["model"])
        if block != cur:
            cur = block
            print(f"\n--- {s['domain']} | {s['dataset']} | {s['model']} "
                  f"({s['metric']}) ---")
        flag = f" collapsed:{s['collapsed']}" if s["collapsed"] else ""
        print(f"  {s['arm']:<22} n={s['n']:<2} {s['mean']:.4f} ± {s['std']:.4f}"
              f"  t={s['mean_time_s']:.0f}s{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = load_results()
    print(f"loaded {len(rows)} runs")
    summary = summarize(rows, args.domain)
    print_table(summary)
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"\nwrote {args.csv}")
