#!/usr/bin/env python3
"""Generate the revised paper's figures (PDF) from stats/landscape artifacts.

  fig_cd_diagram.pdf        Demsar-style critical-difference diagram (CV arms)
  fig_factorial_effects.pdf factorial effect estimates with t-based 95% CIs
  fig_landscape.pdf         1D checkpoint interpolations + 2D surface (CIFAR-10/ResNet18)

Colorblind-safe palette (validated): color follows the arm identity across
all figures; direct labels everywhere (contrast relief).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]
RUNS = BASE / "runs"
FIGS = BASE / "paper" / "Quantum_Optimizer_Benchmarking" / "figures"
PREVIEW = Path(__file__).resolve().parent / "_preview"
PREVIEW.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
STATS = json.loads((RUNS / "stats_summary.json").read_text())

ARM_COLOR = {"adam": "#0072B2", "adamw": "#56B4E9", "sgd": "#009E73",
             "ngd": "#D55E00", "ctrl_qng_hp": "#CC79A7", "spsa": "#E69F00"}
ARM_LABEL = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD",
             "ngd": "NGD", "ctrl_qng_hp": "HP-QNG", "spsa": "SPSA"}

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})


def cd_diagram():
    ranks = STATS["cv_friedman"]["avg_ranks"]
    cd = STATS["cv_friedman"]["cd"]
    order = sorted(ranks.items(), key=lambda x: x[1])  # best first

    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    ax.set_xlim(0.8, 6.2)
    n_left = (len(order) + 1) // 2
    rows = max(n_left, len(order) - n_left)
    ax.set_ylim(-rows - 0.9, 1.9)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)

    # rank axis on top
    ax.axhline(0, color="0.2", lw=1.2)
    for x in range(1, 7):
        ax.plot([x, x], [0, 0.18], color="0.2", lw=1)
        ax.text(x, 0.32, str(x), ha="center", va="bottom", fontsize=8)

    # CD ruler above the axis
    ax.plot([1, 1 + cd], [1.35, 1.35], color="0.25", lw=1.6)
    for x in (1, 1 + cd):
        ax.plot([x, x], [1.22, 1.48], color="0.25", lw=1.6)
    ax.text(1 + cd / 2, 1.58, f"CD = {cd:.2f}", ha="center", fontsize=8)

    # names stacked left (best half) and right (worst half), leader lines to ranks
    for i, (arm, r) in enumerate(order):
        left = i < n_left
        row = (i if left else i - n_left) + 1
        y = -row
        x_text = 0.85 if left else 6.15
        ha = "right" if left else "left"
        ax.plot([r, r], [0, y], color=ARM_COLOR[arm], lw=1.1)
        ax.plot([r, x_text], [y, y], color=ARM_COLOR[arm], lw=1.1)
        ax.text(x_text + (-0.06 if left else 0.06), y,
                f"{ARM_LABEL[arm]} ({r:.2f})", ha=ha, va="center",
                fontsize=8, color="0.1")

    # clique bar just under the axis
    grouped = [r for _, r in order if r - order[0][1] <= cd]
    ax.plot([min(grouped) - 0.06, max(grouped) + 0.06], [-0.38, -0.38],
            color="0.45", lw=3.5, solid_capstyle="round")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cd_diagram.pdf", bbox_inches="tight")
    fig.savefig(PREVIEW / "preview_cd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def factorial_effects():
    ds_names = [("sst2", "SST-2"), ("imdb", "IMDb"), ("ag_news", "AG News")]
    effects = [("lr", "learning-rate effect", "#0072B2"),
               ("wd", "weight-decay effect", "#009E73"),
               ("interaction", "interaction", "#999999")]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.axhline(0, color="0.3", lw=0.8)
    for j, (key, label, color) in enumerate(effects):
        xs, ys, lo, hi = [], [], [], []
        for i, (ds, _) in enumerate(ds_names):
            e = STATS["factorial"][ds][key]
            xs.append(i + (j - 1) * 0.22)
            ys.append(e["mean"] * 100)
            lo.append((e["mean"] - e["ci"][0]) * 100)
            hi.append((e["ci"][1] - e["mean"]) * 100)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=5, lw=1.4, capsize=3,
                    color=color, label=label)
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels([n for _, n in ds_names])
    ax.set_ylabel("effect on test accuracy (pt)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_factorial_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def landscape():
    adir = RUNS / "qmi-a" / "analysis"
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    for other in ["ngd", "sgd", "ctrl_qng_hp"]:
        p = adir / f"interp_cifar10_ResNet18_adam_{other}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        al = [c["alpha"] for c in d["curve"]]
        ls = [c["loss"] for c in d["curve"]]
        ax.plot(al, ls, lw=1.6, color=ARM_COLOR[other],
                label=f"Adam $\\leftrightarrow$ {ARM_LABEL[other]}")
    # checkpoint training losses from the run logs (final epoch, seed 42);
    # the sampled grid comes within 1.25% of the endpoints but not exactly on
    # them, and near-endpoint contamination is informative sharpness signal
    exact = {"adam": 0.029, "ngd": 0.044, "sgd": 0.015, "ctrl_qng_hp": 0.079}
    ax.plot(0, exact["adam"], marker="x", ms=6, color=ARM_COLOR["adam"],
            zorder=5)
    for other in ["ngd", "sgd", "ctrl_qng_hp"]:
        ax.plot(1, exact[other], marker="x", ms=6, color=ARM_COLOR[other],
                zorder=5)
    ax.set_yscale("log")
    ax.axvline(0, color="0.6", lw=0.7, ls=":")
    ax.axvline(1, color="0.6", lw=0.7, ls=":")
    ax.text(0, 0.006, "Adam min.", fontsize=7, ha="center", color="0.35")
    ax.text(1, 0.006, "other min.", fontsize=7, ha="center", color="0.35")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center")
    ax.set_xlabel(r"interpolation coefficient $\alpha$")
    ax.set_ylabel("training loss (log)")
    ax.set_title("(a) linear paths between checkpoints", fontsize=9)

    ax = axes[1]
    p = adir / "surface_cifar10_ResNet18_adam.json"
    if p.exists():
        d = json.loads(p.read_text())
        C = np.array(d["loss_grid"])
        x = np.array(d["coords"])
        levels = np.linspace(C.min(), np.percentile(C, 90), 9)
        cf = ax.contourf(x, x, C.T, levels=levels, cmap="Blues", alpha=0.85,
                         extend="max")
        cb = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label("training loss", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.plot(0, 0, "o", color="#D55E00", ms=6, zorder=5)
        ax.annotate("Adam checkpoint", (0, 0), textcoords="offset points",
                    xytext=(8, 8), fontsize=7.5, color="#D55E00")
    ax.set_xlabel("direction 1 (filter-normalized)")
    ax.set_ylabel("direction 2")
    ax.set_title("(b) loss surface around the Adam checkpoint", fontsize=9)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_landscape.pdf", bbox_inches="tight")
    fig.savefig(PREVIEW / "preview_landscape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def cv_curves():
    """Validation-accuracy curves on CIFAR-10/ResNet18, mean over 5 seeds."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ends = []
    for arm in ["adam", "adamw", "sgd", "ngd", "ctrl_qng_hp", "spsa"]:
        curves = []
        for p in RUNS.glob(f"*/cifar10_ResNet18_{arm}_*_v2cv*/history.json"):
            h = json.loads(p.read_text())
            curves.append([e["val_acc"] for e in h])
        if not curves:
            continue
        n = min(len(c) for c in curves)
        m = np.mean([c[:n] for c in curves], axis=0)
        xs = np.arange(1, n + 1)
        ax.plot(xs, m, lw=1.6, color=ARM_COLOR[arm])
        ends.append([arm, float(m[-1])])
    # vertical dodge for the end-of-line labels (min 3.2pp separation)
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 3.2:
            ends[i][1] = ends[i - 1][1] + 3.2
    for arm, y in ends:
        ax.annotate(ARM_LABEL[arm], (50.8, y), fontsize=7.5,
                    color=ARM_COLOR[arm], va="center",
                    annotation_clip=False)
    ax.set_xlim(1, 50)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cv_curves.pdf", bbox_inches="tight")
    fig.savefig(PREVIEW / "preview_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cd_diagram()
    factorial_effects()
    landscape()
    cv_curves()
    for f in ["fig_cd_diagram.pdf", "fig_factorial_effects.pdf",
              "fig_landscape.pdf", "fig_cv_curves.pdf"]:
        print("wrote", FIGS / f)
