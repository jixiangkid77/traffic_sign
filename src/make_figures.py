r"""
make_figures.py  Regenerate manuscript figures 1, 4, 5, 6, 7, 11.

WHERE TO PUT THIS FILE
  Place it in  D:\Project\traffic_sign\src\  (next to K_merge_results.py).
  It reads the locked result files from  ..\outputs_revision\  and writes
  PNG + SVG into  ..\outputs_revision\figures\ .
  If your project lives elsewhere, edit PROJECT_ROOT just below.

RUN
  python make_figures.py            # makes all six
  python make_figures.py 5 11       # makes only Fig. 5 and Fig. 11

REQUIRES
  pip install matplotlib numpy
  (no GPU, no torch; pure plotting from the CSV/JSON artifacts)

FIGURES PRODUCED (manuscript numbering)
  Fig. 1  pipeline overview (schematic, vertical)      -> no data file needed
  Fig. 4  accuracy vs severity, real CURE-TSR          -> cells_per_method.csv
  Fig. 5  per-challenge boundary bars (with CIs)       -> merged_per_image.csv
  Fig. 6  routing headroom ladder                      -> extended_results.json
  Fig. 7  routing behavior (branch shares)             -> merged_per_image.csv
  Fig. 11 accuracy vs enhancement cost                 -> extended + L timing
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ------------------------------------------------------------------ paths
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")     # <-- edit if needed
OUT = PROJECT_ROOT / "outputs_revision"
FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CELLS = OUT / "cells_per_method.csv"
MERGED = OUT / "merged_per_image.csv"
EXT = OUT / "extended_results.json"
LJS = OUT / "L_timing_enhance_only.results.json"

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
})

C = {"passthrough": "#8c8c8c", "gamma": "#6a9a3f", "clahe": "#d99a2b",
     "stretch": "#4a86c7", "va_rule": "#1f3a6e", "cidnet": "#b0468e",
     "adair": "#cc4125", "oracle4": "#2e8b57", "oracle6": "#9b7fb0"}
LAB = {"passthrough": "Baseline", "gamma": "Gamma", "clahe": "CLAHE",
       "stretch": "Stretch", "va_rule": "VA-Adaptive", "cidnet": "HVI-CIDNet",
       "adair": "AdaIR", "oracle4": "Oracle-4", "oracle6": "Oracle-6"}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGDIR / f"{name}.png")


def load_merged():
    ch, sev, true = [], [], []
    cols = {k: [] for k in ["rule_branch", "pred_va_rule", "pred_cidnet",
                            "pred_adair"]}
    with open(MERGED) as f:
        for r in csv.DictReader(f):
            ch.append(int(r["ch"])); sev.append(int(r["sev"]))
            true.append(int(r["true"]))
            cols["rule_branch"].append(r["rule_branch"])
            for k in ["pred_va_rule", "pred_cidnet", "pred_adair"]:
                cols[k].append(int(r[k]))
    return (np.array(ch), np.array(sev), np.array(true),
            {k: np.array(v) for k, v in cols.items()})


# ============================================================= Figure 1
def fig1():
    """Vertical pipeline schematic, large text."""
    fig, ax = plt.subplots(figsize=(7.5, 12))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(cx, cy, w, h, text, fc, fs=15, bold=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.4,rounding_size=1.6",
                     linewidth=1.4, edgecolor="#333333", facecolor=fc))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=22, linewidth=1.6, color="#333333"))

    ax.text(50, 98, "VA-Adaptive routing pipeline", ha="center",
            fontsize=19, fontweight="bold")
    box(50, 90, 34, 7, "Input image", "#eef2f7", 16)
    arrow(50, 86.5, 50, 83)
    box(50, 79, 44, 7.5, "Visibility statistics\nb, c, e", "#eef2f7", 16)
    arrow(50, 75, 50, 71.5)
    box(50, 67, 40, 8, "Threshold routing\nT1 to T4", "#dfe9f3", 16, bold=True)

    # four branches spread horizontally
    ys = 52
    xs = [16, 39, 61, 84]
    names = ["gamma", "clahe", "stretch", "passthrough"]
    for x, nm in zip(xs, names):
        arrow(50, 63, x, ys + 5)
        box(x, ys, 20, 8, LAB[nm], "#ffffff", 14)
        arrow(x, ys - 5, 50, 37)
    ax.text(50, 58.5, "priority order: first matching condition applies",
            ha="center", fontsize=12, style="italic", color="#555555")

    box(50, 33, 46, 8, "Enhanced image\nresized to 32 x 32", "#eef2f7", 16)
    arrow(50, 29, 50, 25.5)
    box(50, 21, 34, 7.5, "Frozen CNN", "#dfe9f3", 16, bold=True)
    arrow(50, 17, 50, 13.5)
    box(50, 9, 30, 7, "Class label", "#eef2f7", 16)

    ax.text(50, 1.5, "thresholds calibrated once on clean data; "
            "no learned parameters in routing", ha="center", fontsize=12,
            style="italic", color="#555555")
    save(fig, "Fig01_pipeline_overview")


# ============================================================= Figure 4
def fig4():
    rows = list(csv.DictReader(open(CELLS)))
    challenges = ["Darkening", "Noise", "Rain", "Snow", "Haze"]
    methods = ["passthrough", "gamma", "clahe", "stretch", "va_rule",
               "cidnet", "adair"]
    style = {"va_rule": (3.2, "-", 6, 1.0), "adair": (2.4, "-", 5, 1.0),
             "cidnet": (2.4, "-", 5, 1.0), "passthrough": (1.6, "--", 3, 0.9),
             "gamma": (1.4, "-", 2, 0.75), "clahe": (1.4, "-", 2, 0.75),
             "stretch": (1.4, "-", 2, 0.75)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ai, ch in enumerate(challenges):
        ax = axes[ai]
        sub = sorted([r for r in rows if r["ch_name"] == ch],
                     key=lambda r: int(r["sev"]))
        xs = [int(r["sev"]) for r in sub]
        for m in methods:
            lw, ls, z, al = style[m]
            ax.plot(xs, [float(r[m]) for r in sub], ls, color=C[m],
                    linewidth=lw, zorder=z, alpha=al,
                    marker="o" if m == "va_rule" else None, markersize=6)
        ax.plot(xs, [float(r["oracle4"]) for r in sub], ":",
                color=C["oracle4"], linewidth=2.2, zorder=4)
        ax.set_title(ch, fontweight="bold")
        ax.set_xlabel("Severity"); ax.set_xticks(xs)
        ax.grid(True, alpha=0.3, linewidth=0.7)
        ax.tick_params(labelsize=13)
        if ai % 3 == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
    axes[5].axis("off")
    handles = [plt.Line2D([0], [0], color=C[m], linewidth=style[m][0],
               linestyle=style[m][1], label=LAB[m]) for m in methods]
    handles.append(plt.Line2D([0], [0], color=C["oracle4"], linewidth=2.2,
                   linestyle=":", label="Oracle-4"))
    axes[5].legend(handles=handles, loc="center", fontsize=15,
                   title="Method", title_fontsize=16)
    fig.tight_layout()
    save(fig, "Fig04_severity_curves")


# ============================================================= Figure 5
def paired_ci(a, b, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = a.astype(float); b = b.astype(float); n = a.size
    pt = (a.mean() - b.mean()) * 100
    boot = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        boot[i] = (a[s].mean() - b[s].mean()) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt, pt - lo, hi - pt


def fig5():
    ch, sev, true, cols = load_merged()
    name = {4: "Darkening", 8: "Noise", 9: "Rain", 11: "Snow", 12: "Haze"}
    order = [4, 8, 9, 11, 12]
    va = (cols["pred_va_rule"] == true).astype(int)
    ad = (cols["pred_adair"] == true).astype(int)
    cid = (cols["pred_cidnet"] == true).astype(int)
    adp, adl, adh, cip, cil, cih, labels = ([] for _ in range(7))
    for c in order:
        m = ch == c
        p, lo, hi = paired_ci(ad[m], va[m]); adp.append(p); adl.append(lo); adh.append(hi)
        p, lo, hi = paired_ci(cid[m], va[m]); cip.append(p); cil.append(lo); cih.append(hi)
        labels.append(name[c])
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x - w / 2, adp, w, yerr=[adl, adh], capsize=5,
           color=C["adair"], label="AdaIR minus VA-Adaptive",
           edgecolor="#333", linewidth=0.8,
           error_kw={"elinewidth": 1.4, "capthick": 1.4})
    ax.bar(x + w / 2, cip, w, yerr=[cil, cih], capsize=5,
           color=C["cidnet"], label="HVI-CIDNet minus VA-Adaptive",
           edgecolor="#333", linewidth=0.8,
           error_kw={"elinewidth": 1.4, "capthick": 1.4})
    ax.axhline(0, color="#333333", linewidth=1.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Accuracy difference vs VA-Adaptive (pp)")
    ax.set_title("Per-challenge gap of learned restoration over routing")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.7)
    ax.legend(loc="lower left", fontsize=14)
    ax.text(0.985, 0.97, "above 0: learned method wins\nbelow 0: routing wins",
            transform=ax.transAxes, va="top", ha="right", fontsize=13,
            color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    fig.tight_layout()
    save(fig, "Fig05_boundary_bars")


# ============================================================= Figure 6
def fig6():
    ext = json.load(open(EXT))
    steps = [("Baseline", 52.56, C["passthrough"]),
             ("Learned router\n(logreg)", 56.04, "#b5651d"),
             ("VA-Adaptive", 57.32, C["va_rule"]),
             ("Learned router\n(MLP)", 58.05, "#7a9a3f"),
             ("AdaIR", 58.73, C["adair"]),
             ("Oracle-4\n(classical pool)", ext["oracle4"], C["oracle4"]),
             ("Oracle-6\n(+AdaIR, CIDNet)", ext["oracle6"], C["oracle6"]),
             ("Clean ceiling", ext["ceiling_cf"], "#333333")]
    labels = [s[0] for s in steps]; vals = [s[1] for s in steps]
    cols = [s[2] for s in steps]; x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.bar(x, vals, color=cols, edgecolor="#333", linewidth=0.8, width=0.64)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.6, f"{v:.2f}", ha="center", fontsize=14,
                fontweight="bold")
    ax.plot(x, vals, "-", color="#999999", linewidth=1.4, zorder=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("CURE-TSR degraded-average accuracy (%)")
    ax.set_ylim(48, 84)
    ax.set_title("Routing headroom: the bottleneck is selection, "
                 "not operator capacity")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.7)
    fig.tight_layout()
    save(fig, "Fig06_routing_ladder")


# ============================================================= Figure 7
def fig7():
    ch, sev, true, cols = load_merged()
    name = {4: "Darkening", 8: "Noise", 9: "Rain", 11: "Snow", 12: "Haze"}
    order = [4, 8, 9, 11, 12]
    branches = ["passthrough", "gamma", "clahe", "stretch"]
    frac = {b: [] for b in branches}
    for c in order:
        m = ch == c; rb = cols["rule_branch"][m]; n = m.sum()
        for b in branches:
            frac[b].append(100 * np.sum(rb == b) / n)
    x = np.arange(len(order)); bottom = np.zeros(len(order))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for b in branches:
        ax.bar(x, frac[b], bottom=bottom, color=C[b], label=LAB[b],
               edgecolor="white", linewidth=0.7, width=0.62)
        for xi, (f, bo) in enumerate(zip(frac[b], bottom)):
            if f > 6:
                ax.text(xi, bo + f / 2, f"{f:.0f}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if b in ("passthrough", "stretch")
                        else "#333333")
        bottom += np.array(frac[b])
    ax.set_xticks(x); ax.set_xticklabels([name[c] for c in order], fontsize=15)
    ax.set_ylabel("Share of images routed to branch (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Routing behavior on real CURE-TSR (pooled over severities)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              fontsize=14)
    fig.tight_layout()
    save(fig, "Fig07_routing_behavior")


# ============================================================= Figure 11
def fig11():
    ext = json.load(open(EXT)); L = json.load(open(LJS))["results"]
    P = ext["point_metrics"]; par = ext["params"]
    pts = [("Baseline", 0.01, P["passthrough"]["deg_acc"], 145291,
            C["passthrough"], (0, -26)),
           ("VA-Adaptive", L["va_rule"]["median_ms"], P["va_rule"]["deg_acc"],
            145291, C["va_rule"], (0, 16)),
           ("HVI-CIDNet", L["cidnet_enhance"]["median_ms"], P["cidnet"]["deg_acc"],
            par["CIDNet"] + 145291, C["cidnet"], (0, 18)),
           ("AdaIR", L["adair_enhance"]["median_ms"], P["adair"]["deg_acc"],
            par["AdaIR"] + 145291, C["adair"], (-58, 6))]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axhline(ext["oracle4"], color=C["oracle4"], linestyle=":", linewidth=2.0,
               label=f"Oracle-4 ceiling ({ext['oracle4']:.1f}%)")
    ax.axhline(P["passthrough"]["deg_acc"], color=C["passthrough"],
               linestyle="--", linewidth=1.3,
               label=f"Baseline ({P['passthrough']['deg_acc']:.1f}%)")
    for nm, lat, acc, prm, col, off in pts:
        size = 260 + 1500 * (np.log10(prm) - 5) / (np.log10(3e7) - 5)
        ax.scatter(lat, acc, s=max(size, 180), color=col, edgecolor="#222",
                   linewidth=1.0, zorder=5, alpha=0.9)
        ax.annotate(nm, (lat, acc), xytext=off, textcoords="offset points",
                    ha="center", fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Enhancement latency, median ms per image\n"
                  "(single CPU thread, batch size 1)")
    ax.set_ylabel("CURE-TSR degraded-average accuracy (%)")
    ax.set_title("Accuracy vs enhancement cost\n"
                 "(marker area grows with log parameter count)", fontsize=17)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim(51, 67)
    fig.tight_layout()
    save(fig, "Fig11_accuracy_cost")


FIGS = {1: fig1, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 11: fig11}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(FIGS)
    for n in want:
        if n in FIGS:
            FIGS[n]()
        else:
            print(f"no Fig. {n} here (available: {sorted(FIGS)})")
    print("done ->", FIGDIR)
