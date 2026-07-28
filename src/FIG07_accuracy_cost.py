# -*- coding: utf-8 -*-
"""
FIG07_accuracy_cost.py
Working figure 9, which the paper prints as its Figure 7. The two numberings are
kept apart on purpose: scripts and their outputs keep the label they were built
with, and the paper assigns its own numbers by order of first citation. The map
between them lives in _papercaps.json; renaming an output here would break it.

Paper figure 7: what each front end buys, what it costs, and what it charges on
images that were never degraded.

WHAT CHANGED FROM THE FIRST VERSION
  The latencies now come from N_timing_stable.results.json rather than from
  L2_timing_full_pool. L2 timed each crop once and reported the median of those
  single readings, which moved by a factor of 3.8 between two runs of the same
  seed on the same machine. The replacement times each crop five times and keeps
  the fastest, cycles the front ends crop by crop so none of them is measured
  while the processor is hot from another, and repeats the whole exercise twice
  so the disagreement between the two passes is visible. Every front end now
  agrees between passes to within 2.2 per cent except Zero-DCE, at 7.1.

  Three front ends that were absent can now be drawn. Zero-DCE, FFA-Net and
  PromptIR had accuracies but no latency, so the first version had nine points;
  this one has twelve. Their accuracies still come from the earlier run, which
  is why they carry no paired interval, but their latencies were measured here
  under the protocol above.

  Left. Degraded-average accuracy against measured front-end latency, log axis.
  The dark channel prior, a 0.206 ms classical operator, reaches 60.72 and is
  the most accurate front end in the pool, above AdaIR at 57.78 and above every
  other learned restorer. It is more accurate AND about a thousand times
  cheaper, so the learned models are dominated rather than merely expensive.

  Right. The same methods priced on clean images. Every learned model charges
  something there. The prior charges 0.07 points and the selector charges
  nothing at all, because on clean input the routing rule never leaves
  passthrough and its output is bit-identical to no enhancement.

TIMING PROTOCOL (method, and it belongs in the paper)
  Per crop, batch of one, a single CPU thread, at the original crop resolution.
  200 degraded crops drawn round-robin from all sixty (challenge, severity)
  cells, seed 42, ten leading crops discarded. Each crop is timed five times and
  the fastest kept; front ends under five milliseconds are timed twenty-five
  times, since one scheduling hiccup is a large fraction of a short reading. The
  front ends are cycled inside the crop loop so that each meets the same machine
  state. The whole run is repeated and the two passes compared; the figure
  quotes each number to the precision the two passes support.

READS   outputs_revision/N_timing_stable.results.json
        outputs_revision/cure_tsr_per_image_predictions_12class.csv
        outputs_revision/merged_per_image.csv
        outputs_revision/dcp_cure.csv
WRITES  outputs_revision/figures/fig07_accuracy_cost.png    (400 dpi)
        outputs_revision/figures/fig07_accuracy_cost.svg
RUN     python FIG07_accuracy_cost.py
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fig_style as fs

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

CELLS = [(c, s) for c in range(1, 13) for s in range(1, 6)]
GATE_PCT = 0.189          # frozen gate: branch in {clahe, stretch}, degraded share
TAU_VB = 0.475            # modal tau over the 50 folds of M_multiseed_cascade

SLATE = "#5B7FA6"
GREEN = "#2E7D4F"
PURPLE = "#8C5FBF"
GREY = "#8A8A8A"

# name, prediction key, latency scope, colour, marker, is_ours
# The latency scope names are the keys of N_timing_stable's `final` block, not
# L2's `results` block: the two files name the same measurements differently and
# silently reading the wrong one would put a front end at the wrong x.
METHODS = [
    ("stretch",        "stretch",   "stretch",    GREY,   "v", False),
    ("gamma",          "gamma",     "gamma",      GREY,   "^", False),
    ("CLAHE",          "clahe",     "clahe",      GREY,   "s", False),
    ("base rule",      "va_rule",   "rule",       fs.RED, "o", False),
    ("selector V-B",   "vb",        "vb",         SLATE,  "D", False),
    ("DCP",            "dcp",       "dcp",        GREEN,  "*", True),
    ("CIDNet",         "cidnet",    "cidnet",     PURPLE, "P", False),
    ("Zero-DCE",       "zero_dce",  "zero_dce",   PURPLE, "<", False),
    ("FFA-Net",        "ffa_net",   "ffa_net",    PURPLE, ">", False),
    ("PromptIR",       "promptir",  "promptir",   PURPLE, "p", False),
    ("AdaIR",          "adair",     "adair",      PURPLE, "X", False),
    ("cascade on V-B", "casc_vb",   None,         SLATE,  "h", False),
]

# There is no legacy pool any more. Zero-DCE, FFA-Net and PromptIR were scored
# on this machine, on the same crops as everything else, and their per-image
# predictions sit in the merged file beside the other nine. They take the same
# path and carry the same paired intervals. Under the old arrangement their
# accuracies were read from an aggregate the earlier sweep had recorded, which
# is how the article went on quoting 55.37, 55.36 and 50.97 for weeks after the
# rerun had made them 55.91, 55.24 and 50.93: the wording about an earlier run
# was deleted, the numbers behind it were not.

# Locked 2026-07-11 from L2_timing_full_pool.results.json and the per-image files.
# The figure asserts that the dark channel prior is the most accurate front end
# and that AdaIR trails it. Both are claims about a difference, so both carry a
# paired interval, computed analytically on the per-cell paired difference so it
# has no random seed. "above" is locked rather than the bound, which keeps the
# audit stable while still failing if a claim stops holding.
EXPECT_CI = {"dcp_over_runner_up": "above", "dcp_over_adair": "above"}

EXPECT = dict(
    acc={"passthrough": 54.49, "stretch": 50.41, "gamma": 52.57, "clahe": 46.41,
         "va_rule": 56.65, "vb": 58.23, "dcp": 60.72, "cidnet": 56.31,
         "adair": 57.78, "casc_vb": 58.39,
         "zero_dce": 50.93, "ffa_net": 55.91, "promptir": 55.24},
    cf={"passthrough": 80.77, "stretch": 77.29, "gamma": 79.81, "clahe": 66.05,
        "va_rule": 80.77, "vb": 80.77, "dcp": 80.70, "cidnet": 80.10,
        "adair": 79.51, "casc_vb": 80.77,
        "zero_dce": 78.85, "ffa_net": 79.51, "promptir": 78.99},
    ms={"stretch": 0.012, "gamma": 0.024, "clahe": 0.041,
        "rule": 0.058, "vb": 0.0543, "dcp": 0.206,
        "cidnet": 18.2, "zero_dce": 118.0, "ffa_net": 138.46,
        "adair": 210.39, "promptir": 219.83, "classifier": 1.21},
    casc_ms=39.82, ratio=3875)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS),
                    default="ieee1col")
    ap.add_argument("--timing",
                    default=str(OUT_DIR / "N_timing_stable.results.json"))
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    for p in (args.timing, args.merged, args.dcp):
        if not Path(p).exists():
            raise SystemExit(f"ABORT: {p} not found.")
    T = json.load(open(args.timing))
    # The guard names what this figure needs rather than what an older file
    # happened to contain. L2's file has a `results` block keyed op_dcp and no
    # figure for the three earlier-run models; feeding it here would silently
    # place five front ends at the wrong x, so it is refused by name.
    if "final" not in T:
        raise SystemExit("ABORT: the timing file has no `final` block. This is "
                         "not N_timing_stable.results.json. L2's file reports "
                         "single-pass medians that moved by a factor of 3.8 "
                         "between runs and may not be used here.")
    F_ = T["final"]
    need = [m[2] for m in METHODS if m[2]]
    absent = [k for k in need if k not in F_]
    if absent:
        raise SystemExit(f"ABORT: the timing file is missing {', '.join(absent)}")
    unstable = [k for k in need if F_[k].get("quote_ms") is None]
    if unstable:
        raise SystemExit(f"ABORT: {', '.join(unstable)} were too unstable to "
                         "quote; re-run the timing on a quieter machine")
    # The three earlier-run models are scored from their per-image predictions,
    # not from a stored aggregate. There are two versions of the aggregate file
    # in circulation, one covering five challenges and one covering twelve, and
    # reading the wrong one understates these three by several points without
    # any sign that anything is wrong. Counting from the predictions removes
    # the choice: the file either has the twelve challenges or it does not, and
    # the count below says which.
    CH_NAME = {"Decolorization", "LensBlur", "CodecError", "Darkening",
               "DirtyLens", "Exposure", "GaussianBlur", "Noise", "Rain",
               "Shadow", "Snow", "Haze"}
    rows = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    dmap = {}
    for r in csv.DictReader(open(args.dcp, newline="", encoding="utf-8")):
        dmap[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = \
            int(r["pred_dcp"])

    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    br = np.array([r["rule_branch"] for r in rows])
    pa = np.array([float(r["prob_adair"]) for r in rows])
    P = {k: np.array([int(r["pred_" + k]) for r in rows])
         for k in ["passthrough", "gamma", "clahe", "stretch", "va_rule",
                   "adair", "cidnet", "zero_dce", "ffa_net", "promptir"]}
    P["dcp"] = np.array([dmap[(r["filename"], int(r["occ"]), int(r["ch"]),
                               int(r["sev"]))] for r in rows])
    P["vb"] = np.where(br == "clahe", P["dcp"], P["va_rule"])
    gate = np.isin(br, ("clahe", "stretch"))
    P["casc_vb"] = np.where(gate & (pa > TAU_VB), P["adair"], P["vb"])

    masks = [(ch == c) & (sev == s) for c, s in CELLS]
    cf_m = ~np.isin(ch, range(1, 13))

    def deg(p):
        return float(np.mean([100.0 * np.mean(p[m] == tru[m]) for m in masks]))

    def cf(p):
        return 100.0 * float(np.mean(p[cf_m] == tru[cf_m]))

    # Latency comes from the two-pass figure the timing script says is safe to
    # quote, not from either pass on its own.
    med = {k: v["quote_ms"] for k, v in F_.items()}

    # Every front end goes through the same two functions. There is no second
    # path for a subset read from an aggregate.
    def deg_of(key):
        return deg(P[key])

    def cf_of(key):
        return cf(P[key])

    base_deg, base_cf = deg(P["passthrough"]), cf(P["passthrough"])
    casc_ms = med["vb"] + GATE_PCT * med["adair"]

    # ---------------- audit ----------------
    print("=== EXPECTED-OUTPUT AUDIT (against values locked 2026-07-11) ===")
    bad = 0
    for k, e in EXPECT["acc"].items():
        got = deg_of(k)
        if abs(round(got, 2) - e) > 1e-9:
            print(f"  MISMATCH deg-avg {k}: got {got:.4f} (prints {got:.2f}), "
                  f"expected {e:.2f}")
            bad += 1
    for k, e in EXPECT["cf"].items():
        got = cf_of(k)
        if abs(round(got, 2) - e) > 1e-9:
            print(f"  MISMATCH clean {k}: got {got:.4f}, expected {e:.2f}")
            bad += 1
    for k, e in EXPECT["ms"].items():
        if abs(med[k] - e) > 1e-9:
            print(f"  MISMATCH latency {k}: got {med[k]}, expected {e}")
            bad += 1
    if abs(round(casc_ms, 2) - EXPECT["casc_ms"]) > 0.011:
        print(f"  MISMATCH cascade cost: got {casc_ms:.2f}, "
              f"expected {EXPECT['casc_ms']:.2f}")
        bad += 1
    ratio = med["adair"] / med["vb"]
    if abs(round(ratio) - EXPECT["ratio"]) > 1:
        print(f"  MISMATCH AdaIR/V-B ratio: got {ratio:.0f}, "
              f"expected {EXPECT['ratio']}")
        bad += 1
    # the clean-image identity: on clean the rule and V-B are the same code path
    if abs(cf(P["vb"]) - base_cf) > 1e-9:
        print(f"  MISMATCH: V-B is not bit-identical to no enhancement on clean "
              f"({cf(P['vb']):.2f} against {base_cf:.2f}).")
        bad += 1
    print(f"  V-B {deg(P['vb']):.2f} at {med['vb']:.4f} ms; "
          f"AdaIR {deg(P['adair']):.2f} at {med['adair']:.2f} ms; "
          f"V-B is more accurate and {ratio:.0f} times cheaper")
    print(f"  cascade on V-B: {deg(P['casc_vb']):.2f} at {casc_ms:.1f} ms, "
          f"{casc_ms / med['vb']:.0f} times V-B, for "
          f"{deg(P['casc_vb']) - deg(P['vb']):+.2f} points")
    print(f"  on clean, V-B costs {cf(P['vb']) - base_cf:+.2f} points against "
          f"{cf(P['adair']) - base_cf:+.2f} for AdaIR and "
          f"{cf(P['dcp']) - base_cf:+.2f} for DCP")
    # ---- the two difference claims the figure makes in words ----
    cells = [np.where((ch == c) & (sev == s))[0]
             for c in range(1, 13) for s in range(1, 6)]

    def paired(a, b):
        """Analytic 95 per cent interval on the per-cell paired difference.

        The statistic is the mean over cells of the mean paired difference
        inside a cell, so its variance is the sum of the within-cell variances
        over the squared number of cells. No resampling, so no seed.
        """
        k = len(cells)
        means, var = [], 0.0
        for i in cells:
            d = ((a[i] == tru[i]).astype(np.float64)
                 - (b[i] == tru[i]).astype(np.float64))
            means.append(d.mean())
            var += d.var(ddof=1) / len(i)
        pt = 100.0 * float(np.mean(means))
        half = 1.96 * 100.0 * float(np.sqrt(var)) / k
        return pt, pt - half, pt + half

    # Every front end is in the pool. The runner-up is whichever of them is in
    # fact second, not the best of a subset chosen for having per-image
    # predictions, which all twelve now have.
    pool = ["passthrough", "gamma", "clahe", "stretch", "dcp", "vb", "va_rule",
            "adair", "cidnet", "zero_dce", "ffa_net", "promptir"]
    runner_up = max((o for o in pool if o != "dcp"), key=lambda o: deg(P[o]))
    CI = {}
    for name, other in (("dcp_over_runner_up", runner_up),
                        ("dcp_over_adair", "adair")):
        d, lo, hi = paired(P["dcp"], P[other])
        CI[other] = (d, lo, hi)
        got = "above" if lo > 0 else ("below" if hi < 0 else "level")
        if got != EXPECT_CI[name]:
            print(f"  MISMATCH {name}: interval says {got}, "
                  f"expected {EXPECT_CI[name]}")
            bad += 1
        print(f"  DCP over {other:11s} {d:+.2f}  95 per cent CI "
              f"[{lo:+.2f}, {hi:+.2f}]  -> {got}")
    print("  AUDIT PASSED: every value reproduces." if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    # The figure is a full-width (two-column-spanning) float. Bump the base font
    # for readability at that width, and use a more landscape aspect so the same
    # text reads larger relative to the panel and the float uses less page height.
    F = F * 1.20 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    H = 5.4 if args.target == "ieee" else (6.87 if ONE else 5.05)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    # The gap between the panels has to hold the right panel's names and the
    # column of markers beside them. At the old spacing that block was wider
    # than the gap, so it reached into the left panel and collided with whatever
    # label sat near its right edge. The gap is widened to fit the block; the
    # left panel gives up the width, which it can afford, since its data span
    # five decades on a log axis.
    if ONE:
        # Stacked, and each panel placed on its own. Side by side the two would
        # have about an inch of drawing area each, and the upper one carries a
        # log axis over five decades: at that width the tick labels 0.01 to 1000
        # run into one another before a single point is placed.
        #
        # The two do not share a left margin either. The lower panel spends 1.19
        # inch on its front-end names; the upper spends 0.40 on a rotated axis
        # label and two-digit ticks. Aligning their axes would leave three
        # quarters of an inch blank beside the upper panel and take the same
        # width off the scatter, which is where the crowding is.
        axL = fig.add_axes([0.115, 0.6790, 0.958 - 0.115, 0.2508])
        axR = fig.add_axes([0.314, 0.2690, 0.958 - 0.314, 0.2820])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.00], left=0.088,
                              right=0.985, top=0.9074, bottom=0.3000,
                              wspace=0.78)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1])
    # The right panel's names each carry a marker, so they need room between the
    # text and the axis. The pad is set here, before anything is placed, because
    # changing it later re-flows the whole figure: the left panel narrows, its
    # labels keep their data coordinates while their glyphs keep their size, and
    # two that cleared each other by half a pixel stop clearing.
    # The pad is in points: F * 2.6 comes to forty of them, over half an inch, which
    # is room the two-column layout has beside its right panel and a column does not.
    axR.tick_params(axis="y", length=0, pad=F * (1.45 if ONE else 2.6))
    AXL = []

    # The right panel's names are set up first, before a single label is placed
    # on the left. They sit in the gap between the panels, and the left panel's
    # labels may legitimately reach the same gap; whichever is drawn second
    # cannot see the other. Placing them now means the left panel's placer can
    # treat them, and the column of markers beside them, as obstacles like any
    # other. Tuning the spacing until it happens to clear on one machine is not
    # a fix: the two collided on a machine whose font metrics differ from the
    # one the figure was checked on.
    _keymap = {n: k for n, k, *_ in METHODS}
    _named = [n for n, *_ in METHODS if n in _keymap]
    _order0 = sorted(_named, key=lambda n: -(cf_of(_keymap[n]) - base_cf))
    _ys0 = np.arange(len(_order0))
    axR.set_yticks(_ys0)
    axR.set_yticklabels(_order0, fontsize=F * (1.02 if ONE else 0.66))
    axR.invert_yaxis()
    # Tight to the data and the two label widths, not a round number. At (-21, 4.6)
    # the bars used 57 per cent of the panel and the rest was blank on both
    # sides of them.
    axR.set_xlim(*((-18.2, 2.9) if ONE else (-21.0, 4.6)))
    fig.canvas.draw()

    # ---- left: accuracy against measured cost ----
    axL.axhline(base_deg, color="#BBBBBB", lw=1.2, ls=(0, (4, 2)), zorder=2)
    # The frontier climbs from 0.028 to 0.312 and passes straight through the
    # left-hand end of this baseline, so the label goes to the empty stretch on
    # the right, below the line.
    # Anchored to the panel's right edge in axes fraction, not to a value on
    # the axis. Placed by value it kept its data coordinate while the panel
    # narrowed, ran past the edge and collided with the right panel's tick
    # labels, which a reader would have seen as two captions touching.
    _blend = matplotlib.transforms.blended_transform_factory(
        axL.transAxes, axL.transData)
    axL.text(0.985, base_deg - 0.55, "no enhancement (costs nothing)",
             transform=_blend, fontsize=F * 0.66, color=fs.MUTED,
             va="top", ha="right", zorder=6)

    pts = {}
    for name, key, scope, col, mk, ours in METHODS:
        x = casc_ms if scope is None else med[scope]
        y = deg_of(key)
        pts[name] = (x, y)
        axL.plot([x], [y], marker=mk, ms=F * (1.30 if ours else 0.70),
                 color=col, mec="white", mew=0.9 if ours else 0.6,
                 zorder=8 if ours else 6, ls="none")

    # the frontier: nothing is both cheaper and more accurate than these
    front = sorted([(x, y, n) for n, (x, y) in pts.items()])
    keep, best = [], -1e9
    for x, y, n in front:
        if y > best:
            keep.append((x, y))
            best = y
    axL.plot([p[0] for p in keep], [p[1] for p in keep], color="#CCCCCC", lw=1.2,
             ls="-", zorder=3)

    # DCP dominates every learned model: the arrow runs from DCP to the most
    # accurate of them, AdaIR, showing more accuracy at a fraction of the cost.
    # AdaIR is not the dearest of the five, PromptIR is, and the figure shows
    # that plainly by placing PromptIR further right; calling AdaIR the dearest
    # would have set the caption against the picture.
    axL.annotate("", xy=(pts["AdaIR"][0], pts["AdaIR"][1]),
                 xytext=(pts["DCP"][0], pts["DCP"][1]),
                 arrowprops=dict(arrowstyle="->", color=fs.RED, lw=1.5,
                                 shrinkA=10, shrinkB=8,
                                 connectionstyle="arc3,rad=-0.28"), zorder=7)

    axL.set_xscale("log")
    # The right end runs a little past the last point so that AdaIR, at 210 ms,
    # has somewhere to put its label. It is not set generously: a wider axis
    # spreads the points out and starves the middle of the panel, which is what
    # cost CIDNet its label until this was tightened.
    axL.set_xlim(0.009, 1200)
    # The labels are derived from the tick positions. They used to be a separate
    # list, and when the axis was widened by a decade the ticks moved while the
    # labels did not: a tick drawn at 0.01 carried the text 0.02, so every
    # position on the axis read high by a factor of two.
    XT = [0.01, 0.1, 1, 10, 100, 1000]
    axL.set_xticks(XT)
    axL.set_xticklabels([("%g" % v) for v in XT])
    axL.minorticks_off()
    axL.set_ylim(44.0, 63.2)
    axL.set_yticks([46, 50, 54, 58, 62])
    axL.tick_params(labelsize=F * 0.74, length=2.5, pad=1.5)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axL.spines[sp].set_color("#999999")
    axL.set_xlabel(("latency per image\n(ms, log scale)" if ONE else
     "front-end latency per image (ms, log scale)"),
                   fontsize=F * 0.76, labelpad=2)
    axL.set_ylabel("degraded-average accuracy (%)", fontsize=F * 0.76, labelpad=2)
    axL.set_title("what each front end buys,\nand what it costs",
                  fontsize=F * (1.04 if ONE else 0.86), pad=5,
                  linespacing=1.2)
    AXL.append(axL)

    # Labelling. Nine names will not fit: the base rule and V-B sit at 0.098 and
    # 0.116 ms, a factor of 1.18 apart on a log axis three inches wide. But the
    # three fixed operators do not need names here, because their message is that
    # they are ALL BELOW the no-enhancement line, and that is a single statement.
    # So they get one group label, and the six that carry an argument get their own.
    # Twelve points do not all take an adjacent label on a log axis. Those that
    # fit get one; those that do not are named in a small key instead. The
    # placer never overlaps anything, and it no longer aborts the run for want
    # of a position: a figure with a key is readable, a figure with two labels
    # on top of each other is not.
    # Placement is first come, first served, so the list is in order of how much
    # the argument needs the label next to its point. The article turns on the
    # prior against AdaIR, and the red arrow joins exactly those two, so a run
    # that pushed AdaIR into the key left the arrow pointing at an unnamed
    # marker. Those two lead; the routers and the operators follow; the three
    # that can be read from the key without loss come last.
    # With the right panel now showing each name beside its marker, the left
    # panel does not have to name everything. It names the points the argument
    # rests on and the ones a reader will look for first; the rest are one
    # glance away, and the panel stays legible.
    LABEL = ["DCP", "AdaIR", "selector V-B", "base rule", "CIDNet",
             "Zero-DCE", "gamma", "CLAHE", "stretch"]
    # Point labels are kept short so they fit the crowded low-latency corner; the
    # legend and caption carry the full names.
    SHORT = {"selector V-B": "V-B", "cascade on V-B": "cascade",
             "base rule": "base rule"}
    GROUP = ["stretch", "gamma", "CLAHE"]

    gx = float(np.exp(np.mean([np.log(pts[n][0]) for n in GROUP])))
    gy = min(pts[n][1] for n in GROUP)
    # left-anchored and nudged right so the first character clears the axis edge
    # The group annotation that used to sit here is gone. Each of the three
    # fixed operators now carries its own label, and the point the annotation
    # made, that all of them sit below the no-enhancement line, is visible in
    # the figure and stated in the caption. Its space is better spent on the
    # labels and the key.

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_bb = axL.get_window_extent(renderer=rend)
    MARK = {}
    for nm, (x, y) in pts.items():
        px, py = axL.transData.transform((x, y))
        # The obstacle must be the size of the marker actually drawn. It was a
        # fixed F * 0.95 in pixels, about fifteen, while a marker drawn at
        # F * 0.70 points spans nearly sixty pixels at this resolution: the
        # placer believed every marker was half its real size and could put a
        # label over one without noticing.
        _ms_pt = F * (1.30 if nm == "DCP" else 0.70)
        r = 0.5 * _ms_pt * fig.dpi / 72.0 + 2.0
        MARK[nm] = matplotlib.transforms.Bbox([[px - r, py - r],
                                               [px + r, py + r]])
    placed = []
    for t in (axL.get_xticklabels() + axL.get_yticklabels() + [axL.title] +
              [c for c in axL.texts if c.get_text()] +
              list(axR.get_yticklabels())):
        placed.append(t.get_window_extent(renderer=rend))
    # The marker column has no artist yet, so reserve the band it will occupy.
    _rbb = axR.get_window_extent(renderer=rend)
    _mr = F * 0.62 * fig.dpi / 72.0
    placed.append(matplotlib.transforms.Bbox(
        [[_rbb.x0 - F * 1.05 * fig.dpi / 72.0 - _mr, _rbb.y0],
         [_rbb.x0, _rbb.y1]]))

    CAND = [(0, 1.0, "center", "bottom"), (0, -1.0, "center", "top"),
            (1, 0.0, "left", "center"), (-1, 0.0, "right", "center"),
            (1, 0.8, "left", "bottom"), (-1, 0.8, "right", "bottom"),
            (1, -0.8, "left", "top"), (-1, -0.8, "right", "top"),
            (0, 1.9, "center", "bottom"), (0, -1.9, "center", "top"),
            (1.9, 0.0, "left", "center"), (-1.9, 0.0, "right", "center"),
            (-1.9, 1.4, "right", "bottom"), (-1.9, -1.4, "right", "top"),
            (1.9, 1.4, "left", "bottom"), (1.9, -1.4, "left", "top"),
            (-2.8, 0.0, "right", "center"), (0, 2.8, "center", "bottom"),
            (0, -2.8, "center", "top"), (-2.8, 0.8, "right", "bottom"),
            # Far candidates. They were pointless before, since a label this far
            # from its marker is unreadable on its own; with a leader line drawn
            # whenever the gap is large, they are usable, and they are what keeps
            # a crowded corner from pushing a headline label into the key.
            (2.8, 1.6, "left", "bottom"), (-2.8, 1.6, "right", "bottom"),
            (2.8, -1.6, "left", "top"), (-2.8, -1.6, "right", "top"),
            (0, 4.0, "center", "bottom"), (0, -4.0, "center", "top"),
            (-4.0, 0.0, "right", "center"), (-4.0, 2.2, "right", "bottom"),
            (-4.0, -2.2, "right", "top"), (2.8, 3.2, "left", "bottom"),
            (-2.8, 3.2, "right", "bottom")]
    # The offset must clear the point's OWN marker, whose half-width is F * 0.95.
    PAD_X, PAD_Y = F * 1.35, F * 1.35

    # The placer must also know about the frontier line, or a label lands on it and
    # the fifth gate catches what the placer should have prevented.
    FRONT_PX = [axL.transData.transform(pp) for pp in keep]
    SEGS = list(zip(FRONT_PX[:-1], FRONT_PX[1:]))

    # The red arrow must be avoided as a curve, not as a box. It was reaching
    # the placer as an annotation, whose bounding box is the rectangle that
    # encloses the whole arc; that rectangle covers most of the panel and, in
    # particular, everything around the point the arrow ends at, so the label
    # for AdaIR could never find a spot near its own marker. Sampling the arc
    # and adding it to the line test blocks the ink and nothing else.
    _a0 = np.array(axL.transData.transform(pts["DCP"]), dtype=float)
    _a1 = np.array(axL.transData.transform(pts["AdaIR"]), dtype=float)
    _mid = 0.5 * (_a0 + _a1)
    _perp = np.array([-(_a1 - _a0)[1], (_a1 - _a0)[0]])
    _ctrl = _mid + 0.28 * _perp          # matches connectionstyle rad
    _t = np.linspace(0.0, 1.0, 24)[:, None]
    _arc = ((1 - _t) ** 2) * _a0 + 2 * (1 - _t) * _t * _ctrl + (_t ** 2) * _a1
    SEGS += list(zip([tuple(q) for q in _arc[:-1]],
                     [tuple(q) for q in _arc[1:]]))

    # The no-enhancement baseline runs the width of the panel and was never on
    # this list, so nothing stopped a label from landing across it: CIDNet's sat
    # with the dashes running through the middle of the word. A line a reader is
    # meant to read a value against must be kept clear like any other.
    _b0 = axL.transData.transform((axL.get_xlim()[0], base_deg))
    _b1 = axL.transData.transform((axL.get_xlim()[1], base_deg))
    SEGS.append((tuple(_b0), tuple(_b1)))

    def hits_line(bb):
        for (x0, y0), (x1, y1) in SEGS:
            dx, dy = x1 - x0, y1 - y0
            t0, t1 = 0.0, 1.0
            for pq in ((-dx, x0 - bb.x0), (dx, bb.x1 - x0),
                       (-dy, y0 - bb.y0), (dy, bb.y1 - y0)):
                pp, qq = pq
                if pp == 0:
                    if qq < 0:
                        break
                else:
                    r = qq / pp
                    if pp < 0:
                        t0 = max(t0, r)
                    else:
                        t1 = min(t1, r)
            else:
                if t0 <= t1:
                    return True
        return False

    def clashes(bb, own):
        # The placer measures during layout and the gate measures the finished
        # render; the two differ by a fraction of a pixel, which is enough for a
        # pair the placer thought was clear to fail the gate by four tenths of a
        # pixel. Reserving four pixels rather than two absorbs that difference.
        bb = bb.expanded(1.0, 1.0)
        bb = matplotlib.transforms.Bbox([[bb.x0 - 4, bb.y0 - 4],
                                         [bb.x1 + 4, bb.y1 + 4]])
        if not ax_bb.fully_contains(bb.x0, bb.y0) or \
                not ax_bb.fully_contains(bb.x1, bb.y1):
            return True
        others = [b for n, b in MARK.items() if n != own]
        if any(bb.overlaps(o) for o in placed + others):
            return True
        return hits_line(bb)

    # A probe is placed in DISPLAY pixels to test the candidate, then removed, and
    # the winner is re-placed in DATA coordinates. This matters: the renderer used
    # for the collision test runs at the figure's own dpi, but savefig re-renders
    # at 400. A text pinned to display pixels keeps its raw pixel numbers through
    # that change and lands in the corner of the saved file, while every gate,
    # which questions the same low-dpi renderer, reports the figure as clean.
    # transData is recomputed at save time and is immune.
    inv = axL.transData.inverted()
    spilled = []
    # A label belongs beside its point. A short leader is acceptable when the
    # neighbourhood is full; a long one is not, because the reader then has to
    # trace a line across the panel, and two long ones that cross are worse
    # again. So candidates are tried nearest first, a leader longer than the cap
    # is refused, and a leader that would cross one already drawn is refused.
    # Anything left over is named in a small key, which costs one glance rather
    # than a trace.
    LEADER_CAP = 4.3 * PAD_X

    def seg_cross(a, b, c, d):
        def side(p, q, r):
            return ((q[0] - p[0]) * (r[1] - p[1]) -
                    (q[1] - p[1]) * (r[0] - p[0]))
        d1, d2 = side(c, d, a), side(c, d, b)
        d3, d4 = side(a, b, c), side(a, b, d)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    leaders = []
    order_by_reach = sorted(CAND, key=lambda c: (c[0] ** 2 + c[1] ** 2) ** 0.5)
    for name in LABEL:
        x, y = pts[name]
        px, py = axL.transData.transform((x, y))
        for ox, oy, ha, va in order_by_reach:
            tx, ty = px + ox * PAD_X, py + oy * PAD_Y
            label_txt = SHORT.get(name, name)
            probe = axL.text(tx, ty, label_txt, fontsize=F * (0.72 if ONE else 0.68), ha=ha, va=va,
                             transform=None)
            bb = probe.get_window_extent(renderer=rend)
            probe.remove()
            if clashes(bb, name):
                continue
            cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
            gap = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if gap > LEADER_CAP:
                continue
            needs_leader = gap > 2.4 * PAD_X
            if needs_leader:
                ex = bb.x1 if px > cx else bb.x0
                ey = bb.y0 if py < cy else bb.y1
                if any(seg_cross((ex, ey), (px, py), u, v) for u, v in leaders):
                    continue
            dx, dy = inv.transform((tx, ty))
            axL.text(dx, dy, label_txt, fontsize=F * (0.72 if ONE else 0.68),
                     color=fs.INK,
                     ha=ha, va=va, zorder=9)
            placed.append(bb)
            if needs_leader:
                sx, sy = inv.transform((ex, ey))
                axL.annotate("", xy=(x, y), xytext=(sx, sy),
                             arrowprops=dict(arrowstyle="-", lw=0.6,
                                             color=fs.FAINT,
                                             shrinkA=1.0, shrinkB=6.0),
                             zorder=6)
                leaders.append(((ex, ey), (px, py)))
            break
        else:
            spilled.append(name)


    print(f"  {len(LABEL) - len(spilled)} of {len(LABEL)} labels sit beside "
          f"their point, {len(leaders)} of them with a short leader; the other "
          f"{len(METHODS) - len(LABEL)} front ends are named in the right panel "
          f"beside the marker they wear.")

    # There is no key. Every front end is named in the right panel beside the
    # marker it wears, so a point whose label will not fit beside it is still
    # one glance from being identified. A box floating in the panel, listing a
    # method or two, would cover data to repeat what is already on the page.
    if spilled:
        print(f"  named in the right panel only: {', '.join(spilled)}")

    # ---- right: what each one charges on images that were never degraded ----
    # The bars are sorted by what they cost, not by a list typed out by hand.
    # The hand-written order had PromptIR at -1.70 above FFA-Net at -1.48 above
    # AdaIR at -1.26, which reads as a ranking and is not one.
    keymap = {n: k for n, k, *_ in METHODS}
    named = [n for n, *_ in METHODS if n in keymap]
    order = sorted(named, key=lambda n: -(cf_of(keymap[n]) - base_cf))
    dcf = [cf_of(keymap[n]) - base_cf for n in order]
    assert all(dcf[i] >= dcf[i + 1] - 1e-9 for i in range(len(dcf) - 1)), \
        "the clean-cost bars are not in order"
    ys = np.arange(len(order))
    cols = [SLATE if n in ("selector V-B", "base rule", "cascade on V-B")
            else fs.RED for n in order]
    axR.barh(ys, dcf, height=0.62, color=cols, zorder=3)
    axR.axvline(0, color="#999999", lw=1.0, zorder=4)
    for y, v in zip(ys, dcf):
        axR.text(v - 0.35 if v < -0.1 else 0.35, y,
                 # U+2212 MINUS SIGN, which is what the axis ticks use; an
                 # ASCII hyphen here put two different minus signs on one figure.
                 "0.00" if abs(v) < 1e-9 else f"{v:.2f}".replace("-", "\u2212"),
                 va="center", ha="right" if v < -0.1 else "left",
                 fontsize=F * (0.78 if ONE else 0.62), color=fs.INK,
                 zorder=6)
    # The axis was already inverted when these names were set up as obstacles
    # for the left panel. Inverting again would undo it, which is what turned
    # the ranking upside down: the costliest front end led the list instead of
    # the cheapest. The order is asserted rather than recomputed silently.
    assert order == _order0, "the two orderings of the right panel disagree"
    axR.set_yticks(ys)
    axR.set_yticklabels(order, fontsize=F * 0.66)

    # Tight to the data and the two label widths, not a round number. At (-21, 4.6)
    # the bars used 57 per cent of the panel and the rest was blank on both
    # sides of them.
    axR.set_xlim(*((-18.2, 2.9) if ONE else (-21.0, 4.6)))
    axR.set_xticks([-15, -10, -5, 0])
    axR.tick_params(axis="x", labelsize=F * 0.74, length=2.5, pad=1.5)
    # pad 6 rather than 3: the U+2212 sign is wider than the ASCII hyphen it
    # replaced, and the longest value label, minus 14.72, came within a pixel
    # of the CLAHE tick. Moving the ticks out is stabler than squeezing the
    # value labels toward the bars.
    # Room for a marker between each name and the axis, so the two panels can
    # be read as one: every name here wears the symbol it wears on the left.
    fig.canvas.draw()
    _style = {n: (c, m) for n, _k, _s, c, m, _o in METHODS}
    _axbb = axR.get_window_extent(renderer=rend)
    _mx = _axbb.x0 - F * 1.05 * fig.dpi / 72.0
    _inv = fig.transFigure.inverted()
    _mboxes = []
    for _lab, _y in zip(axR.get_yticklabels(), ys):
        _n = _lab.get_text()
        if _n not in _style:
            continue
        _col, _mk = _style[_n]
        _bb = _lab.get_window_extent(renderer=rend)
        _fx, _fy = _inv.transform((_mx, 0.5 * (_bb.y0 + _bb.y1)))
        fig.add_artist(Line2D([_fx], [_fy], marker=_mk, ms=F * 0.62,
                              color=_col, mec="white", mew=0.5, ls="none",
                              transform=fig.transFigure, zorder=6))
        _r = F * 0.62 * fig.dpi / 72.0 * 0.5
        _mboxes.append((_n, matplotlib.transforms.Bbox(
            [[_mx - _r, 0.5 * (_bb.y0 + _bb.y1) - _r],
             [_mx + _r, 0.5 * (_bb.y0 + _bb.y1) + _r]]), _bb))
    # The layout gate compares text with text and would not have caught a marker
    # sitting on a name, which is exactly what happened when these were drawn
    # before the axis limits were fixed. Check it here, where it can be seen.
    fig.canvas.draw()
    _worst = None
    for _n, _mb, _tb in _mboxes:
        _tb = _lab_bb = axR.get_yticklabels()[
            [t.get_text() for t in axR.get_yticklabels()].index(_n)
        ].get_window_extent(renderer=rend)
        _gap = _mb.x0 - _tb.x1
        if _worst is None or _gap < _worst[1]:
            _worst = (_n, _gap)
        if _mb.overlaps(_tb) or _gap < 1.0:
            raise SystemExit(
                f"ABORT: the marker for {_n} sits on its name in the right "
                f"panel (gap {_gap:.1f} px). Increase the tick pad.")
    if _worst:
        print(f"  right panel: every name clears its marker; the tightest is "
              f"{_worst[0]} at {_worst[1]:.1f} px.")
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.spines["bottom"].set_color("#999999")
    axR.set_xlabel("clean accuracy given up (points)" if ONE else
                   "clean accuracy\ngiven up (points)",
                   fontsize=F * (0.92 if ONE else 0.76),
                   labelpad=2, linespacing=1.2)
    axR.set_title("what it charges on\nclean images",
                  fontsize=F * (1.04 if ONE else 0.86), pad=5,
                  linespacing=1.2)

    # Every line below is kept inside the 7.16 in canvas. A note wider than the
    # figure is a note that gets clipped at both ends.
    h = [Line2D([], [], color=GREEN, lw=0, marker="*", ms=F * 1.10, mec="white",
                mew=0.8),
         Line2D([], [], color="#BBBBBB", lw=1.2, ls=(0, (4, 2)))]
    # These strings are built from the measured numbers rather than typed. The
    # first draft carried 0.24 ms and 850 times long after both had changed,
    # because a number written into a caption does not move when the data does.
    fig.legend(h, [f"dark channel prior ({med['dcp']:.3f} ms classical)",
                   "no enhancement (zero cost)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.1490 if ONE else 0.1296),
               ncol=1 if ONE else 2,
               frameon=False, fontsize=F * (0.92 if ONE else 0.74),
               handlelength=2.0,
               handletextpad=0.55, columnspacing=2.4)

    # At column width these three run past both edges. Each is broken where it
    # divides naturally rather than set in smaller type: the finding is what a
    # reader takes from the figure and should not be the smallest thing on it.
    fig.text(0.5, 0.1150 if ONE else 0.1019,
             f"The {med['dcp']:.3f} ms dark channel prior\nis the most "
             f"accurate front end here." if ONE else
             f"The {med['dcp']:.3f} ms dark channel prior is the most accurate "
             f"front end here.",
             ha="center", va="center", fontsize=F * (1.02 if ONE else 0.72),
             color=fs.RED, linespacing=1.25)
    fig.text(0.5, 0.0600 if ONE else 0.0667,
             f"AdaIR (ICLR 2025) trails it by\n"
             f"{deg_of('dcp') - deg_of('adair'):.1f} points at "
             f"{med['adair'] / med['dcp']:,.0f} times the cost." if ONE else
             f"AdaIR (ICLR 2025) trails it by "
             f"{deg_of('dcp') - deg_of('adair'):.1f} points at "
             f"{med['adair'] / med['dcp']:,.0f} times the cost.",
             ha="center", va="center", fontsize=F * (1.05 if ONE else 0.74),
             color=fs.INK, linespacing=1.25)
    # Two repeat counts, not one: the protocol times a crop five times, and
    # twenty-five times when the front end runs under five milliseconds. The
    # headline point of this figure is the prior at 0.206 ms, which is in the
    # second group, so a note saying only "five" describes the wrong protocol
    # for the number the reader is looking at.
    fig.text(0.5, 0.0170 if ONE else 0.0315,
             "fastest of five timings, 25 under 5 ms, one thread" if ONE else
             "Fastest of five timings per crop, twenty-five for front ends "
             "under 5 ms, one CPU thread, original crops.",
             ha="center", va="center", fontsize=F * (0.84 if ONE else 0.66),
             color=fs.MUTED)

    ok = fs.run_gates(fig, args.outdir, "fig07_accuracy_cost", bar_axes=[axR],
                      line_axes=AXL)
    # The multipliers here have to be the multipliers used above. Twice now a
    # figure has reported a size it was not drawing, because the drawing code
    # was changed and this dictionary was not, and a report that does not match
    # what is on the page is worse than no report.
    fs.report_sizes(args.target, W, F, {
        "panel titles": 1.04, "point labels": 0.72, "method labels": 1.02,
        "bar values": 0.78, "the key": 0.92,
        "the first finding": 1.02, "the second finding": 1.05,
        "the protocol note": 0.84, "axis labels ": 0.92} if ONE else {
        "panel titles": 0.86, "point labels": 0.68, "method labels": 0.72,
        "bar values": 0.70, "axis labels": 0.76, "the key": 0.74,
        "the findings": 0.74})

    print("\nCAPTION:")
    # Built from the measured values. A caption typed by hand goes stale the
    # moment the data moves, and says so to nobody.
    _d, _a = deg_of("dcp"), deg_of("adair")
    _casc, _vb = deg_of("casc_vb"), deg_of("vb")
    for ln in [
        f"Fig. 7. What each front end buys, what it costs, and what it charges on images",
        f"that were never degraded. Left: degraded-average accuracy against front-end",
        f"latency, log axis. The dark channel prior, a {med['dcp']:.3f} ms classical operator,",
        f"reaches {_d:.2f} per cent and is the most accurate front end in the pool: more",
        f"accurate than AdaIR ({_a:.2f} per cent at {med['adair']:.0f} ms) and than every other",
        f"learned restorer, at about a thousandth of AdaIR's cost, so the learned",
        f"models are dominated rather than merely expensive; the margin over AdaIR is",
        f"{_d - _a:.2f} points with an analytic paired interval of "
        f"{CI['adair'][1]:.2f} to {CI['adair'][2]:.2f}, and over the",
        f"runner-up, the selector, {_d - _vb:.2f} with an interval of {CI[runner_up][1]:.2f} to {CI[runner_up][2]:.2f}. The",
        f"training-free selector sits just below at {_vb:.2f} per cent, and the cascade at",
        f"{casc_ms:.0f} ms buys {_casc - _vb:.2f} points over it. No enhancement costs nothing and cannot",
        f"be placed on a log axis; it is drawn as the dashed baseline, and every fixed",
        f"operator applied unconditionally falls below it. The grey line joins the front",
        f"ends that nothing else beats at their own cost or below, and the red arrow runs",
        f"from the prior to the most accurate learned restorer, the comparison the",
        f"article turns on. Right: the same methods priced on clean images. Every",
        f"learned model charges something, from {abs(cf_of('cidnet') - base_cf):.2f} points for",
        f"CIDNet to {abs(cf_of('zero_dce') - base_cf):.2f} for Zero-DCE, and CLAHE applied unconditionally charges",
        f"{abs(cf_of('clahe') - base_cf):.2f}. The dark channel prior charges {abs(cf_of('dcp') - base_cf):.2f} points and the selector charges",
        f"nothing, because on clean input the routing rule never leaves passthrough and",
        f"is bit-identical to no enhancement. The most accurate front end on degraded",
        f"images is thus also among the cheapest to run and, of those that always act,",
        f"the least costly on clean ones.",
        f"Latency is measured per crop, batch of one, on a single CPU thread, at the",
        f"original crop resolution, over {T['protocol']['n_degraded']} degraded crops drawn round-robin from all",
        f"sixty (challenge, severity) cells; each crop is timed {T['protocol']['repeats']} times and the fastest",
        f"kept, twenty-five times for front ends under five milliseconds, and the whole",
        f"run is repeated so that only the digits both passes support are quoted. The",
        f"Points whose label would have crowded a neighbour are named in the right",
        f"panel instead, beside the marker they wear. The",
        f"accuracies of Zero-DCE, FFA-Net and PromptIR come from an earlier run under the",
        f"same protocol and carry no paired interval. Scope: CompactCNN (145,291",
        f"parameters, 32 x 32 input) on CURE-TSR.",
    ]:
        print("  " + ln)
    if not ok or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
