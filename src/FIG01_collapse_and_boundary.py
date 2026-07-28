# -*- coding: utf-8 -*-
"""
FIG01_collapse_and_boundary.py
Paper figure 1. Script, output and paper number now agree: FIG01 writes
fig01_collapse_and_boundary, and the article prints it as Fig. 1. The earlier
set numbered scripts by the order they were written, which no longer matched
the order the article cites them in.

Figure 1: how far recognition falls on each kind of degradation, and which family
of front end is ahead when it does.

WHAT THE FIGURE SAYS
On clean crops the classifier reaches 80.77 per cent. Degradation does not take
one amount away; it takes twelve different amounts, from a Shadow that still
leaves 73.0 per cent at severity 5 to a Gaussian blur that leaves 6.7. The right
panel asks the question the paper is about: on each challenge, is the best
training-free operator ahead of the best learned restorer, and by how much. The
answer is asymmetric. Where a training-free operator leads it leads by as much as
a wide margin; where a learned restorer leads it never leads by much. Both
figures are computed and printed by the script, never typed here.

WHY THIS COMPARISON AND NOT "WHICH OPERATOR WINS"
An earlier draft of this figure ranked all seven front ends per challenge and
named a winner. Two of those winners were not separable from the runner up, and
on Darkening the best training-free operator and the best learned one were level
while BOTH recovered more than twelve points, which made the headline "every
large recovery is training-free" true as written and fragile underneath. The
family comparison is the question the paper actually asks, it is robust to which
member of a family happens to come first, and it is what the capability boundary
of Fig. 3 is later built on.

WHY THE INTERVALS ARE ANALYTIC AND NOT BOOTSTRAPPED
The statistic is a mean over the five severity cells of the paired difference
inside each cell, so its variance is the sum of the within-cell variances over
the squared number of cells. Writing that down gives the same intervals as a
paired bootstrap, agreeing on all twelve challenges to within 0.03 points, and it
gives them without a random seed. That matters here: Darkening sits at +0.80 with
a lower bound of -0.01, and a bootstrapped verdict for it flipped between "level"
and "training-free ahead" depending on the seed. A verdict that moves with the
seed is not a verdict.

WHAT THE PANELS SHARE
Both panels carry the same twelve rows in the same order, sorted by the right
panel's difference, so a reader can run a finger along one row and read "this is
how far it fell, and this is which family was ahead". The left panel's diamond is
the no-enhancement average over severities 1 to 5; the small markers are the five
severities, palest first.

FOUR CELLS THAT LOOK LIKE A BUG AND ARE NOT
Four of the sixty severity cells read above the 80.77 clean baseline: Exposure at
severities 1 and 2 (84.17 and 81.07) and DirtyLens at severities 2 and 3 (82.17
and 82.54). A mild exposure change, or a little dirt on the lens, is not purely
destructive for this classifier. The figure reports them rather than clipping
them, the caption names all four, and the audit prints them on every run so a
future change in their number cannot pass unnoticed.

READS   outputs_revision/merged_per_image.csv   (K_merge_results.py)
        outputs_revision/dcp_cure.csv           (Q_dcp_branch.py)
WRITES  outputs_revision/figures/fig01_collapse_and_boundary.png   (400 dpi)
        outputs_revision/figures/fig01_collapse_and_boundary.svg
RUN     python FIG01_collapse_and_boundary.py
        python FIG01_collapse_and_boundary.py --target word
"""
import argparse
import csv
import os
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

NAMES = {1: "Decolorization", 2: "LensBlur", 3: "CodecError", 4: "Darkening",
         5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur", 8: "Noise",
         9: "Rain", 10: "Shadow", 11: "Snow", 12: "Haze"}
# OPS drives what is read from the merged file, so it has to list every front
# end the figure names, not only the ones it draws a row for.
OPS = ["passthrough", "gamma", "clahe", "stretch", "dcp", "cidnet", "adair",
       "zero_dce", "ffa_net", "promptir"]
PRETTY = {"passthrough": "none", "gamma": "Gamma", "clahe": "CLAHE",
          "stretch": "Stretch", "dcp": "DCP", "cidnet": "CIDNet",
          "adair": "AdaIR", "zero_dce": "Zero-DCE", "ffa_net": "FFA-Net",
          "promptir": "PromptIR"}
TRAINING_FREE = ["gamma", "clahe", "stretch", "dcp"]
# All five learned restorers are scored on this machine now, so the family is
# represented by whichever of the five is strongest on the challenge in hand
# rather than by whichever of two happened to be aligned.
LEARNED = ["cidnet", "adair", "zero_dce", "ffa_net", "promptir"]
SEVS = [1, 2, 3, 4, 5]

# One column is 3.48 inches wide and the row labels have to leave room for two
# panels inside it. These are the names used at that width; the full ones are
# kept for the wide version and for the caption, which spells them out.
# Not abbreviations: the internal keys are run together, and these are the same
# words written the way a reader expects them. Nothing is shortened, because a
# label a reader has to decode costs more than the width it saves.
SHORT_CH = {"GaussianBlur": "Gaussian blur", "CodecError": "Codec error",
            "DirtyLens": "Dirty lens", "LensBlur": "Lens blur"}

GREEN = "#2E7D4F"          # the training-free family is ahead
PURPLE = "#8C5FBF"         # the learned family is ahead
GREY = "#9A9A9A"           # neither, at 95 per cent

# Locked from merged_per_image.csv and dcp_cure.csv, recomputed image by image.
EXPECT_CLEAN = 80.77
# challenge: (no-enhancement average over severities 1 to 5,
#             best training-free minus best learned, verdict)
EXPECT = {
    "Rain":            (35.31,  10.53, "free"),
    "GaussianBlur":    (28.43,   9.56, "free"),
    "Haze":            (46.41,   7.59, "free"),
    "LensBlur":        (37.71,   6.78, "free"),
    "CodecError":      (52.37,   2.01, "free"),
    "Shadow":          (75.81,   0.12, "level"),
    "DirtyLens":       (76.64,   0.07, "level"),
    "Noise":           (53.62,  -0.93, "learned"),
    "Decolorization":  (65.81,  -1.15, "learned"),
    "Snow":            (64.81,  -1.49, "learned"),
    "Exposure":        (54.35,  -1.82, "learned"),
    "Darkening":       (62.63,  -1.89, "learned"),
}
# The span of each challenge's severity curve, which is what the left panel draws
# as a line. The averages above constrain the sum of the five points; these pin
# the two ends a reader actually looks at.
EXPECT_SPAN = {
    "Decolorization": (54.73, 77.81), "LensBlur": (13.39, 67.83),
    "CodecError": (45.04, 61.39),     "Darkening": (32.47, 76.70),
    "DirtyLens": (57.99, 82.54),      "Exposure": (13.83, 84.17),
    "GaussianBlur": (6.66, 59.99),    "Noise": (26.26, 76.18),
    "Rain": (25.44, 52.07),           "Shadow": (73.00, 77.88),
    "Snow": (44.08, 79.07),           "Haze": (18.20, 77.00),
}
EXPECT_LEADS = (10.53, 1.89)   # largest training-free lead, largest learned lead


def paired_cell_diff(hit_a, hit_b, idx):
    """Difference of two front ends on one challenge, with an analytic interval.

    The reported statistic is the mean over the five severity cells of the mean
    paired difference inside the cell, so the variance is the sum of the
    within-cell variances of that difference, each divided by its own count, over
    the squared number of cells. No resampling, so no seed, so no verdict that
    moves when the seed does.
    """
    k = len(idx)
    means, var = [], 0.0
    for i in idx:
        d = hit_a[i].astype(np.float64) - hit_b[i].astype(np.float64)
        means.append(d.mean())
        var += d.var(ddof=1) / len(i)
    point = 100.0 * float(np.mean(means))
    half = 1.96 * 100.0 * float(np.sqrt(var)) / k
    return point, point - half, point + half


def load(merged_path, dcp_path):
    rows = list(csv.DictReader(open(merged_path, newline="", encoding="utf-8")))
    need = ["ch", "sev", "true"] + [f"pred_{o}" for o in OPS if o != "dcp"]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit(f"ABORT: {merged_path} has no columns {missing}.")
    dcp = {}
    for r in csv.DictReader(open(dcp_path, newline="", encoding="utf-8")):
        dcp[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = \
            int(r["pred_dcp"])
    key = [(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
           for r in rows]
    absent = sum(1 for k in key if k not in dcp)
    if absent:
        raise SystemExit(f"ABORT: {absent} rows of {merged_path} have no DCP "
                         f"prediction in {dcp_path}; the two files are not the "
                         f"same run.")
    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    pred = {o: np.array([int(r[f"pred_{o}"]) for r in rows])
            for o in OPS if o != "dcp"}
    pred["dcp"] = np.array([dcp[k] for k in key])
    return ch, sev, tru, {o: (v == tru) for o, v in pred.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS), default="ieee1col")
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    for p in (args.merged, args.dcp):
        if not Path(p).exists():
            raise SystemExit(f"ABORT: {p} not found.")
    ch, sev, tru, hit = load(args.merged, args.dcp)

    clean_m = sev == 0
    if clean_m.sum() == 0:
        raise SystemExit("ABORT: no severity 0 rows, so there is no clean "
                         "baseline to draw.")
    clean = 100.0 * float(np.mean(hit["passthrough"][clean_m]))

    row = {}
    for cid, name in NAMES.items():
        idx = [np.where((ch == cid) & (sev == s))[0] for s in SEVS]
        if any(len(i) == 0 for i in idx):
            raise SystemExit(f"ABORT: {name} is missing a severity.")
        per_sev = [100.0 * float(hit["passthrough"][i].mean()) for i in idx]
        acc = {o: 100.0 * float(np.mean([hit[o][i].mean() for i in idx]))
               for o in OPS}
        b_free = max(TRAINING_FREE, key=lambda o: acc[o])
        b_learn = max(LEARNED, key=lambda o: acc[o])
        d, lo, hi = paired_cell_diff(hit[b_free], hit[b_learn], idx)
        verdict = "free" if lo > 0 else ("learned" if hi < 0 else "level")
        row[name] = dict(per_sev=per_sev, base=acc["passthrough"], diff=d,
                         lo=lo, hi=hi, verdict=verdict,
                         b_free=b_free, b_learn=b_learn)

    # ---------------- audit ----------------
    print("=== EXPECTED-OUTPUT AUDIT ===")
    bad = 0
    if abs(round(clean, 2) - EXPECT_CLEAN) > 1e-9:
        print(f"  MISMATCH clean baseline: got {clean:.2f}, "
              f"expected {EXPECT_CLEAN:.2f}")
        bad += 1
    for name, (e_base, e_diff, e_verdict) in EXPECT.items():
        d = row[name]
        for label, got, exp in (("no-enhancement average", d["base"], e_base),
                                ("family difference", d["diff"], e_diff)):
            if abs(round(got, 2) - exp) > 1e-9:
                print(f"  MISMATCH {name} {label}: got {got:.2f}, "
                      f"expected {exp:.2f}")
                bad += 1
        if d["verdict"] != e_verdict:
            print(f"  MISMATCH {name} verdict: got {d['verdict']}, "
                  f"expected {e_verdict}")
            bad += 1
    for name, (e_lo, e_hi) in EXPECT_SPAN.items():
        got_lo = min(row[name]["per_sev"])
        got_hi = max(row[name]["per_sev"])
        if abs(round(got_lo, 2) - e_lo) > 1e-9 or \
                abs(round(got_hi, 2) - e_hi) > 1e-9:
            print(f"  MISMATCH {name} severity span: got "
                  f"{got_lo:.2f} to {got_hi:.2f}, expected "
                  f"{e_lo:.2f} to {e_hi:.2f}")
            bad += 1
    above = [(n, s_i + 1, v) for n, d in row.items()
             for s_i, v in enumerate(d["per_sev"]) if v > clean]
    print(f"  {len(above)} of the 60 severity cells read above the clean "
          f"baseline: " + ", ".join(f"{n} severity {s_i} at {v:.2f}"
                                    for n, s_i, v in sorted(above)))

    lead_free = max(d["diff"] for d in row.values() if d["verdict"] == "free")
    lead_learn = -min(d["diff"] for d in row.values()
                      if d["verdict"] == "learned")
    if (abs(round(lead_free, 2) - EXPECT_LEADS[0]) > 1e-9 or
            abs(round(lead_learn, 2) - EXPECT_LEADS[1]) > 1e-9):
        print(f"  MISMATCH leads: got {lead_free:.2f} and {lead_learn:.2f}, "
              f"expected {EXPECT_LEADS[0]:.2f} and {EXPECT_LEADS[1]:.2f}")
        bad += 1

    print("  best training-free against best learned, analytic 95 per cent "
          "interval on the paired cell difference:")
    for name in sorted(row, key=lambda n: -row[n]["diff"]):
        d = row[name]
        print(f"    {name:15s} {PRETTY[d['b_free']]:7s} vs "
              f"{PRETTY[d['b_learn']]:6s} {d['diff']:+6.2f} "
              f"[{d['lo']:+.2f}, {d['hi']:+.2f}]  {d['verdict']}")
    n_free = sum(1 for d in row.values() if d["verdict"] == "free")
    n_learn = sum(1 for d in row.values() if d["verdict"] == "learned")
    n_level = 12 - n_free - n_learn
    print(f"  training-free ahead on {n_free}, learned ahead on {n_learn}, "
          f"level on {n_level}; largest leads {lead_free:.2f} and "
          f"{lead_learn:.2f}")
    print("  AUDIT PASSED: every value reproduces." if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    # Full-width (two-column-spanning) float; bump the base font for readability
    # at that width, matching the other figures in the set.
    F = F * 1.18 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    # The grey note that used to sit under the finding repeated what the caption
    # says and has gone. The canvas keeps its height and the panels take the
    # room instead: the plotting area grows from 3.48 to 3.61 inches, which is
    # what a reader looks at, and the type grows with it.
    H = 5.2 if args.target == "ieee" else (5.05 if ONE else 4.95)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    # In one column the row labels eat a fifth of the width, so the panels are
    # nearer to equal and the gutter is thinner; at full width the left panel
    # can afford to be the wider of the two.
    if ONE:
        # 0.88 inch for the row names, then two equal panels of 1.24 and a
        # thin gutter. Measured against the longest name at eight point rather
        # than guessed, so nothing has to be cut short.
        # The panels are not equal. The left one plots accuracy on a scale a
        # reader reads at a glance and needs little width; the right one has to
        # hold its own title, its bars and the name of the front end that leads
        # each of them, so it gets the wider share.
        gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.38], left=0.253,
                              right=0.983, top=0.958, bottom=0.300,
                              wspace=0.048)
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.00], left=0.163,
                              right=0.972, top=0.945, bottom=0.265,
                              wspace=0.085)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    order = sorted(NAMES.values(), key=lambda n: -row[n]["diff"])
    ypos = {n: i for i, n in enumerate(order)}

    # ---- left: where the accuracy goes ----
    axL.axvline(clean, color=fs.RED, lw=1.2, ls="--", zorder=2)
    axL.text(clean - 1.5, -0.95, f"clean {clean:.1f}", ha="right", va="center",
             fontsize=F * 0.68, color=fs.RED)
    for name in order:
        y = ypos[name]
        ps = row[name]["per_sev"]
        # The five severities used to be five shades of grey joined by a grey
        # line, on a grey rule, and at column width they read as a smudge. The
        # rule is now much fainter, the first severity is drawn hollow and the
        # last nearly black, so the direction of the sequence is visible as
        # fill rather than as a difference in tone, and the white edges are
        # thinner so they no longer eat a small marker.
        axL.plot([min(ps), max(ps)], [y, y], color="#E6E6E6", lw=1.1, zorder=3,
                 solid_capstyle="round")
        for k, v in enumerate(ps):
            hollow = k == 0
            axL.plot([v], [y], marker="o", ms=F * (0.30 if ONE else 0.25),
                     mfc="white" if hollow else plt.cm.Greys(0.30 + 0.13 * k),
                     mec="#8A8A8A" if hollow else "white",
                     mew=0.7 if hollow else 0.45, zorder=4)
        # The severity ramp stops at a dark grey rather than at black so that
        # the mean, which is black and diamond-shaped, is told apart from the
        # fifth severity by tone as well as by shape. Red is not used here: the
        # clean baseline and the finding already carry it.
        axL.plot([row[name]["base"]], [y], marker="D",
                 ms=F * (0.36 if ONE else 0.30),
                 color="#111111", mec="white", mew=0.7, zorder=6)

    axL.set_yticks(range(len(order)))
    axL.set_yticklabels([SHORT_CH.get(n, n) if ONE else n for n in order],
                        fontsize=F * 0.74)
    axL.set_ylim(len(order) - 0.4, -1.6)
    axL.set_xlim(0, 92)
    axL.set_xticks([0, 20, 40, 60, 80])
    # The panel titles carry the context, so at column width the axis labels
    # only have to name the quantity. Spelled out they are wider than the panel
    # and the two of them meet in the gutter.
    axL.set_xlabel("accuracy with\nno enhancement (%)" if ONE
                   else "accuracy with no enhancement (%)",
                   fontsize=F * 0.74, labelpad=2, linespacing=1.2)
    axL.tick_params(axis="x", labelsize=F * 0.72, length=2.5, pad=1.5)
    axL.tick_params(axis="y", length=0, pad=3)
    for sp in ("top", "right", "left"):
        axL.spines[sp].set_visible(False)
    axL.spines["bottom"].set_color("#999999")

    # ---- right: which family is ahead ----
    axR.axvline(0, color="#999999", lw=0.9, zorder=2)
    for name in order:
        y = ypos[name]
        d = row[name]
        col = {"free": GREEN, "learned": PURPLE, "level": GREY}[d["verdict"]]
        axR.barh(y, d["diff"], height=0.55, color=col, zorder=3)
        # Every label sits to the right, either just past a positive bar or
        # just past the zero line when the bar runs left. Put beyond the end of
        # a negative bar it would need room the panel does not have at column
        # width, and the row is unambiguous either way.
        lab = ("level" if d["verdict"] == "level" else
               PRETTY[d["b_free"] if d["verdict"] == "free" else d["b_learn"]])
        x, ha = max(d["diff"], 0.0) + 0.45, "left"
        axR.text(x, y, lab, ha=ha, va="center", fontsize=F * 0.68,
                 color=fs.INK, zorder=6)

    axR.set_yticks(range(len(order)))
    axR.set_yticklabels([])
    axR.set_ylim(len(order) - 0.4, -1.6)
    # The bars carry the name of whichever front end holds the lead, written
    # beyond the bar's end, so the axis has to reach past the longest bar by
    # enough to hold the longest name. It is set from the data rather than
    # typed: the leads shrank when the learned family grew, and a fixed limit
    # would have left the labels hanging over the canvas edge.
    _hi = max(d["diff"] for d in row.values())
    _lo = min(d["diff"] for d in row.values())
    axR.set_xlim(_lo - 1.1 if ONE else _lo - 3.6,
                 _hi + 4.0 if ONE else _hi + 5.6)
    axR.set_xticks([-4, 0, 4, 8, 12])
    axR.set_xlabel("training-free lead\n(points)" if ONE
                   else "training-free lead (points)",
                   fontsize=F * 0.74, labelpad=2)
    axR.tick_params(axis="x", labelsize=F * 0.72, length=2.5, pad=1.5)
    axR.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.spines["bottom"].set_color("#999999")

    axL.set_title("how far it falls", fontsize=F * 0.84, pad=6)
    axR.set_title("which family is ahead", fontsize=F * 0.84, pad=6)

    h = [Line2D([], [], color="#CFCFCF", lw=1.4, marker="o", ms=F * 0.25,
                mfc="#8A8A8A", mec="white", mew=0.65),
         Line2D([], [], color="none", marker="D", ms=F * 0.30, mfc="#111111",
                mec="white", mew=0.7),
         Line2D([], [], color=GREEN, lw=5),
         Line2D([], [], color=PURPLE, lw=5)]
    # At column width the spelled-out legend is wider than the page. The short
    # form says the same thing; the caption carries the full wording.
    # One column, four rows, every entry written out. Two columns of this
    # wording measured 3.98 inches against a canvas of 3.48; the choice is
    # between stacking them and cutting the words, and the words are the point.
    _lab = (["severity 1 hollow, severity 5 darkest",
             "the mean of the five",
             "a training-free operator is ahead",
             "a learned restorer is ahead"] if ONE else
            ["severities 1 to 5, palest first", "their average, the diamond",
             "a training-free operator is ahead", "a learned restorer is ahead"])
    fig.legend(h, _lab,
               loc="lower center", bbox_to_anchor=(0.5, 0.088 if ONE else 0.093),
               ncol=1 if ONE else 2,
               frameon=False, fontsize=F * 0.70, handlelength=1.8,
               handletextpad=0.55, columnspacing=1.9, labelspacing=0.30)

    # The finding is stated once, in the figure, with both numbers taken from
    # the data. It used to read 12.9 and 1.8 as literals, which stayed put when
    # the learned family grew from two members to five and the leads became
    # 10.5 and 1.9. What a reader can check on the page has to come from the
    # same place the bars do.
    # One line of this is four inches wide, which a column does not have. It
    # breaks after the semicolon, where the sentence already pauses.
    _find = (f"A training-free lead reaches {lead_free:.2f} points;\n"
             f"a learned lead never passes {lead_learn:.2f}." if ONE else
             f"A training-free lead reaches {lead_free:.2f} points; a learned "
             f"lead never passes {lead_learn:.2f}.")
    fig.text(0.5, 0.052 if ONE else 0.060, _find, ha="center", va="center",
             fontsize=F * 0.74, color=fs.RED, linespacing=1.25)

    if os.environ.get("PROBE"):
        fig.canvas.draw()
        _r = fig.canvas.get_renderer(); _d = fig.dpi
        print("\n[probe] inches from the bottom")
        print(f"  axes bottom {axL.get_window_extent(renderer=_r).y0/_d:.3f}")
        _x = axL.xaxis.get_label().get_window_extent(renderer=_r)
        print(f"  xlabel      {_x.y0/_d:.3f} - {_x.y1/_d:.3f}")
        for _l in fig.legends:
            _b = _l.get_window_extent(renderer=_r)
            print(f"  legend      y {_b.y0/_d:.3f}-{_b.y1/_d:.3f}  x {_b.x0/_d:.3f}-{_b.x1/_d:.3f} (canvas {fig.get_size_inches()[0]:.2f})")
        for _t in fig.texts:
            if _t.get_text().startswith("A training-free"):
                _b = _t.get_window_extent(renderer=_r)
                print(f"  finding     {_b.y0/_d:.3f} - {_b.y1/_d:.3f}")
        _a = axR.get_window_extent(renderer=_r)
        print(f"  right panel right edge {_a.x1/_d:.3f} of {fig.get_size_inches()[0]:.3f}")

    ok_gate = fs.run_gates(fig, args.outdir, "fig01_collapse_and_boundary",
                           bar_axes=[axR], line_axes=[axL])
    fs.report_sizes(args.target, W, F, {
        "panel titles": 0.84, "challenge names": 0.74, "axis labels": 0.74,
        "tick labels": 0.72, "family labels": 0.68, "the legend": 0.70,
        "the finding": 0.72, "the scope note": 0.68})

    print("\nCAPTION:")
    for ln in [
        "Fig. 1. What degradation costs, and which family of front end is ahead when it",
        "does. Left: the accuracy of the classifier with no enhancement on each of the",
        "twelve CURE-TSR challenges, one row per challenge. The five small markers are",
        "severities 1 to 5, palest first, and the diamond is their average. The dashed",
        "line is the clean baseline of 80.77 per cent. Degradation does not take one",
        "amount away: at severity 5 Shadow still reads 73.0 per cent while Gaussian",
        "blur reads 6.7. Four of the sixty severity cells read above the clean",
        "baseline, Exposure at severities 1 and 2 and DirtyLens at severities 2 and 3,",
        "the highest being 84.2; a mild exposure change or a little dirt on the lens is",
        "not purely destructive for this classifier, and the values are reported rather",
        "than clipped. Right: on the same challenge, the degraded-average accuracy of",
        "the best training-free operator minus that of the best learned restorer, both",
        "chosen within the challenge. The training-free four are gamma, CLAHE, stretch",
        f"and the dark channel prior; the learned five are "
        + ", ".join(PRETTY[o] for o in LEARNED[:-1])
        + f" and {PRETTY[LEARNED[-1]]}. Bars are",
        "labelled with whichever front end holds the lead. Both panels carry the same",
        "rows in the same order, sorted by that difference. The asymmetry is the point:",
        f"where a training-free operator leads it leads by as much as {lead_free:.2f} points,",
        f"and where a learned restorer leads it never leads by more than "
        f"{lead_learn:.2f}. Level marks a",
        "challenge on which the two families are not separable, and no claim is made",
        "about which is ahead there. Intervals are analytic on the paired difference",
        "within each severity cell rather than bootstrapped, so they carry no random",
        "seed; they agree with a paired bootstrap on all twelve challenges to within",
        "0.03 points. Scope: CompactCNN (145,291 parameters, 32x32 input) on CURE-TSR;",
        "accuracies are averaged over severities 1 to 5 within each challenge.",
    ]:
        print("  " + ln)
    if not ok_gate or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
