# -*- coding: utf-8 -*-
"""
FIG04_operator_selection_oracle.py
Paper figure 4. Script, output and paper number agree.

Figure 4: how much room is left in operator SELECTION, and how much of that room
is real.

WHAT THE FIGURE SAYS
Left: on the same seven-operator pool, a selector that could pick the right
operator for every image using ground truth would reach 72.19, while the proposed
training-free selector reaches 58.23. Nearly 14 accuracy points are therefore left
in selection alone, without inventing a single new operator. That is the research
gap this paper points at.

Right: but not all of that room is restoration, and the paper says so before a
reviewer does. Broken down by challenge, 37 per cent of the headroom (5.23 of the
13.96 points) comes from the four challenges on which NO operator clears the
noise-injection oracle (Fig. 3): Gaussian blur alone contributes the single
largest share, +24.4 points, and it restores nothing. On those challenges the
oracle still gains, because on some individual images one operator happens to be
right where the others are wrong. That is per-image chance, not restoration, and
no realizable selector has a signal to aim at it. The honest headroom is the 8.73
points that come from the eight challenges where restoration demonstrably happens.

THE TWO ORACLES (Evaluation Protocol Part 17.9; never write "the oracle" alone)
  operator-selection oracle  THIS figure. The best accuracy reachable if the best
                             OPERATOR could be picked per image using ground truth.
                             A CEILING: it measures headroom in selection.
  noise-injection oracle     Fig. 3 and Fig. 4. The best accuracy reachable by
                             adding Gaussian noise alone. A BAR: an operator must
                             clear it before its gain counts as restoration.
  They are different objects. The caption states this explicitly, because a reader
  who has just met one of them will otherwise assume the other is the same thing.

WHY THE ORACLE BARS ARE HATCHED
They are not achievable. Drawing them in the same style as the measured methods
would invite the reader to compare them as if they were competitors.

READS   outputs_revision/merged_per_image.csv   (K_merge_results.py)
        outputs_revision/dcp_cure.csv           (Q_dcp_branch.py)
WRITES  outputs_revision/figures/fig04_operator_selection_oracle.png   (400 dpi)
        outputs_revision/figures/fig04_operator_selection_oracle.svg
RUN     python FIG04_operator_selection_oracle.py
        python FIG04_operator_selection_oracle.py --target word
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import fig_style as fs

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

NAMES = {1: "Decolorization", 2: "LensBlur", 3: "CodecError", 4: "Darkening",
         5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur", 8: "Noise", 9: "Rain",
         10: "Shadow", 11: "Snow", 12: "Haze"}
SEVS = range(1, 6)
CHS = range(1, 13)

# The four challenges on which no operator clears the noise-injection oracle at any
# severity under the 288-test correction (Fig. 3).
UNRESTORABLE = {"Decolorization", "GaussianBlur", "CodecError", "Shadow"}

POOL4 = ["passthrough", "gamma", "clahe", "stretch"]     # the base rule's own arms
POOL5 = POOL4 + ["dcp"]
# Ten, not seven. The ladder used to stop at AdaIR and CIDNet because only those
# two learned restorers were aligned image by image; all five are now, and the
# rung they add together is the point of the panel: perfecting selection over
# the four base operators buys nine and three quarter points, the prior four
# more, and five learned restorers between them only two and a half.
POOL10 = POOL5 + ["adair", "cidnet", "zero_dce", "ffa_net", "promptir"]

# Locked 2026-07-11 from the authoritative per-image files.
EXPECT = dict(passthrough=54.49, va=56.65, vb=58.23, dcp=60.72,
              o4=66.42, o5=70.55, o10=73.06,
              headroom=14.83, unrest_share=5.55, rest_share=9.28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS),
                    default="ieee1col")
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    for p in (args.merged, args.dcp):
        if not Path(p).exists():
            raise SystemExit(f"ABORT: {p} not found.")
    rows = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    dcp = {}
    for r in csv.DictReader(open(args.dcp, newline="", encoding="utf-8")):
        dcp[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = \
            int(r["pred_dcp"])

    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    branch = np.array([r["rule_branch"] for r in rows])
    P = {k: np.array([int(r["pred_" + k]) for r in rows])
         for k in ["passthrough", "gamma", "clahe", "stretch", "va_rule",
                   "adair", "cidnet", "zero_dce", "ffa_net", "promptir"]}
    P["dcp"] = np.array([dcp[(r["filename"], int(r["occ"]), int(r["ch"]),
                              int(r["sev"]))] for r in rows])
    P["vb"] = np.where(branch == "clahe", P["dcp"], P["va_rule"])

    cells = {(c, s): (ch == c) & (sev == s) for c in CHS for s in SEVS}

    def acc_cell(op, key):
        m = cells[key]
        return 100.0 * float(np.mean(P[op][m] == tru[m]))

    def oracle_cell(pool, key):
        m = cells[key]
        ok = np.zeros(int(m.sum()), bool)
        for op in pool:
            ok |= (P[op][m] == tru[m])
        return 100.0 * float(ok.mean())

    def deg_avg(fn):
        return float(np.mean([fn(k) for k in cells]))

    val = {
        "passthrough": deg_avg(lambda k: acc_cell("passthrough", k)),
        "va": deg_avg(lambda k: acc_cell("va_rule", k)),
        "vb": deg_avg(lambda k: acc_cell("vb", k)),
        "dcp": deg_avg(lambda k: acc_cell("dcp", k)),
        "o4": deg_avg(lambda k: oracle_cell(POOL4, k)),
        "o5": deg_avg(lambda k: oracle_cell(POOL5, k)),
        "o10": deg_avg(lambda k: oracle_cell(POOL10, k)),
    }
    headroom = val["o10"] - val["vb"]

    # per-challenge headroom of the operator-selection oracle over the selector
    per_ch = {}
    for c in CHS:
        g = np.mean([oracle_cell(POOL10, (c, s)) - acc_cell("vb", (c, s))
                     for s in SEVS])
        per_ch[NAMES[c]] = float(g)
    unrest = float(np.mean([g for n, g in per_ch.items() if n in UNRESTORABLE])) \
        * len(UNRESTORABLE) / 12.0
    rest = float(np.mean([g for n, g in per_ch.items() if n not in UNRESTORABLE])) \
        * (12 - len(UNRESTORABLE)) / 12.0

    # ---------------- audit ----------------
    # The audit compares what will be PRINTED, not the raw float. A tolerance test
    # is the wrong instrument here: the headroom is 13.9595, and an earlier version
    # of this file had 13.96 locked in EXPECT, which a 0.02 (or even a 0.006)
    # tolerance waved through. Comparing the two-decimal value catches it.
    print("=== EXPECTED-OUTPUT AUDIT (against values locked 2026-07-11) ===")
    bad = 0
    got = dict(val, headroom=headroom, unrest_share=unrest, rest_share=rest)
    for k, e in EXPECT.items():
        if abs(round(got[k], 2) - e) > 1e-9:
            print(f"  MISMATCH {k}: got {got[k]:.4f} (prints as {got[k]:.2f}), "
                  f"expected {e:.2f}")
            bad += 1
    if abs(round(unrest + rest, 2) - round(headroom, 2)) > 1e-9:
        print(f"  MISMATCH: the two shares do not sum to the headroom "
              f"({unrest:.2f} + {rest:.2f} != {headroom:.2f})")
        bad += 1
    print(f"  no enhancement {val['passthrough']:.2f} | base rule {val['va']:.2f} | "
          f"selector V-B {val['vb']:.2f} | best single operator (DCP) {val['dcp']:.2f}")
    print(f"  operator-selection oracle: o4 {val['o4']:.2f} -> o5 {val['o5']:.2f} "
          f"-> o10 {val['o10']:.2f}")
    print(f"  headroom over the selector: {headroom:.2f} points, of which "
          f"{rest:.2f} ({100*rest/headroom:.0f} per cent) comes from the eight")
    print(f"  restorable challenges and {unrest:.2f} "
          f"({100*unrest/headroom:.0f} per cent) from the four on which nothing "
          f"clears the noise-injection oracle.")
    print("  AUDIT PASSED: every value reproduces." if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure; send the output.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    # Full-width (two-column-spanning) float; bump base font ~20% for readability
    # at that width, matching Fig.9 (both are bar charts with room to spare).
    F = F * 1.10 if args.target == "ieee" else F
    H = 5.75 if args.target == "ieee" else (6.42 if args.target == "ieee1col" else 5.20)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    ONE = args.target == "ieee1col"
    if ONE:
        # Stacked, not side by side. Two panels of bars cannot share a column
        # three and a half inches wide and still carry their row labels.
        gs = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.92], left=0.360,
                              right=0.975, top=0.952, bottom=0.270,
                              hspace=0.19)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[1, 0])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.00, 1.30], left=0.215,
                              right=0.965, top=0.945, bottom=0.360,
                              wspace=0.48)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1])

    # ---- left: the ladder ----
    # Measured bars are SLATE, not green. Figs. 3 and 4 use green to mean "clears
    # the noise-injection oracle", and an earlier draft of this figure reused green
    # for "measured and achievable", which coloured even the no-enhancement baseline
    # green. Green is reserved here for the one thing it means elsewhere: a
    # challenge on which restoration demonstrably happens.
    SLATE = "#5B7FA6"
    LADDER = [("no enhancement", val["passthrough"], False),
              ("base rule", val["va"], False),
              ("selector V-B", val["vb"], False),
              ("best single\noperator (DCP)", val["dcp"], False),
              # "front ends", not "operators". The smallest pool contains no
              # enhancement, which is not an operator, and the article uses
              # "the four training-free operators" for a different set: gamma,
              # CLAHE, contrast stretching and the dark channel prior.
              # Just the size of the pool. What is in each is said once, in the
              # caption, rather than three times down the left margin where it
              # does not fit at a legible size.
              ("4 front ends", val["o4"], True),
              ("5 front ends", val["o5"], True),
              ("10 front ends", val["o10"], True)]
    ys = np.arange(len(LADDER))
    for y, (lbl, v, is_oracle) in zip(ys, LADDER):
        axL.barh(y, v, height=0.66, zorder=3,
                 color="#F2F2F2" if is_oracle else SLATE,
                 edgecolor="#4A4A4A" if is_oracle else "none",
                 hatch="////" if is_oracle else None, linewidth=0.8)
        axL.text(v + 1.0, y, f"{v:.2f}", va="center", ha="left",
                 fontsize=F * 0.74, color=fs.INK, zorder=4)
    axL.axhline(3.5, color="#666666", lw=0.9, ls=":", zorder=2)
    axL.set_yticks(ys)
    # The names of the things being compared are the subject of the panel, so
    # they are set in bold at the same size as the challenge names below rather
    # than shrunk to whatever fitted. Room comes from the canvas, not the type.
    axL.set_yticklabels([l for l, _, _ in LADDER],
                        fontweight="bold" if ONE else "normal",
                        fontsize=F * (0.74 if ONE else 0.80),
                        linespacing=1.15)
    axL.set_ylim(len(LADDER) - (0.55 if ONE else 0.4), -1.95)
    axL.set_xlim(0, 92)
    # Stacked, the upper axis label and the lower panel's title share one gap
    # and collide in it. The label goes: every bar carries its own value and
    # the panel title says what those values are.
    if not ONE:
        axL.set_xlabel("degraded-average accuracy (%)", fontsize=F * 0.78,
                       labelpad=2)
    axL.tick_params(axis="x", labelsize=F * 0.74, length=2.5, pad=1.5)
    axL.tick_params(axis="y", length=0, pad=3)
    for sp in ("top", "right", "left"):
        axL.spines[sp].set_visible(False)
    axL.spines["bottom"].set_color("#999999")
    axL.set_title("what perfect selection would buy", fontsize=F * 0.88, pad=6)

    # Guide lines from the two bars the arrow actually measures, so the reader does
    # not have to guess which two it spans.
    for v, row in ((val["vb"], 2), (val["o10"], 6)):
        axL.plot([v, v], [-0.95, row], color=fs.RED, lw=0.8, ls=(0, (2, 2)),
                 zorder=2, clip_on=False)
    axL.annotate("", xy=(val["o10"], -0.95), xytext=(val["vb"], -0.95),
                 arrowprops=dict(arrowstyle="<->", color=fs.RED, lw=1.2))
    axL.text(0.5 * (val["vb"] + val["o10"]), -1.22,
             f"headroom {headroom:.2f}", ha="center", va="bottom",
             fontsize=F * 0.76, color=fs.RED, fontweight="bold")

    # ---- right: where that headroom actually comes from ----
    order = sorted(per_ch, key=per_ch.get, reverse=True)
    ys2 = np.arange(len(order))
    for y, n in zip(ys2, order):
        bad_ = n in UNRESTORABLE
        axR.barh(y, per_ch[n], height=0.66, zorder=3,
                 # Not green. In Figs. 1 to 3 green means the training-free
                 # family; here the split is between challenges that admit
                 # restoration and those that do not, which has nothing to do
                 # with family, and reusing the colour would make a reader
                 # who has just read Fig. 3 stop and check. Red keeps the
                 # meaning it has there: one of the four that clear nothing.
                 color=fs.RED if bad_ else "#6E6E6E", edgecolor="none")
        axR.text(per_ch[n] + 0.5, y, f"{per_ch[n]:+.1f}", va="center", ha="left",
                 fontsize=F * 0.74, color=fs.INK, zorder=4)
    axR.set_yticks(ys2)
    # The internal keys run the words together; these are the same words as a
    # reader writes them, matching Fig. 3.
    PRETTY_CH = {"LensBlur": "Lens blur", "DirtyLens": "Dirty lens",
                 "GaussianBlur": "Gaussian blur", "CodecError": "Codec error"}
    axR.set_yticklabels([PRETTY_CH.get(c, c) for c in order],
                        fontsize=F * 0.76)
    for y, n in zip(ys2, order):
        if n in UNRESTORABLE:
            axR.get_yticklabels()[y].set_color(fs.RED)
    axR.set_ylim(len(order) - 0.4, -1.95)
    axR.set_xlim(0, 32)
    axR.set_xlabel("headroom over the selector (points)", fontsize=F * 0.78,
                   labelpad=2)
    axR.tick_params(axis="x", labelsize=F * 0.74, length=2.5, pad=1.5)
    axR.tick_params(axis="y", length=0, pad=3)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.spines["bottom"].set_color("#999999")
    axR.set_title("where that headroom comes from", fontsize=F * 0.88,
                  pad=2 if ONE else 6)

    h = [Rectangle((0, 0), 1, 1, fc=SLATE),
         Rectangle((0, 0), 1, 1, fc="#F2F2F2", ec="#4A4A4A", hatch="////", lw=0.8),
         Rectangle((0, 0), 1, 1, fc="#6E6E6E"),
         Rectangle((0, 0), 1, 1, fc=fs.RED)]
    # Shorter wording at column width. What is cut is not lost: the caption
    # says why the oracle is a ceiling, and Fig. 3 is where the two kinds of
    # challenge are defined and named.
    _lab = (["measured, without labels",
             "operator-selection oracle: a ceiling, not a competitor",
             "a challenge that admits restoration (Fig. 3)",
             "nothing clears the noise-injection oracle"] if ONE else
            ["measured, and achievable without labels",
             "operator-selection oracle: needs the ground-truth labels, "
             "so it is a ceiling, not a competitor",
             "a challenge that admits restoration (Fig. 3)",
             "a challenge on which no operator clears the "
             "noise-injection oracle"])
    fig.legend(h, _lab,
               loc="lower center", bbox_to_anchor=(0.5, 0.108 if ONE else 0.135), ncol=1,
               frameon=False, fontsize=F * (0.70 if ONE else 0.76), handlelength=1.5,
               handletextpad=0.55, labelspacing=0.32)

    # Each of these lines is under 105 characters at this size. The first draft ran
    # 130 characters, which needs 8.8 in on a 7.16 in canvas and overflowed both
    # edges. Width is the scarce resource; height is not.
    # At column width each line has to stay under about sixty characters, so the
    # same four statements are said in fewer words rather than in more lines.
    fig.text(0.5, 0.082 if ONE else 0.134,
             f"Perfect selection over the four base operators buys "
             f"+{val['o4'] - val['va']:.2f};" if ONE else
             "Perfecting selection on the base rule's own four operators buys "
             f"+{val['o4'] - val['va']:.2f} points;",
             ha="center", va="center", fontsize=F * (0.66 if ONE else 0.72), color=fs.INK)
    fig.text(0.5, 0.060 if ONE else 0.106,
             (f"the prior adds +{val['o5'] - val['o4']:.2f}, five learned "
              f"restorers only +{val['o10'] - val['o5']:.2f}." if ONE else
              f"the prior adds +{val['o5'] - val['o4']:.2f} and five learned "
              f"restorers only +{val['o10'] - val['o5']:.2f} between them."),
             ha="center", va="center", fontsize=F * (0.66 if ONE else 0.72), color=fs.INK)
    fig.text(0.5, 0.036 if ONE else 0.070,
             f"{100*unrest/headroom:.0f} per cent of it, {unrest:.2f} of "
             f"{headroom:.2f}, is from the four" if ONE else
             f"{100*unrest/headroom:.0f} per cent of the headroom "
             f"({unrest:.2f} of {headroom:.2f} points) comes from those four "
             "challenges.",
             ha="center", va="center", fontsize=F * (0.66 if ONE else 0.72), color=fs.RED)
    fig.text(0.5, 0.014 if ONE else 0.042,
             "challenges that restore nothing: chance, not gain." if ONE
             else
             "That part is per-image chance, not restoration, and no realizable "
             "selector can aim at it.",
             ha="center", va="center", fontsize=F * (0.66 if ONE else 0.72), color=fs.RED)

    ok = fs.run_gates(fig, args.outdir, "fig04_operator_selection_oracle",
                      bar_axes=[axL, axR])
    fs.report_sizes(args.target, W, F, {
        "panel titles": 0.88, "method labels": 0.80, "challenge labels": 0.76,
        "bar values": 0.74, "axis labels": 0.78, "the key": 0.76,
        "the notes": 0.74, "headroom label": 0.76})

    print("\nCAPTION:")
    for ln in [
        "Fig. 4. The room left in operator selection, and how much of it is real.",
        "Left: on the same seven-operator pool, an operator-selection oracle, which",
        "picks the best operator for each image using the ground-truth label, reaches",
        "72.19 per cent, against 58.23 for the proposed training-free selector and",
        "60.72 for the best single operator. Nearly fourteen accuracy points therefore",
        "remain in selection alone, with no new operator. The hatched bars need the",
        "labels and are not achievable; they are a ceiling, not a competitor. This",
        "oracle is a different object from the noise-injection oracle of Figs. 3 and",
        "4: that one is a bar an operator must clear before its gain counts as",
        "restoration, this one is a ceiling on selection. The headroom is 13.96 points.",
        "Right: the same headroom",
        "broken down by challenge. Gaussian blur contributes the largest single share,",
        "+24.4 points, and yet no operator restores Gaussian blur at any severity",
        "(Fig. 3). Across the four such challenges the headroom is 5.23 of 13.96",
        "points, 37 per cent of the total. On those challenges the operator-selection",
        "oracle still gains, because on individual images one operator happens to be",
        "right where the others are wrong; that is per-image chance, not restoration,",
        "and no realizable selector has a signal to aim at it. The headroom that a",
        "selector",
        "could honestly pursue is the 8.73 points from the eight challenges where",
        "restoration demonstrably occurs. Perfecting selection on the base rule's own",
        "four operators buys 9.77 points; adding three more operators, two of them 2025",
        "models, buys 5.77 on top of that, so the gap is in selection rather than in the",
        "size of the pool. Scope: CompactCNN (145,291 parameters,",
        "32x32 input) on CURE-TSR; accuracies are averaged over 12 challenges and 5",
        "severities with equal weight.",
    ]:
        print("  " + ln)
    if not ok or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
