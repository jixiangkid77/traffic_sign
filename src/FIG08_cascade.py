# -*- coding: utf-8 -*-
"""
FIG08_cascade.py
Figure 7: what a deep model is still worth once the front end is good, and what it
costs to find out.

WHAT THE FIGURE SAYS
A confidence-gated cascade escalates an image to AdaIR (28.8M parameters) when the
front end's routing branch is one of the two hard ones and AdaIR is confident. The
pre-registered protocol (Protocol Part 3, frozen gate and frozen tau grid) selects
tau by nested five-fold cross-validation and reports the held-out estimate. Run on
two different front ends, it gives:

  base = frozen four-operator rule   56.65 -> 57.80 held out, the cascade buys +1.16
  base = DCP-augmented selector V-B  58.23 -> 58.38 held out, the cascade buys +0.15

The same cascade, the same deep model, the same gate. Nearly eight times less value
on the better front end. That is the collapse of marginal value: the deep model is
not adding capability, it is filling a hole a better front end has already filled.

Two further facts, both of them awkward for the deep model:

  AdaIR alone reaches 57.78, BELOW the training-free selector's 58.23. Escalating
  EVERY image to AdaIR therefore makes V-B worse, from 58.23 down to 57.78.

  At the operating point the cascade selected, V-B accepts AdaIR's answer on only
  4.7 per cent of degraded images, against 14.4 per cent for the weaker rule. The
  better front end both needs the deep model less often and gains less when it does.

EVERY DIFFERENCE HERE CARRIES A PAIRED BOOTSTRAP
Images are resampled inside their own (challenge, severity) cell, which is the unit
the degraded average is built from, 2000 draws at seed 42. V-B over AdaIR +0.45
[+0.25, +0.62]; the cascade on the frozen rule +1.16 [+1.06, +1.25]; the cascade on
V-B +0.16 [+0.11, +0.20]. All three hold. The verdicts are locked in EXPECT_CI, so
a claim that stops holding fails the audit instead of surviving in the prose.

NOTE TO SELF, not for the paper: the acceptance rates were previously written as
4.6 and 14.1, which is the same count over ALL 82472 images rather than the 81120
degraded ones the rest of the figure uses. Both are arithmetically right; only one
is consistent with the gate percentage beside it, so the degraded base is now used
and named. Restoring the transmission refinement moved V-B from 58.21 to 58.23 and
the cascade on it from +0.14 to +0.15, which is why the ratio is no longer a round
eight.

WHAT IS PRE-REGISTERED AND WHAT IS NOT
  Pre-registered : the gate (routing branch in {clahe, stretch}), the tau grid, the
                   nested five-fold protocol, and the two operating points marked
                   with stars. Their held-out values are reproduced here on the full
                   data to within 0.02 points.
  Post hoc       : nothing in the left panel any more. The sweep used to order
                   every crop by AdaIR's confidence and escalate the top k per
                   cent with no branch gate, which is a different policy from the
                   one the stars mark, and on shared axes the star for the frozen
                   rule sat a full point above its own curve. The sweep now runs
                   the deployed policy over the same tau grid the protocol froze,
                   so every point on the curve is a threshold the search could
                   have returned, and the star is one of them.

READS   outputs_revision/merged_per_image.csv     (K_merge_results.py)
        outputs_revision/dcp_cure.csv             (Q_dcp_branch.py)
        outputs_revision/M_multiseed_cascade.json (M_multiseed_cascade.py)
WRITES  outputs_revision/figures/fig08_cascade.png   (400 dpi)
        outputs_revision/figures/fig08_cascade.svg
RUN     python FIG08_cascade.py
        python FIG08_cascade.py --target word
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import fig_style as fs

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

CELLS = [(c, s) for c in range(1, 13) for s in range(1, 6)]
GATE_BRANCHES = ("clahe", "stretch")     # frozen, Protocol Part 3
TAU = {"rule": 0.2, "vb": 0.475}         # modal tau over the 50 folds of M
# The grid Protocol Part 3 froze, and the grid M searched. The sweep in this
# figure uses the same one, so that the star it marks is a point the search
# could have returned rather than a point on some other curve.
TAUS = np.round(np.arange(0.0, 0.6, 0.025), 3)

RULE_COL = "#B3261E"
VB_COL = "#5B7FA6"
ADAIR_COL = "#8C5FBF"

# Locked 2026-07-11 from the authoritative per-image files and M_multiseed_cascade.
EXPECT = dict(rule=56.65, vb=58.23, adair=57.78,
              rule_casc=57.80, vb_casc=58.39,
              rule_use=14.37, vb_use=4.67, gate_pct=18.9)

# Every "A is above B" in this figure is a claim about a difference, so each one
# carries a paired bootstrap over images within their own (challenge, severity)
# cell, which is the unit the degraded average is built from. Locking the verdict
# rather than the interval keeps the audit stable against the resampling noise of
# a finite B while still failing loudly if a claim stops holding.
BOOT_B, BOOT_SEED = 2000, 42
EXPECT_CI = {
    "vb_over_adair":  "above",   # the selector beats the deep model outright
    "rule_cascade":   "above",   # the cascade helps the weak front end
    "vb_cascade":     "above",   # and still helps the strong one, by much less
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS),
                    default="ieee1col")
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--mseed", default=str(OUT_DIR / "M_multiseed_cascade.json"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    for p in (args.merged, args.dcp, args.mseed):
        if not Path(p).exists():
            raise SystemExit(f"ABORT: {p} not found.")
    rows = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    dcp_map = {}
    for r in csv.DictReader(open(args.dcp, newline="", encoding="utf-8")):
        dcp_map[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = \
            int(r["pred_dcp"])
    mseed = json.load(open(args.mseed))

    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    br = np.array([r["rule_branch"] for r in rows])
    va = np.array([int(r["pred_va_rule"]) for r in rows])
    ad = np.array([int(r["pred_adair"]) for r in rows])
    pa = np.array([float(r["prob_adair"]) for r in rows])
    dcp = np.array([dcp_map[(r["filename"], int(r["occ"]), int(r["ch"]),
                             int(r["sev"]))] for r in rows])
    vb = np.where(br == "clahe", dcp, va)

    deg = np.isin(ch, range(1, 13)) & np.isin(sev, range(1, 6))
    masks = [(ch == c) & (sev == s) for c, s in CELLS]
    # the same cells as integer indices, for the paired bootstrap below
    idx_cells = [np.where(m)[0] for m in masks]

    def deg_avg(pred):
        return float(np.mean([100.0 * np.mean(pred[m] == tru[m]) for m in masks]))

    base = {"rule": va, "vb": vb}
    acc = {k: deg_avg(v) for k, v in base.items()}
    acc["adair"] = deg_avg(ad)
    gate = np.isin(br, GATE_BRANCHES)
    gate_pct = 100.0 * float(np.mean(gate[deg]))

    # ---- the two pre-registered operating points, reproduced on the full data ----
    op = {}
    for k in ("rule", "vb"):
        use = gate & (pa > TAU[k])
        op[k] = dict(rate=100.0 * float(np.mean(use[deg])),
                     acc=deg_avg(np.where(use, ad, base[k])))
        op[k]["gain"] = op[k]["acc"] - acc[k]

    # ---- the sweep: the same gated cascade at every tau on the frozen grid ----
    # It used to escalate the top k per cent of ALL crops by AdaIR's confidence,
    # with no branch gate. That is a different policy from the one the stars
    # mark, and on the same axes the star for the frozen rule sat a full point
    # above its own curve, which a reader can only read as an error. The curve
    # is now the policy the article deploys, swept over the grid the protocol
    # froze, so the star is a point on it by construction.
    #
    # The grid is TAUS. tau = 1 is added at the left as the endpoint where
    # nothing is escalated, which is the front end on its own; it is not a
    # candidate the protocol could have selected, and it is where each curve
    # starts rather than a point the search considered.
    grid = list(TAUS) + [1.0]
    curve = {k: [] for k in ("rule", "vb")}
    rates = []
    for t in sorted(grid, reverse=True):
        m = gate & (pa > t)
        # Over degraded crops, which is the population the y axis is built from.
        # It was over every crop, clean ones included, so the curve sat a
        # quarter of a point to the left of the star it was supposed to pass
        # through, and the share the gate admits printed as 18.6 here against
        # the 18.9 the article gives. Two denominators, one axis.
        rates.append(100.0 * float(np.mean(m[deg])))
        for k in ("rule", "vb"):
            curve[k].append(deg_avg(np.where(m, ad, base[k])))
    ks = np.array(rates)

    # ---------------- audit ----------------
    print("=== EXPECTED-OUTPUT AUDIT (against values locked 2026-07-11) ===")
    bad = 0
    got = dict(rule=acc["rule"], vb=acc["vb"], adair=acc["adair"],
               rule_casc=op["rule"]["acc"], vb_casc=op["vb"]["acc"],
               rule_use=op["rule"]["rate"], vb_use=op["vb"]["rate"],
               gate_pct=gate_pct)
    for k, e in EXPECT.items():
        d = 0.01 if k.endswith("use") or k == "gate_pct" else 0.005
        if abs(round(got[k], 2) - e) > 0.011:
            print(f"  MISMATCH {k}: got {got[k]:.4f} (prints as {got[k]:.2f}), "
                  f"expected {e:.2f}")
            bad += 1
    # cross-check against the multi-seed held-out estimates
    hm = {"frozen 4-operator rule": "rule", "DCP-augmented selector (V-B)": "vb"}
    for name, key in hm.items():
        b = mseed["bases"][name]
        print(f"  {name}: held out over {len(mseed['seeds'])} seeds "
              f"{b['heldout_mean']:.2f} (sd {b['heldout_std_across_seeds']:.3f}), "
              f"cascade buys {b['paired_mean']:+.2f}; full data here "
              f"{op[key]['acc']:.2f} ({op[key]['gain']:+.2f})")
        if abs(b["heldout_mean"] - op[key]["acc"]) > 0.05:
            print(f"    MISMATCH: full-data reproduction differs from held out by "
                  f"{abs(b['heldout_mean'] - op[key]['acc']):.2f}")
            bad += 1
    # The old check was that both curves reach AdaIR at 100 per cent escalation,
    # which only held because the old curve ignored the branch gate. The check
    # that matters now is stronger: the star each panel marks has to be the
    # highest point of its own curve, because the threshold was chosen to
    # maximise exactly this quantity on the same grid. If it is not, either the
    # selection or the sweep is wrong.
    for k in ("rule", "vb"):
        j = int(np.argmax(curve[k]))
        if abs(ks[j] - op[k]["rate"]) > 0.005:
            print(f"  MISMATCH {k}: the star is at {op[k]['rate']:.2f} per cent "
                  f"and the curve peaks at {ks[j]:.2f}; the two rates are not "
                  f"over the same population")
            bad += 1
        if abs(curve[k][j] - op[k]["acc"]) > 0.0005:
            print(f"  MISMATCH {k}: the star sits at {op[k]['acc']:.4f} but the "
                  f"curve peaks at {curve[k][j]:.4f} ({ks[j]:.2f} per cent)")
            bad += 1
        else:
            print(f"  {k}: the selected threshold is the highest point of its "
                  f"own curve, at {ks[j]:.2f} per cent escalated")
    # And the left endpoint is the front end with nothing escalated.
    for k in ("rule", "vb"):
        if abs(curve[k][0] - acc[k]) > 0.0005:
            print(f"  MISMATCH {k}: the curve starts at {curve[k][0]:.4f}, the "
                  f"front end alone is {acc[k]:.4f}")
            bad += 1
    print(f"  gate covers {gate_pct:.1f} per cent of degraded images; the rule "
          f"accepts AdaIR on {op['rule']['rate']:.1f} per cent and V-B on "
          f"{op['vb']['rate']:.1f} per cent")

    # ---- paired bootstrap for the three differences the figure asserts ----
    # Images are resampled inside their own cell, so the resample keeps the shape
    # of the statistic being reported: a mean over the 60 cells of per-cell
    # accuracy, not a pooled rate over unequal cells.
    ok = {"vb": (vb == tru), "adair": (ad == tru),
          "rule": (va == tru),
          "rule_casc": (np.where(gate & (pa > TAU["rule"]), ad, va) == tru),
          "vb_casc": (np.where(gate & (pa > TAU["vb"]), ad, vb) == tru)}
    rng = np.random.default_rng(BOOT_SEED)
    draws = {k: np.empty(BOOT_B) for k in EXPECT_CI}
    for i in range(BOOT_B):
        pick = [rng.integers(0, len(m), len(m)) for m in idx_cells]
        d = {k: 100.0 * float(np.mean([v[m[q]].mean()
                                       for m, q in zip(idx_cells, pick)]))
             for k, v in ok.items()}
        draws["vb_over_adair"][i] = d["vb"] - d["adair"]
        draws["rule_cascade"][i] = d["rule_casc"] - d["rule"]
        draws["vb_cascade"][i] = d["vb_casc"] - d["vb"]
    point = {"vb_over_adair": acc["vb"] - acc["adair"],
             "rule_cascade": op["rule"]["gain"],
             "vb_cascade": op["vb"]["gain"]}
    for k, exp in EXPECT_CI.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        got_v = "above" if lo > 0 else ("below" if hi < 0 else "level")
        if got_v != exp:
            print(f"  MISMATCH {k}: bootstrap says {got_v}, expected {exp}")
            bad += 1
        print(f"  {k:14s} {point[k]:+.3f}  95 per cent CI [{lo:+.3f}, {hi:+.3f}]"
              f"  -> {got_v}")
    print("  AUDIT PASSED: every value reproduces, each star is the peak of "
          "its own curve, and each curve starts at its front end." if bad == 0
          else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    # Full-width (two-column-spanning) float; bump the base font for readability
    # at that width, matching the other figures in the set.
    F = F * 1.10 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    H = 5.30 if args.target == "ieee" else (5.01 if ONE else 4.80)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    if ONE:
        # Side by side survives the column here, where the accuracy-cost figure
        # could not: that one needed five decades of a log axis, this one needs
        # a percentage from nought to a hundred, and three ticks carry it.
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.00], left=0.128,
                              right=0.962, top=0.916, bottom=0.567,
                              wspace=0.42)
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.60, 1.00], left=0.088,
                              right=0.985, top=0.88, bottom=0.428, wspace=0.36)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- left: accuracy against escalation rate ----
    axL.axhline(acc["adair"], color=ADAIR_COL, lw=1.4, ls="--", zorder=3)
    axL.plot(ks, curve["rule"], color=RULE_COL, lw=2.2, zorder=5)
    axL.plot(ks, curve["vb"], color=VB_COL, lw=2.2, zorder=5)
    for k, col in (("rule", RULE_COL), ("vb", VB_COL)):
        axL.plot([op[k]["rate"]], [op[k]["acc"]], marker="*", ms=F * 1.15,
                 color=col, mec="white", mew=0.8, zorder=8)
    axL.set_ylim(55.8, 59.4)
    # Three ticks at column width, six at full width. Six would sit a fifth of
    # an inch apart here and "100" is that wide on its own.
    # The gate admits 18.6 per cent of degraded crops, so that is where the sweep
    # ends; an axis to a hundred would be nine tenths empty.
    axL.set_xlim(-0.9, gate_pct + 1.2)
    axL.set_xticks([0, 5, 10, 15] if ONE else [0, 5, 10, 15, 18.6])
    axL.set_yticks([56, 57, 58, 59])
    axL.tick_params(labelsize=F * 0.76, length=2.5, pad=1.5)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axL.spines[sp].set_color("#999999")
    axL.set_xlabel("per cent of degraded\ncrops sent to AdaIR" if ONE else
                   "per cent of degraded crops sent to AdaIR",
                   fontsize=F * (0.86 if ONE else 0.78),
                   labelpad=2)
    axL.set_ylabel("degraded-average accuracy (%)", fontsize=F * 0.78, labelpad=2)
    # At column width the right panel is about an inch wide and its title has to
    # fit in it; "the same cascade," needs eight and a half point there. The
    # caption says the two panels are the same cascade on two front ends, which
    # costs the caption six words and the panel nothing.
    axL.set_title("accuracy against\nescalation rate",
                  fontsize=F * (0.94 if ONE else 0.88), pad=5,
                  linespacing=1.2)

    # ---- right: what the cascade buys, on each front end ----
    labels = ["frozen rule", "selector V-B"]
    gains = [mseed["bases"]["frozen 4-operator rule"]["paired_mean"],
             mseed["bases"]["DCP-augmented selector (V-B)"]["paired_mean"]]
    sds = [mseed["bases"]["frozen 4-operator rule"]["paired_std_across_seeds"],
           mseed["bases"]["DCP-augmented selector (V-B)"]["paired_std_across_seeds"]]
    xs = np.arange(2)
    axR.bar(xs, gains, width=0.55, color=[RULE_COL, VB_COL], zorder=3)
    axR.errorbar(xs, gains, yerr=sds, fmt="none", ecolor="#333333", capsize=3,
                 lw=1.0, zorder=5)
    for x, g in zip(xs, gains):
        _bar_value_probe = axR.text(
            x, g + 0.045, f"{g:+.2f}", ha="center", va="bottom",
            fontsize=F * (0.86 if ONE else 0.80), color=fs.INK,
            fontweight="bold", zorder=6)
    axR.set_xticks(xs)
    axR.set_xticklabels([l.replace(" ", "\n") for l in labels] if ONE
                        else labels,
                        fontsize=F * (0.80 if ONE else 0.78),
                        linespacing=1.15)
    axR.set_ylim(0, 1.45)
    axR.set_yticks([0, 0.5, 1.0])
    axR.set_yticklabels(["0", "0.5", "1.0"])
    axR.tick_params(axis="y", labelsize=F * 0.76, length=2.5, pad=1.5)
    axR.tick_params(axis="x", length=0, pad=3)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axR.spines[sp].set_color("#999999")
    axR.set_ylabel("what the cascade buys (points)", fontsize=F * 0.78, labelpad=2)
    # Two lines. On one line this title is 2.73 in wide at this size and the panel
    # is only 2.09 in, so it ran off the canvas. Same defect class as the panel
    # titles in Fig. 4: width is scarce, height is free.
    axR.set_title("two front ends" if ONE else
                  "the same cascade,\ntwo front ends",
                  fontsize=F * (0.94 if ONE else 0.88), pad=5,
                  linespacing=1.2)

    h = [Line2D([], [], color=RULE_COL, lw=2.2),
         Line2D([], [], color=VB_COL, lw=2.2),
         Line2D([], [], color=ADAIR_COL, lw=1.4, ls="--"),
         Line2D([], [], color="#555555", lw=0, marker="*", ms=F * 0.95,
                mec="white", mew=0.6)]
    # One column. Two columns need 8.87 in on a 7.16 in canvas, because the fourth
    # entry is long and every entry must keep its full wording.
    fig.legend(h, ["base: frozen 4-operator rule",
                   "base: DCP-augmented selector V-B",
                   "AdaIR alone, applied to every image",
                   ("the operating point the cascade selected" if ONE else
         "the operating point the cascade selected, pre-registered")],
               loc="lower center", bbox_to_anchor=(0.5, 0.3140 if ONE else 0.181), ncol=1,
               frameon=False, fontsize=F * 0.76, handlelength=2.0,
               handletextpad=0.55, labelspacing=0.34)

    # Each of these runs to five inches set on one line, so at column width each
    # is broken where it divides on its own sense rather than set smaller: the
    # finding is what the figure is for.
    _finding_probe = fig.text(0.5, 0.1689 if not ONE else 0.2754,
             "The same cascade buys +1.16 on the frozen\nrule and only +0.15 on the selector." if ONE else
             "The same cascade buys +1.16 on the frozen rule and +0.15 on V-B, "
             "nearly eight times less.",
             ha="center", va="center", fontsize=F * (0.94 if ONE else 0.76),
             color=fs.RED, linespacing=1.25)
    fig.text(0.5, 0.1330 if not ONE else 0.1876,
             "The deep model is not adding capability;\nit fills a hole a better front end fills." if ONE else
             "The deep model is not adding capability; it is filling a hole a "
             "better front end has already filled.",
             ha="center", va="center", fontsize=F * (0.94 if ONE else 0.76),
             color=fs.RED, linespacing=1.25)
    fig.text(0.5, 0.0858 if not ONE else 0.0978,
             "AdaIR alone reaches 57.78 (dashed),\nbelow the selector's own 58.23." if ONE else
             "AdaIR alone reaches 57.78, below V-B's 58.23, so escalating EVERY "
             "image makes V-B worse.",
             ha="center", va="center", fontsize=F * (0.90 if ONE else 0.76),
             color=fs.INK, linespacing=1.25)
    _caveat_probe = fig.text(0.5, 0.0311 if not ONE else 0.0299,
             f"the gate admits {gate_pct:.1f} per cent; the curve is all of it" if ONE else
             f"The gate admits {gate_pct:.1f} per cent of degraded crops and the curve covers "
             "all of it; the star is the threshold the protocol selected.",
             ha="center", va="center", fontsize=F * (0.78 if ONE else 0.72),
             color=fs.MUTED)

    ok = fs.run_gates(fig, args.outdir, "fig08_cascade", bar_axes=[axR],
                      line_axes=[axL])
    # The multipliers are read back off the artists that were actually drawn,
    # not copied from the calls above. Copied, they have gone stale three times
    # in this set of figures, and a size report that disagrees with the page is
    # worse than none: it is a check that passes while the thing it checks is
    # wrong.
    def _pt(obj):
        return round(obj.get_fontsize() / F, 4)
    fs.report_sizes(args.target, W, F, {
        "panel titles": _pt(axL.title),
        "axis labels": _pt(axL.xaxis.label),
        "tick labels": _pt(axL.get_xticklabels()[0]),
        "bar values": _pt(_bar_value_probe),
        "front-end labels": _pt(axR.get_xticklabels()[0]),
        "the findings": _pt(_finding_probe),
        "the caveat": _pt(_caveat_probe)})

    # This block is the figure's own account of itself, and it is the third
    # thing in this project to describe a state the code had already left: it
    # still called the curves a post hoc sweep with no branch gate, and still
    # said both meet AdaIR at a hundred per cent, after the sweep had been
    # changed to run inside the gate and to stop at 18.9. A caption that a
    # script prints is a comment, and a comment is what someone believed when
    # they wrote it, not what the code does now.
    print("\nCAPTION (this is the script's own account; the article's caption")
    print("is in _papercaps.json and is written separately):")
    for ln in [
        "Fig. 8. What a deep model is still worth once the front end is good. A",
        "confidence-gated cascade sends a crop to AdaIR when the front end's",
        "routing branch is one of the two hard ones, and keeps AdaIR's answer only",
        "where AdaIR is confident about it. The gate, the tau grid and the nested",
        "five-fold protocol are pre-registered (Protocol Part 3); tau is selected on",
        "the development folds and the held-out fold is reported, over ten",
        f"independent fold assignments. Right: on the frozen four-operator rule the",
        f"cascade buys +1.16 points (56.65 to 57.80 held out, standard deviation",
        f"0.00 across seeds); on the DCP-augmented selector V-B it buys +0.15 (58.23",
        f"to 58.38, standard deviation 0.003). The bars carry that standard",
        f"deviation, which measures how stable the protocol is across fold",
        f"assignments rather than how precisely the gain is known; a paired bootstrap",
        f"over images, on the full data, puts the two gains at 1.06 to 1.25 and 0.11",
        f"to 0.20. The same cascade, the same deep model, the same gate, and nearly",
        f"eight times less value on the better front end: the deep model is not",
        f"adding capability, it is filling a hole a better front end has already",
        f"filled. Left: degraded-average accuracy against the share of DEGRADED crops",
        f"sent to AdaIR, both axes over the same population. Each curve is the same",
        f"gated cascade at every tau on the frozen grid, so the star, the tau the",
        f"cross-validation selected, is the highest point of its own curve. The",
        f"curves stop at {gate_pct:.1f} per cent because that is the whole of what the",
        f"branch gate admits. AdaIR alone reaches 57.78, below V-B's 58.23 by 0.45",
        f"points with a paired bootstrap interval of 0.25 to 0.62, so escalating",
        f"every image to a 28.8M parameter model makes the training-free selector",
        f"worse. At its operating point V-B accepts AdaIR's answer on 4.7 per cent of",
        f"degraded crops against 14.4 per cent for the weaker rule: the better front",
        f"end needs the deep model less often and gains less when it does. Scope:",
        f"CompactCNN (145,291 parameters, 32x32 input) on CURE-TSR; accuracies",
        f"averaged over 12 challenges and 5 severities.",
    ]:
        print("  " + ln)
    if not ok or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
