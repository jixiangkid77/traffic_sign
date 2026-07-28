# -*- coding: utf-8 -*-
"""
FIG05_rule_and_its_worth.py
Paper figure 5. Script, output and paper number agree.

Figure 5: the routing rule, what it does to an image, and an exact accounting of
what each of its branches is worth.

WHY THIS IS THE METHOD FIGURE AND NOT A BLOCK DIAGRAM
A block diagram of enhance, resize, classify would carry no measurement. This
figure carries the pipeline in one strip and then spends its area on the thing a
reader cannot get from prose: the rule routes three quarters of degraded crops to
no enhancement at all, and that refusal, not a wrong choice, is where it loses to
simply running the dark channel prior on everything.

TWO DECOMPOSITIONS, BOTH EXACT
Each branch contributes to the degraded average in proportion to how many images
it takes inside every (challenge, severity) cell, so a difference between two
front ends splits over the four branches with no residual. Both splits below were
checked against the totals they must reproduce and agree to three decimals.

  what the rule earns over no enhancement          total +2.15
      gamma +1.09   CLAHE +0.99   stretch +0.08   passthrough 0 by construction

  what always-DCP would add over the selector V-B  total +2.49
      passthrough +2.32   stretch +0.99   CLAHE 0 (V-B already sends it to DCP)
      gamma -0.83, which is the rule's one real win: on dark crops gamma beats
      the dark channel prior, and routing them away from gamma would cost 0.83

So the selector is not wrong where it acts. It is expensive where it declines to
act, and the 75.5 per cent of degraded crops it leaves untouched hold 2.32 of the
2.49 points that separate it from always-DCP.

THE RULE, VERBATIM FROM F_master_sweep_cache.py
    b = gray.mean()/255      c = gray.std()/128      e = Canny(50,150).mean()/255
    b < 0.1206                        -> gamma
    elif c < 0.1061                   -> CLAHE      (V-B serves this with DCP)
    elif e < 0.0726 and b > 0.4085    -> stretch
    else                              -> passthrough
The three statistics and the operator both run on the crop at its ORIGINAL size,
before the resize to 32x32. That order is not cosmetic: computing them after the
resize changes which branch an image takes.

WHY THE CLEAN COST IS EXACTLY ZERO AND NOT NEARLY ZERO
On all 1352 clean crops the rule takes the passthrough branch, so its output is
bit-identical to no enhancement there. The figure states the count rather than
the word "negligible".

READS   outputs_revision/merged_per_image.csv   (K_merge_results.py)
        outputs_revision/dcp_cure.csv           (Q_dcp_branch.py)
WRITES  outputs_revision/figures/fig05_rule_and_its_worth.png   (400 dpi)
        outputs_revision/figures/fig05_rule_and_its_worth.svg
RUN     python FIG05_rule_and_its_worth.py
        python FIG05_rule_and_its_worth.py --target word
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle

import fig_style as fs

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

# Verbatim from F_master_sweep_cache.py; the figure prints them, it does not
# invent them.
T1, T2, T3, T4 = 0.1206, 0.1061, 0.0726, 0.4085
BRANCHES = ["gamma", "clahe", "stretch", "passthrough"]
PRETTY = {"gamma": "Gamma", "clahe": "CLAHE", "stretch": "Stretch",
          "passthrough": "no enhancement"}
# Four decimals, which is exactly how many the constants carry. Three would print
# T4 as 0.408 and someone implementing from the figure would route differently.
COND = {"gamma": f"b < {T1:.4f}",
        "clahe": f"c < {T2:.4f}",
        "stretch": f"e < {T3:.4f} and b > {T4:.4f}",
        "passthrough": "otherwise"}

GREEN = "#2E7D4F"      # what the rule earns over doing nothing
AMBER = "#C77B2B"      # what always-DCP would add on top of the rule
RED = "#B3261E"

EXPECT_ANCHOR = dict(passthrough=54.49, va_rule=56.65, vb=58.23, dcp=60.72)
EXPECT_SHARE = {"gamma": 5.63, "clahe": 11.92, "stretch": 6.99,
                "passthrough": 75.46}
EXPECT_EARNED = {"gamma": 1.091, "clahe": 0.989, "stretch": 0.075,
                 "passthrough": 0.000}
EXPECT_MISSED = {"gamma": -0.825, "clahe": 0.000, "stretch": 0.991,
                 "passthrough": 2.322}
EXPECT_CLEAN_ROWS = 1352


def load(merged_path, dcp_path):
    rows = list(csv.DictReader(open(merged_path, newline="", encoding="utf-8")))
    need = ["ch", "sev", "true", "rule_branch", "pred_passthrough",
            "pred_gamma", "pred_clahe", "pred_stretch", "pred_va_rule"]
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
        raise SystemExit(f"ABORT: {absent} rows have no DCP prediction; the two "
                         f"files are not the same run.")
    d = dict(ch=np.array([int(r["ch"]) for r in rows]),
             sev=np.array([int(r["sev"]) for r in rows]),
             tru=np.array([int(r["true"]) for r in rows]),
             br=np.array([r["rule_branch"] for r in rows]))
    for o in ("passthrough", "gamma", "clahe", "stretch", "va_rule"):
        d[o] = np.array([int(r[f"pred_{o}"]) for r in rows])
    d["dcp"] = np.array([dcp[k] for k in key])
    d["vb"] = np.where(d["br"] == "clahe", d["dcp"], d["va_rule"])
    return d


def split(a, b, d, cells):
    """Exact per-branch split of the degraded-average difference b minus a.

    Inside one cell the accuracy is a mean over its images, so a branch that
    holds a fraction of the cell moves the cell mean by that fraction of its own
    change. Averaging those contributions over cells reproduces the difference
    with no residual, which the audit checks rather than assumes.
    """
    out = {}
    for name in BRANCHES:
        per_cell = []
        for i in cells:
            m = d["br"][i] == name
            if m.sum() == 0:
                per_cell.append(0.0)
                continue
            hit_a = (a[i][m] == d["tru"][i][m]).mean()
            hit_b = (b[i][m] == d["tru"][i][m]).mean()
            per_cell.append(100.0 * (hit_b - hit_a) * m.sum() / len(i))
        out[name] = float(np.mean(per_cell))
    return out


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
    d = load(args.merged, args.dcp)

    cells = [np.where((d["ch"] == c) & (d["sev"] == s))[0]
             for c in range(1, 13) for s in range(1, 6)]
    if any(len(i) == 0 for i in cells):
        raise SystemExit("ABORT: a (challenge, severity) cell is empty.")

    def deg_avg(p):
        return 100.0 * float(np.mean([np.mean(p[i] == d["tru"][i])
                                      for i in cells]))

    anchor = {k: deg_avg(d[k]) for k in
              ("passthrough", "va_rule", "vb", "dcp")}
    deg = d["sev"] > 0
    n_deg = int(deg.sum())
    share = {b: 100.0 * float(np.mean(d["br"][deg] == b)) for b in BRANCHES}
    earned = split(d["passthrough"], d["va_rule"], d, cells)
    missed = split(d["vb"], d["dcp"], d, cells)

    clean = d["sev"] == 0
    n_clean = int(clean.sum())
    clean_branches = set(d["br"][clean].tolist())

    # ---------------- audit ----------------
    print("=== EXPECTED-OUTPUT AUDIT ===")
    bad = 0
    for k, e in EXPECT_ANCHOR.items():
        if abs(round(anchor[k], 2) - e) > 1e-9:
            print(f"  MISMATCH {k}: got {anchor[k]:.2f}, expected {e:.2f}")
            bad += 1
    for b in BRANCHES:
        for label, got, exp, nd in (("share", share[b], EXPECT_SHARE[b], 2),
                                    ("earned", earned[b], EXPECT_EARNED[b], 3),
                                    ("missed", missed[b], EXPECT_MISSED[b], 3)):
            if abs(round(got, nd) - exp) > 1e-9:
                print(f"  MISMATCH {b} {label}: got {got:.3f}, "
                      f"expected {exp:.3f}")
                bad += 1
    # the two splits must reproduce the totals they decompose, with no residual
    for label, got_sum, target in (
            ("rule over no enhancement", sum(earned.values()),
             anchor["va_rule"] - anchor["passthrough"]),
            ("always-DCP over V-B", sum(missed.values()),
             anchor["dcp"] - anchor["vb"])):
        if abs(got_sum - target) > 0.005:
            print(f"  MISMATCH split of {label}: parts sum to {got_sum:.3f}, "
                  f"total is {target:.3f}")
            bad += 1
        print(f"  split of {label}: parts {got_sum:+.3f}, "
              f"total {target:+.3f}")
    if n_clean != EXPECT_CLEAN_ROWS or clean_branches != {"passthrough"}:
        print(f"  MISMATCH clean crops: {n_clean} rows taking "
              f"{sorted(clean_branches)}, expected {EXPECT_CLEAN_ROWS} rows "
              f"taking passthrough only")
        bad += 1
    print(f"  {n_deg} degraded crops; branch shares " +
          ", ".join(f"{PRETTY[b]} {share[b]:.2f}%" for b in BRANCHES))
    print(f"  {n_clean} clean crops, all on the passthrough branch, so the "
          f"rule's clean output is bit-identical to no enhancement")
    print("  AUDIT PASSED: every value reproduces." if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    F = F * 1.10 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    H = 4.75 if args.target == "ieee" else (4.55 if ONE else 4.60)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    if ONE:
        # Three bands stacked. The rule and the branch bars sat side by side at
        # full width; a column cannot hold both and still show a condition like
        # "e < 0.0726 and b > 0.4085" at a size a reader can take in.
        gs = fig.add_gridspec(3, 1, height_ratios=[1.22, 1.60, 1.95],
                              left=0.052, right=0.988, top=0.975,
                              bottom=0.300, hspace=0.24)
        axP = fig.add_subplot(gs[0, 0])
        axL = fig.add_subplot(gs[1, 0])
        axR = fig.add_subplot(gs[2, 0])
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[1.00, 2.55],
                              width_ratios=[1.16, 1.00], left=0.052,
                              right=0.988, top=0.975, bottom=0.275,
                              wspace=0.115, hspace=0.2)
        axP = fig.add_subplot(gs[0, :])     # the pipeline strip
        axL = fig.add_subplot(gs[1, 0])     # the rule
        axR = fig.add_subplot(gs[1, 1])     # what each branch is worth

    # ---- pipeline strip ----
    axP.set_xlim(0, 100)
    axP.set_ylim(-0.4, 8.6)
    axP.axis("off")
    boxes = [(2, 20, "crop at its\noriginal size"),
             (26, 20, "front end"),
             (50, 18, "resize to\n32 x 32"),
             (72, 26, "CompactCNN\n145,291 param.")]
    for x, w, label in boxes:
        face = "#EAF1EC" if label == "front end" else "#F2F2F2"
        edge = GREEN if label == "front end" else "#BBBBBB"
        axP.add_patch(Rectangle((x, 2.2), w, 5.6, facecolor=face,
                                edgecolor=edge, lw=1.3, zorder=3))
        axP.text(x + w / 2, 5.0, label, ha="center", va="center",
                 fontsize=F * 0.68, color=fs.INK, zorder=4, linespacing=1.25)
    for x0, x1 in ((22, 26), (46, 50), (68, 72)):
        axP.add_patch(FancyArrow(x0, 5.0, x1 - x0 - 0.8, 0, width=0.12,
                                 head_width=0.9, head_length=1.4,
                                 length_includes_head=True, color="#8A8A8A",
                                 zorder=3))
    axP.text(50 if ONE else 36, 0.7,
             "the operator and the three statistics\nboth run here, before "
             "the resize" if ONE else
             "the operator and the three statistics both run here, "
             "before the resize", ha="center", va="center",
             fontsize=F * 0.64, color=GREEN, linespacing=1.25)

    # ---- the rule ----
    axL.set_xlim(0, 100)
    axL.set_ylim(len(BRANCHES) - 0.35, -1.25)
    axL.axis("off")
    axL.text(0, -1.05, "the rule, on the crop at its original size",
             ha="left", va="center", fontsize=F * 0.74, color=fs.INK)
    for i, b in enumerate(BRANCHES):
        axL.text(0, i, COND[b], ha="left", va="center", fontsize=F * 0.70,
                 color=fs.INK)
        axL.text(53 if ONE else 60, i, "\u2192", ha="center", va="center",
                 fontsize=F * 0.70,
                 color="#8A8A8A")
        # The share and the V-B substitution ride on the branch they belong to,
        # which keeps every string inside one panel and leaves nothing floating
        # between the two to collide with the legend below.
        tail = PRETTY[b] + f"  {share[b]:.1f}%"
        if b == "clahe" and not ONE:
            # At column width this rides off the panel. The caption names the
            # branch instead, which costs the caption six words and the figure
            # nothing it can show legibly.
            tail += "   V-B: DCP"
        axL.text(58 if ONE else 66, i, tail, ha="left", va="center",
                 fontsize=F * 0.70,
                 color=GREEN if b != "passthrough" else fs.MUTED)

    # ---- what each branch is worth ----
    for i, b in enumerate(BRANCHES):
        axR.barh(i - 0.16, earned[b], height=0.30, color=GREEN, zorder=3)
        axR.barh(i + 0.19, missed[b], height=0.30, color=AMBER, zorder=3)
    axR.axvline(0, color="#999999", lw=0.9, zorder=2)
    axR.set_ylim(len(BRANCHES) - 0.35, -1.25)
    axR.set_yticks([])
    axR.set_xlim(-1.35, 2.75)
    axR.set_xticks([-1, 0, 1, 2])
    axR.set_xlabel("points of degraded average",
                   fontsize=F * 0.70, labelpad=2)
    axR.tick_params(axis="x", labelsize=F * 0.68, length=2.5, pad=1.5)
    axR.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.spines["bottom"].set_color("#999999")
    axR.text(0.5, 1.055, "what each branch is worth", transform=axR.transAxes,
             ha="center", va="bottom", fontsize=F * 0.74, color=fs.INK)

    h = [Line2D([], [], color=GREEN, lw=5),
         Line2D([], [], color=AMBER, lw=5)]
    _lg = ([f"the rule over no enhancement, {sum(earned.values()):+.2f}",
            f"always-DCP over V-B, {sum(missed.values()):+.2f}"] if ONE else
           [f"the rule earns this over no enhancement, "
            f"{sum(earned.values()):+.2f} in total",
            f"always-DCP would add this over V-B, "
            f"{sum(missed.values()):+.2f} in total"])
    fig.legend(h, _lg,
               loc="lower center", bbox_to_anchor=(0.5, 0.131 if ONE else 0.098), ncol=1,
               frameon=False, fontsize=F * 0.68, handlelength=1.8,
               handletextpad=0.55, labelspacing=0.30)

    fig.text(0.5, 0.106 if ONE else 0.058,
             "The rule is not wrong where it acts.\nIt is expensive where it "
             "declines to." if ONE else
             "The rule is not wrong where it acts. It is expensive where it "
             "declines to.",
             ha="center", va="center", fontsize=F * 0.72, color=RED,
             linespacing=1.25)
    fig.text(0.5, 0.036 if ONE else 0.018,
             f"All {n_clean:,} clean crops take the passthrough branch,\nso "
             f"the clean cost is exactly zero." if ONE else
             f"All {n_clean:,} clean crops take the passthrough branch, so the "
             f"clean cost is exactly zero.",
             ha="center", va="center", fontsize=F * 0.66, color=fs.MUTED,
             linespacing=1.25)

    ok_gate = fs.run_gates(fig, args.outdir, "fig05_rule_and_its_worth",
                           bar_axes=[axR])
    fs.report_sizes(args.target, W, F, {
        "pipeline boxes": 0.68, "the rule": 0.70, "panel titles": 0.74,
        "branch shares": 0.68, "axis label": 0.70, "tick labels": 0.68,
        "the legend": 0.68, "the finding": 0.72, "the clean note": 0.66})

    print("\nCAPTION:")
    for ln in [
        "Fig. 5. The routing rule and what each of its branches is worth. Top: every",
        "front end runs on the crop at its original size, and the result is then",
        "resized to 32 x 32 for the classifier. The three routing statistics are",
        "computed there too, before the resize, since computing them afterwards changes",
        "which branch an image takes. Left: the rule itself, where b is mean grey level",
        "over 255, c is its standard deviation over 128 and e is the mean of a Canny",
        "edge map over 255. The selector V-B keeps this rule and serves the CLAHE",
        "branch with the dark channel prior instead. Right: what each branch",
        "contributes to the degraded average, in points, with the share of degraded",
        "crops it takes on the left of each pair. Green is what the rule earns over no",
        "enhancement, +2.15 in total, and amber is what running the dark channel prior",
        "on everything would add over V-B, +2.49 in total. Both splits are exact: a",
        "branch moves a cell mean by its own change times the fraction of the cell it",
        "holds, and the four parts reproduce the total they decompose to three",
        "decimals. The shape of the amber column is the paper's negative result about",
        "selection. The rule earns its keep where it acts, most of all on the gamma",
        "branch, where it is 0.83 points better than the dark channel prior would be on",
        "the same dark crops. What costs it is the 75.5 per cent of degraded crops it",
        "leaves untouched, which hold 2.32 of the 2.49 points that separate it from",
        "always-DCP. All 1,352 clean crops take the passthrough branch, so the rule's",
        "clean output is bit-identical to no enhancement and its clean cost is exactly",
        "zero rather than nearly zero. Scope: CompactCNN (145,291 parameters, 32 x 32",
        "input) on CURE-TSR; accuracies averaged over 12 challenges and 5 severities.",
    ]:
        print("  " + ln)
    if not ok_gate or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
