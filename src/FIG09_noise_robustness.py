# -*- coding: utf-8 -*-
"""
FIG09_noise_robustness.py
Figure 6: what additive sensor noise does to each front end. This figure reports a
FAILURE, and reports it plainly.

WHAT THE FIGURE SAYS
CURE-TSR renders haze and rain as noise-free alpha-composited veils. Real cameras
do not. The dark channel prior recovers J = (I - A)/t + A, a division by the
transmission t, which multiplies whatever noise is in the input by 1/t. Sensor
noise is therefore the single factor most likely to break the transfer of the
rendered-veil result, and this experiment was pre-registered to measure it
(Evaluation Protocol Part 12, registered 2026-07-09, before execution).

Three findings, all of them uncomfortable, all of them reported:

  1. On haze the BASE RULE falls below doing nothing at sigma 4: 28.33 against
     31.47 for no enhancement, and 24.20 against 31.33 at sigma 8. Under sensor
     noise the base rule is not merely weaker, it is actively harmful. The
     mechanism is visible in the same panel: the rule sends 74 per cent of haze
     images to CLAHE, and CLAHE is destroyed by noise, from 49.47 down to 6.00.

  2. On RAIN the base rule is below no enhancement at EVERY sigma, including zero:
     27.00 against 28.67 here, and 34.11 against 35.31 on the full data set, so
     this is not an artefact of the 500-image subsample. The rule sends 40 per cent
     of rain images to CLAHE, and CLAHE reaches only 15.60 on rain against 28.67
     for doing nothing. The selector V-B, which serves that branch with DCP
     instead, repairs it: 35.33 at sigma 0, and it stays above no enhancement at
     sigma 4 and 8 on both challenges.

  3. But at sigma 16 the training-free advantage does NOT survive. On haze AdaIR
     overtakes DCP, 26.20 against 21.40, and a paired exact McNemar test on the
     same 1500 images separates that from chance (p = 3e-06). On rain the two are
     level, 24.67 against 23.27, a gap the same test cannot separate from chance
     (p = 0.26), so the reversal is claimed for haze alone. The learned model is
     the more noise-tolerant one at that level, and the manuscript must not claim
     a training-free advantage that survives it.

     NOTE TO SELF, not for the paper: restoring the transmission refinement lifted
     DCP on rain at sigma 16 from 21.73 to 23.27 and cut the gap from 2.93 points
     to 1.40, which took it below significance (it was p = 0.018 before). An
     earlier draft claimed the reversal on both challenges, which the data no
     longer supports. The verdict is now computed by mcnemar_exact and audited
     against EXPECT_MCNEMAR rather than asserted in prose.

ROUTING IS ITSELF NOISE-DEPENDENT, SO EVERY SHARE CARRIES ITS SIGMA
The rule reads its three features from the image it is handed, and noise moves
them, so the branch mix is not fixed across the grid. To CLAHE on haze: 74.3 per
cent at sigma 0, 72.1 at 4, 66.3 at 8, 42.1 at 16. On rain: 39.5, 36.9, 28.6,
4.9. Any share quoted in the text must name the sigma it belongs to. Note also
that on rain CLAHE is never the majority branch; passthrough is. An earlier
version of the figure note said the rule sends "most" rain images to CLAHE,
which is false at every sigma.

SCOPE, WHICH THE FIGURE AND THE CAPTION BOTH STATE
  challenges : haze and rain ONLY, the two veil-rendered challenges. The other ten
               challenges were NOT tested under noise. Nothing here may be
               generalised to them.
  severities : 3, 4, 5 only.
  sample     : 500 images per (challenge, severity) cell, seed 42.
  noise      : additive zero-mean Gaussian on the uint8 BGR image at the original
               crop resolution, applied BEFORE any operator, then clipped. This is
               NOT a sensor-noise model: there is no Poisson component and no
               read-noise floor. It is a sensitivity probe, not a camera simulation.
  This is a necessary, not a sufficient, transfer check. It does not test the
  veil-rendering circularity itself, which is disclosed separately.

READS   outputs_revision/U_noise_selector.csv   (U_noise_selector.py)
WRITES  outputs_revision/figures/fig09_noise_robustness.png   (400 dpi)
        outputs_revision/figures/fig09_noise_robustness.svg
RUN     python FIG09_noise_robustness.py
        python FIG09_noise_robustness.py --target word
"""
import argparse
import csv
import math
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

CH = {12: "Haze", 9: "Rain"}
SIGMAS = [0, 4, 8, 16]

# name, csv column, colour, line width, style, marker, z
SERIES = [
    ("no enhancement",   "passthrough", "#8A8A8A", 1.6, (0, (4, 2)), "o", 3),
    ("CLAHE",            "clahe",       "#E39898", 1.6, "-",         "^", 3),
    ("base rule",        "va",          "#B3261E", 2.4, "-",         "s", 6),
    ("selector V-B",     "vb",          "#5B7FA6", 2.4, "-",         "D", 6),
    ("DCP",              "dcp",         "#2E7D4F", 1.8, "-",         "o", 4),
    ("AdaIR",            "adair",       "#8C5FBF", 1.8, "-",         "v", 4),
]

def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the two discordant counts.

    b images the first method gets right and the second does not, c the reverse.
    Under the null the b of b + c discordant pairs are a fair coin, so the p value
    is the total probability of every outcome no more likely than the observed one.
    Written out rather than imported so the figure needs nothing beyond numpy and
    matplotlib, and so the test that decides a claim in the caption is visible in
    the same file as the claim.
    """
    n = b + c
    if n == 0:
        return 1.0
    obs = math.comb(n, b) * 0.5 ** n
    tot = sum(math.comb(n, k) * 0.5 ** n
              for k in range(n + 1)
              if math.comb(n, k) * 0.5 ** n <= obs * (1 + 1e-12))
    return min(1.0, tot)


# Locked from U_noise_selector.csv, recomputed image by image. The four DCP and
# V-B rows moved when the transmission refinement was restored; the eight rows
# that do not touch DCP are bit-identical, which is what the rerun should do.
EXPECT = {
    ("Haze", "passthrough"): [29.53, 31.47, 31.33, 13.67],
    ("Haze", "clahe"):       [49.47, 23.53, 13.67, 6.00],
    ("Haze", "va"):          [49.93, 28.33, 24.20, 12.87],
    ("Haze", "vb"):          [65.93, 50.27, 35.67, 14.33],
    ("Haze", "dcp"):         [69.47, 52.93, 37.67, 21.40],
    ("Haze", "adair"):       [57.00, 36.53, 29.33, 26.20],
    ("Rain", "passthrough"): [28.67, 29.73, 32.07, 26.07],
    ("Rain", "clahe"):       [15.60, 13.27, 9.00, 2.13],
    ("Rain", "va"):          [27.00, 27.87, 29.07, 25.53],
    ("Rain", "vb"):          [35.33, 36.33, 33.07, 25.80],
    ("Rain", "dcp"):         [44.87, 45.53, 38.53, 23.27],
    ("Rain", "adair"):       [29.67, 25.60, 25.93, 24.67],
}

# DCP against AdaIR at each noise level, paired on the same images. "level" means
# the exact test cannot separate the two at 0.05; it does not mean the point
# estimates are equal. This figure reports a LIMITATION, so an overstated reversal
# is as wrong as a hidden one, and the verdict is computed rather than asserted.
EXPECT_MCNEMAR = {
    ("Haze", 0): "DCP",   ("Haze", 4): "DCP",   ("Haze", 8): "DCP",
    ("Haze", 16): "AdaIR",
    ("Rain", 0): "DCP",   ("Rain", 4): "DCP",   ("Rain", 8): "DCP",
    ("Rain", 16): "level",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS),
                    default="ieee1col")
    ap.add_argument("--csv", default=str(OUT_DIR / "U_noise_selector.csv"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(f"ABORT: {args.csv} not found. It is written by "
                         f"U_noise_selector.py.")
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    need = ["ch", "sigma", "true"] + [f"pred_{c}" for _, c, *_ in SERIES]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise SystemExit(f"ABORT: {args.csv} has no columns {missing}.")

    ch = np.array([int(r["ch"]) for r in rows])
    sg = np.array([int(r["sigma"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    P = {c: np.array([int(r[f"pred_{c}"]) for r in rows]) for _, c, *_ in SERIES}

    acc, bad = {}, 0
    print("=== EXPECTED-OUTPUT AUDIT (recomputed image by image) ===")
    for cid, cname in CH.items():
        for _, col, *_ in SERIES:
            v = []
            for s in SIGMAS:
                m = (ch == cid) & (sg == s)
                if m.sum() == 0:
                    raise SystemExit(f"ABORT: no rows for {cname} sigma {s}.")
                v.append(100.0 * float(np.mean(P[col][m] == tru[m])))
            acc[(cname, col)] = v
            e = EXPECT[(cname, col)]
            for got, exp, s in zip(v, e, SIGMAS):
                if abs(round(got, 2) - exp) > 1e-9:
                    print(f"  MISMATCH {cname} {col} sigma {s}: got {got:.2f}, "
                          f"expected {exp:.2f}")
                    bad += 1
    n_cells = len(set(zip(ch.tolist(), sg.tolist())))
    print(f"  {len(rows)} rows, {n_cells} (challenge, sigma) cells, "
          f"{len(rows)//n_cells} images each")
    mcn = {}
    for cid, cname in CH.items():
        p, va, vb = acc[(cname, "passthrough")], acc[(cname, "va")], \
            acc[(cname, "vb")]
        below = [s for s, a, b in zip(SIGMAS, va, p) if a < b]
        print(f"  {cname}: the base rule is BELOW no enhancement at sigma "
              f"{below if below else 'never'}")
        for s in SIGMAS:
            m = (ch == cid) & (sg == s)
            d_ok = P["dcp"][m] == tru[m]
            a_ok = P["adair"][m] == tru[m]
            nb = int(np.sum(d_ok & ~a_ok))
            nc = int(np.sum(~d_ok & a_ok))
            pv = mcnemar_exact(nb, nc)
            lead = 100.0 * float(np.mean(a_ok) - np.mean(d_ok))
            got = ("AdaIR" if lead > 0 else "DCP") if pv < 0.05 else "level"
            mcn[(cname, s)] = (lead, pv, got)
            exp = EXPECT_MCNEMAR[(cname, s)]
            if got != exp:
                print(f"  MISMATCH {cname} sigma {s}: McNemar says {got}, "
                      f"expected {exp}")
                bad += 1
            ahead = "AdaIR" if lead > 0 else "DCP"
            print(f"  {cname} sigma {s:2d}: {ahead} ahead by {abs(lead):5.2f} "
                  f"(discordant {nb} / {nc}, p = {pv:.2e}) -> {got}")
    print("  AUDIT PASSED: every value reproduces." if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    # Full-width (two-column-spanning) float; bump the base font for readability
    # at that width, matching the other figures in the set.
    F = F * 1.10 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    H = 5.30 if args.target == "ieee" else (4.21 if ONE else 4.80)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    # Side by side survives the column: each panel carries four tick positions,
    # not a continuous axis, and the two share one y axis.
    gs = fig.add_gridspec(1, 2, left=(0.148 if ONE else 0.093),
                          right=(0.982 if ONE else 0.988),
                          top=(0.9287 if ONE else 0.951),
                          bottom=(0.5511 if ONE else 0.430),
                          wspace=(0.120 if ONE else 0.185))
    AXES = []
    x = np.arange(len(SIGMAS))

    for i, (cid, cname) in enumerate(CH.items()):
        ax = fig.add_subplot(gs[0, i])
        AXES.append(ax)
        pas = acc[(cname, "passthrough")]

        # Fill the gap between the base rule and doing nothing wherever the rule is
        # the WORSE of the two. A shaded column would only say "here"; the fill says
        # "here, and by this much". On rain the fill never closes, not even at
        # sigma 0: the base rule is below no enhancement on rain with no noise at
        # all (27.00 against 28.67 here, and 34.11 against 35.31 on the full set).
        va = acc[(cname, "va")]
        worse = [a < b for a, b in zip(va, pas)]
        ax.fill_between(x, va, pas, where=worse, interpolate=True,
                        color="#B3261E", alpha=0.16, lw=0, zorder=1)
        # The fill alone is not enough. On rain the base rule is below no
        # enhancement by only 0.5 to 3.0 points, which is thinner than the lines
        # that bound it, so the fill is hidden underneath them and the panel would
        # silently contradict the sentence printed below the figure. This strip
        # states the verdict independently of its magnitude.
        for j, w in enumerate(worse):
            if w:
                ax.add_patch(Rectangle((j - 0.40, 71.5), 0.80, 4.0,
                                       facecolor="#B3261E", edgecolor="none",
                                       zorder=7))

        for label, col, colour, lw, ls, mk, z in SERIES:
            ax.plot(x, acc[(cname, col)], color=colour, lw=lw, ls=ls, marker=mk,
                    ms=F * 0.30, mfc="white", mew=1.2, zorder=z, clip_on=True)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SIGMAS])
        ax.set_xlim(-0.5, len(SIGMAS) - 0.5)
        ax.set_ylim(0, 78)
        ax.set_yticks([0, 20, 40, 60])
        ax.tick_params(labelsize=F * 0.76, length=2.5, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#999999")
        # At column width each panel is under an inch and a half; the full sentence
        # needs two of them.
        ax.set_xlabel("added noise sigma" if ONE else
                      "sigma of the added Gaussian noise",
                      fontsize=F * (0.86 if ONE else 0.78),
                      labelpad=2)
        if i == 0:
            ax.set_ylabel("accuracy (%)", fontsize=F * (0.90 if ONE else 0.78),
                          labelpad=2)
        elif ONE:
            # Both panels run 0 to 60 on the same scale. Printing the numbers
            # twice takes a fifth of an inch off the right panel to say what the
            # left one has already said.
            ax.set_yticklabels([])
        ax.set_title(cname if ONE else f"{cname},  severities 3 to 5",
                     fontsize=F * (1.02 if ONE else 0.88), pad=5)

    handles = [Line2D([], [], color=c, lw=lw, ls=ls, marker=mk, ms=F * 0.30,
                      mfc="white", mew=1.2, label=lab)
               for lab, _, c, lw, ls, mk, _ in SERIES]
    # The shaded band is explained in the red note below, in the same words the
    # legend would use. At column width the entry costs a whole legend row and
    # says nothing the reader has not already been told two lines lower.
    if not ONE:
        handles.append(Rectangle((0, 0), 1, 1, fc="#B3261E"))
    labels = [s[0] for s in SERIES] + \
        ([] if ONE else ["base rule worse than doing nothing"])
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.3290 if ONE else 0.213), ncol=2 if ONE else 3, frameon=False,
               fontsize=F * (0.84 if ONE else 0.78), handlelength=1.8, handletextpad=0.55,
               columnspacing=2.2, labelspacing=0.35)

    # Four lines at full width, four at column width too, but each broken so no
    # line runs past the column. The finding is the point of the figure and is
    # not the thing to shrink.
    _n1 = fig.text(0.5, 0.2047 if not ONE else 0.2760,
                   "Shaded: the base rule is worse\nthan doing nothing." if ONE
                   else "Shaded: the base rule is worse than doing nothing. "
                        "On haze this begins at sigma 4,",
                   ha="center", va="center",
                   fontsize=F * (0.96 if ONE else 0.76), color=fs.RED,
                   linespacing=1.25)
    fig.text(0.5, 0.1689 if not ONE else 0.1948,
             "On haze from sigma 4, on rain\nat every sigma including zero."
             if ONE else
             "28.33 against 31.47; on rain at every sigma including zero, where "
             "40 per cent go to CLAHE.",
             ha="center", va="center", fontsize=F * (0.90 if ONE else 0.76),
             color=fs.RED, linespacing=1.25)
    fig.text(0.5, 0.1217 if not ONE else 0.1057,
             "At sigma 16 AdaIR overtakes DCP\non haze, 26.20 against 21.40."
             if ONE else
             "At sigma 16 the training-free advantage does not survive: on haze "
             "AdaIR overtakes DCP,",
             ha="center", va="center", fontsize=F * (0.90 if ONE else 0.76),
             color=fs.RED, linespacing=1.25)
    if not ONE:
        fig.text(0.5, 0.0858,
                 "26.20 against 21.40; on rain the gap is not significant. "
                 "We show the whole sigma grid.",
                 ha="center", va="center", fontsize=F * 0.76, color=fs.RED)
    _scope = fig.text(0.5, 0.0311 if not ONE else 0.0356,
             "haze and rain only, 500 crops per cell" if ONE else
             "Scope: haze and rain only, severities 3 to 5, 500 per cell. "
             "The other ten were not tested.",
             ha="center", va="center", fontsize=F * (0.80 if ONE else 0.72),
             color=fs.MUTED)

    ok = fs.run_gates(fig, args.outdir, "fig09_noise_robustness", line_axes=AXES)
    # Read back off the artists, not copied from the calls above; copied, these
    # have gone stale in three of these figures already.
    def _pt(obj):
        return round(obj.get_fontsize() / F, 4)
    fs.report_sizes(args.target, W, F, {
        "panel titles": _pt(AXES[0].title),
        "axis labels": _pt(AXES[0].yaxis.label),
        "tick labels": _pt(AXES[0].get_xticklabels()[0]),
        "the findings": _pt(_n1),
        "the scope line": _pt(_scope)})

    print("\nCAPTION:")
    for ln in [
        "Fig. 9. What additive noise does to each front end, and where the result stops",
        "holding. CURE-TSR renders haze and rain as noise-free veils; real cameras do",
        "not, and the dark channel prior divides by the transmission, which multiplies",
        "whatever noise is present. This pre-registered sensitivity test (Protocol Part",
        "12, registered before execution) adds zero-mean Gaussian noise of standard",
        "deviation sigma to the uint8 image before any operator runs. Three results.",
        "First, on haze the base rule falls BELOW no enhancement at sigma 4, 28.33",
        "against 31.47, and again at sigma 8, 24.20 against 31.33: under noise it is",
        "not merely weaker, it is harmful. The mechanism is in the same panel, since",
        "the rule routes 74 per cent of haze images to CLAHE at sigma 0 and 72 per cent",
        "at sigma 4, while CLAHE collapses from 49.47 to 6.00. Second, on rain the base",
        "rule is below no enhancement at EVERY sigma, including zero: 27.00 against",
        "28.67 here, and 34.11 against 35.31 on the full data set, so it is not an",
        "artefact of the 500-image subsample. At sigma 0 the rule sends 40 per cent of",
        "rain images to CLAHE, a minority but the largest enhanced share, and CLAHE",
        "reaches only 15.60 on rain against 28.67 for doing nothing. The selector V-B,",
        "which serves that branch with DCP, repairs it (35.33 at sigma 0) and stays",
        "above no enhancement at sigma 4 and 8 on both challenges. Third, at sigma 16",
        "the training-free advantage does not survive. On haze AdaIR overtakes DCP,",
        "26.20 against 21.40, and a paired exact McNemar test on the same 1500 images",
        "separates that from chance (p = 3e-06). On rain the two are level, 24.67",
        "against 23.27, a gap the same test cannot separate from chance (p = 0.26), so",
        "the reversal is claimed for haze alone. We report this rather than restrict",
        "the sigma grid to where the advantage survives.",
        "Scope: haze and rain only, the two veil-rendered challenges, at severities 3",
        "to 5, with 500 images per cell and seed 42. The other ten challenges were not",
        "tested under noise and nothing here transfers to them. The noise is additive",
        "and Gaussian; it is a sensitivity probe, not a camera model, and carries no",
        "Poisson component. It is a necessary, not a sufficient, transfer check, and it",
        "does not address the veil-rendering circularity, which is disclosed",
        "separately. Scope of the classifier: CompactCNN (145,291 parameters, 32x32",
        "input) on CURE-TSR.",
    ]:
        print("  " + ln)
    if not ok or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
