# -*- coding: utf-8 -*-
"""
FIG02_noise_injection_oracle.py
Paper figure 2. Script, output and paper number agree: FIG02 writes
fig02_noise_injection_oracle, and the article prints it as Fig. 2.

Figure 2: what the noise-injection oracle is, and why its height depends on the
kind of degradation rather than on the strength of the method.

WHAT THE FIGURE SAYS
Adding Gaussian noise restores nothing. It destroys. Yet on BLUR it hands the
classifier back 13 to 14 accuracy points, while on VEIL and ILLUMINATION it hands
back 1 to 2. So the bar an operator must clear before its gain can be credited to
restoration is not a property of the operator: it is a property of the
degradation. That is the whole reason the test exists, and it is the reason the
capability boundary of Fig. 3 looks the way it does.

  GaussianBlur sev 4   degraded 13.1 -> 27.4 at sigma 12   noise alone: +14.3
  LensBlur     sev 4   degraded 19.1 -> 32.1 at sigma 12   noise alone: +13.0
  Haze         sev 4   degraded 24.7 -> 26.6 at sigma  3   noise alone:  +1.9
  Darkening    sev 4   degraded 59.8 -> 60.8 at sigma  2   noise alone:  +1.0

All four panels are at severity 4. Fixing the severity across panels is
deliberate: choosing a different severity per panel would let the figure pick its
own evidence.

THE TWO ORACLES (Evaluation Protocol Part 17.9; never write "the oracle" alone)
  noise-injection oracle     the best accuracy reachable by adding Gaussian noise
  (this figure)              alone, with sigma chosen optimally on the test set.
                             An oracle because that choice needs the test labels,
                             which no deployed system has. An oracle over NOISE
                             LEVELS only, so a restoration operator can exceed it,
                             and on three of these four panels one does.
  operator-selection oracle  the best accuracy reachable if the best OPERATOR could
  (Section IV-E)             be picked per image. Not shown here.

WHY ONLY TWO OPERATORS PER PANEL
The best training-free operator and the best learned one. All six appear in Fig. 3;
repeating them here would crowd the panels and add nothing, and the best operator
is the strongest case that can be made FOR restoration: if even it fails to clear
the bar, as on Gaussian blur, the conclusion is not an artefact of a weak choice.

HOW "CLEARS THE BAR" IS DECIDED
Significance, not size. The green/grey split reads the verdict ZA recorded, the
same one that colours the cells of Fig. 3, so the two figures cannot disagree.
Comparing the two point estimates instead would let a gain of a point or two,
well inside what chance allows, count as restoration. That is not hypothetical:
the refined DCP lands 1.9 points above the line on Gaussian blur severity 4 with
an interval of [-1.0, +4.7] and p = 0.22, so its line is drawn above the dashed
one and coloured grey, and its panel legend carries n.s.

WORDING DISCIPLINE
The test bounds the BENEFIT that may be attributed to restoration. It does not
establish a mechanism. Neither the figure nor the caption may assert stochastic
resonance, high-frequency injection, or any other mechanism as fact.

READS   outputs_revision/Z_all12_per_image.csv     (Z_injection_definitive.py)
        outputs_revision/ZA_deep_vs_injection.json (ZA_deep_vs_injection.py)
WRITES  outputs_revision/figures/fig02_noise_injection_oracle.png   (400 dpi)
        outputs_revision/figures/fig02_noise_injection_oracle.svg
RUN     python FIG02_noise_injection_oracle.py                 # T-ITS full width
        python FIG02_noise_injection_oracle.py --target word   # plain portrait page
"""
import os
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fig_style as fs

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

NAMES = {1: "Decolorization", 2: "LensBlur", 3: "CodecError", 4: "Darkening",
         5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur", 8: "Noise", 9: "Rain",
         10: "Shadow", 11: "Snow", 12: "Haze"}
IDS = {v: k for k, v in NAMES.items()}

SEV = 4
SIGMAS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]      # sigma 0 is the column pred_raw
TRAINING_FREE = ["gamma", "clahe", "stretch", "dcp"]
# One definition of the learned family's stroke, used both where the lines are
# drawn and where the key is built, so the two cannot drift apart.
# Solid where the gain is significantly above the noise-injection oracle,
# dotted where it is not. Dashed is the oracle's own stroke and is not reused.
DOTTED = (0, (1.6, 1.5))
# All five, now that all five are scored on the same run and can be compared
# with the noise-injection oracle on the same footing.
LEARNED = ["adair", "cidnet", "zero_dce", "ffa_net", "promptir"]
PRETTY = {"gamma": "Gamma", "clahe": "CLAHE", "stretch": "Stretch", "dcp": "DCP",
          "adair": "AdaIR", "cidnet": "CIDNet",
          "zero_dce": "Zero-DCE", "ffa_net": "FFA-Net",
          "promptir": "PromptIR"}

PANELS = [("GaussianBlur", "blur"), ("LensBlur", "blur"),
          ("Haze", "veil"), ("Darkening", "illumination")]

YLIM = (-33, 50)      # accuracy gain over the degraded image, in points

# Locked 2026-07-11 from the authoritative per-image file. The figure aborts if the
# inputs disagree, and cross-checks every peak against what ZA recorded.
# best_ln is locked as well as the rest. It was not, and when the learned family
# grew from two members to five the panel that names the best of them changed
# from CIDNet to Zero-DCE with nothing to say so.
EXPECT = {
    "GaussianBlur": dict(raw=13.1, peak=14.3, sigma=12, clears=False,
                         best_tf="dcp", best_ln="adair"),
    "LensBlur":     dict(raw=19.1, peak=13.0, sigma=12, clears=True,
                         best_tf="dcp", best_ln="adair"),
    "Haze":         dict(raw=24.7, peak=1.9,  sigma=3,  clears=True,
                         best_tf="dcp", best_ln="adair"),
    "Darkening":    dict(raw=59.8, peak=1.0,  sigma=2,  clears=True,
                         best_tf="gamma", best_ln="zero_dce"),
}



def _seg_hits_box(bb, p, q):
    """Liang-Barsky: does the segment p->q cross the box bb?"""
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]),
                   (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


def label_lines_above(ax, ends, F):
    """Name the two operator lines above the panel, each in its own colour.

    Inside the panel there is nowhere to put them. A label at this size covers a
    third of the width and a tenth of the height, and the noise curve sweeps the
    full height, so on Gaussian blur every one of twenty measured positions along
    the AdaIR line was crossed by it. The room that does exist is vertical, so the
    labels go above the axes, where the colour ties each to its line and nothing
    can be hidden behind them.
    """
    out = []
    for k, (g, col, lbl, st) in enumerate(sorted(ends, key=lambda t: -t[0])):
        y = 1.022 + 0.118 * (1 - k)
        # A short sample of the line itself, in its own colour and style, so a
        # reader can tie the name to the stroke without counting from the top.
        ax.plot([0.0, 0.085], [y + 0.030] * 2, transform=ax.transAxes,
                color=col, lw=1.4, ls=st, clip_on=False, zorder=8)
        ax.text(0.108, y, lbl, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=F * 0.74, color=col)
        out.append(lbl)
    return " / ".join(out)


def place_legend(ax, handles, fontsize):
    """Put the legend where it covers no part of the noise curve.

    matplotlib's loc="best" minimises overlap, it does not eliminate it: on the
    Haze panel it still clipped one segment of the curve. This tries the standard
    locations in order and takes the first that misses the curve entirely, so the
    result is deterministic and reproduces on any machine.

    The test covers the CURVE, which is the measured trace and must stay legible,
    and not the horizontal reference lines. Those span the full panel by
    construction, so no legend inside the axes can miss them, and their values are
    printed in the legend itself. An earlier version claimed to test every line but
    transformed all of them through transData; an axhline carries its x in axes
    fractions, so that test only ever looked at a sliver near data x = 0 to 1 and
    silently passed. Using each line's own transform is correct for both kinds.
    """
    order = ["upper right", "upper left", "lower left", "lower right",
             "upper center", "lower center", "center left", "center right",
             "center"]
    fig = ax.figure
    for loc in order:
        leg = ax.legend(handles=handles, loc=loc, fontsize=fontsize, frameon=True,
                        framealpha=0.95, edgecolor="#DDDDDD", handlelength=1.3,
                        handletextpad=0.5, borderpad=0.4, labelspacing=0.3)
        leg.set_zorder(9)
        fig.canvas.draw()
        bb = leg.get_window_extent(renderer=fig.canvas.get_renderer())
        clash = 0
        for ln in ax.lines:
            if ln.get_gid() != "noise-curve":
                continue
            xy = ln.get_xydata()
            if len(xy) < 2:
                continue
            pts = ln.get_transform().transform(xy)
            clash += sum(_seg_hits_box(bb, p, q) for p, q in zip(pts[:-1], pts[1:]))
        if clash == 0:
            return loc
        leg.remove()
    leg = ax.legend(handles=handles, loc="best", fontsize=fontsize, frameon=True,
                    framealpha=0.95, edgecolor="#DDDDDD", handlelength=1.3,
                    handletextpad=0.5, borderpad=0.4, labelspacing=0.3)
    leg.set_zorder(9)
    print(f"  NOTE: no legend position on this panel misses the curve; "
          f"loc='best' was used.")
    return "best"


def load(csv_path, json_path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    za = json.load(open(json_path))["deep_vs_injection"]
    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    cols = ["pred_raw"] + [f"pred_n{s}" for s in SIGMAS]
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise SystemExit(f"ABORT: {csv_path} has no columns {missing}. This figure "
                         f"needs the per-sigma predictions written by "
                         f"Z_injection_definitive.py.")
    pred = {c: np.array([int(r[c]) for r in rows]) for c in cols}
    return ch, sev, tru, pred, za


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS),
                    default="ieee1col")
    ap.add_argument("--csv", default=str(OUT_DIR / "Z_all12_per_image.csv"))
    ap.add_argument("--za", default=str(OUT_DIR / "ZA_deep_vs_injection.json"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    for p in (args.csv, args.za):
        if not Path(p).exists():
            raise SystemExit(f"ABORT: {p} not found.")
    ch, sev, tru, pred, za = load(args.csv, args.za)

    def acc(col, mask):
        return 100.0 * float(np.mean(pred[col][mask] == tru[mask]))

    data, bad = {}, 0
    print("=== EXPECTED-OUTPUT AUDIT (against values locked 2026-07-11) ===")
    for name, _ in PANELS:
        m = (ch == IDS[name]) & (sev == SEV)
        if m.sum() == 0:
            raise SystemExit(f"ABORT: no rows for {name} severity {SEV}.")
        raw = acc("pred_raw", m)
        curve = [(0, 0.0)] + [(s, acc(f"pred_n{s}", m) - raw) for s in SIGMAS]
        pk_s, pk_g = max(curve[1:], key=lambda x: x[1])

        cell = za[f"{name}_sev{SEV}"]
        ops = {o: cell["ops"][o]["acc"] - raw
               for o in TRAINING_FREE + LEARNED}
        # Clearing the bar is a question of significance, not of size, and it is
        # settled by the same verdict that colours the cells of Fig. 3, so the two
        # figures read the same file and cannot disagree. Comparing the two point
        # estimates instead would let a gain of a point or two, well inside what
        # chance allows, count as restoration: on Gaussian blur the refined DCP
        # sits 1.9 points above the line with an interval of [-1.0, +4.7].
        sig = {o: cell["ops"][o]["verdict"].startswith("RESTORES")
               for o in TRAINING_FREE + LEARNED}
        ns = {o: cell["ops"][o]["verdict"] == "NOT above injection"
              for o in TRAINING_FREE + LEARNED}
        best_tf = max(TRAINING_FREE, key=lambda o: ops[o])
        best_ln = max(LEARNED, key=lambda o: ops[o])
        data[name] = dict(raw=raw, curve=curve, pk_s=pk_s, pk_g=pk_g, ops=ops,
                          sig=sig, ns=ns, best_tf=best_tf, best_ln=best_ln)

        e = EXPECT[name]
        # cross-check the peak against what ZA independently recorded
        za_gain = cell["oracle_acc"] - cell["raw"]
        for label, got, exp, tol in (("raw", raw, e["raw"], 0.1),
                                     ("peak gain", pk_g, e["peak"], 0.1),
                                     ("peak sigma", pk_s, e["sigma"], 0),
                                     ("ZA cross-check", pk_g, za_gain, 0.05)):
            if abs(got - exp) > tol:
                print(f"  MISMATCH {name} {label}: got {got}, expected {exp}")
                bad += 1
        clears = sig[best_tf]
        for _k, _got in (("best_tf", best_tf), ("best_ln", best_ln)):
            if e.get(_k) and _got != e[_k]:
                print(f"  MISMATCH {_k} {name}: got {_got}, expected {e[_k]}")
                bad += 1
        if clears != e["clears"]:
            print(f"  MISMATCH {name}: best training-free operator clears={clears}, "
                  f"expected {e['clears']}")
            bad += 1
        print(f"  {name:14s} degraded {raw:5.1f}  noise peak {pk_g:+5.1f} at sigma "
              f"{pk_s:2d}  best training-free {PRETTY[best_tf]:8s} {ops[best_tf]:+6.1f}"
              f"  best learned {PRETTY[best_ln]:7s} {ops[best_ln]:+6.1f}")
    print("  AUDIT PASSED: every value reproduces, and every peak agrees with ZA."
          if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure; send the output.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    F = F * 1.10 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    H = 6.10 if args.target == "ieee" else (5.72 if ONE else 5.55)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))
    AXES, LEGEND_LOC = [], {}
    if ONE:
        # A column leaves each panel about 1.4 inches of plotting area. A boxed
        # legend inside a panel that size covers between a half and six sevenths
        # of it, so the operator values are written at the right-hand end of
        # their own lines instead and the box is gone. The line is already
        # drawn across the panel; the label costs no width that was not there.
        # The band under the axes has to hold the tick labels and a two-line
        # axis label, then a four-line key, then two findings of two lines each.
        # That is 1.74 inches, which is what 0.313 of 5.55 comes to.
        gs = fig.add_gridspec(2, 2, left=0.168, right=0.988, top=0.868,
                              bottom=0.328, wspace=0.16, hspace=0.80)
    else:
        gs = fig.add_gridspec(2, 2, left=0.118, right=0.984, top=0.915,
                              bottom=0.245, wspace=0.20, hspace=0.42)

    TICK = F * 0.76
    from matplotlib.lines import Line2D
    for i, (name, kind) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        d = data[name]
        xs = [s for s, _ in d["curve"]]
        ys = [g for _, g in d["curve"]]
        hi, lo = YLIM

        ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=1)
        ax.plot(xs, ys, "-o", color="#6E6E6E", lw=1.6, ms=F * 0.26, mfc="white",
                mew=1.1, zorder=4, gid="noise-curve")

        # Thinner than it was. The training-free line on Gaussian blur sits 1.8
# points from this one, which at the scale of a one-column panel is the
# thickness of a single stroke: drawn at the old widths the two fused, and
# the grey solid line, which occurs nowhere else in the figure, could not be
# found at all.
        ax.axhline(d["pk_g"], color=fs.RED,
                   lw=1.2 if ONE else 1.5, ls="--", zorder=5)
        ax.plot([d["pk_s"]], [d["pk_g"]], marker="*", ms=F * 1.05, color=fs.RED,
                mec="white", mew=0.6, zorder=6)
        # Only the VALUE goes on the plot, three points clear of its own dashed line
        # and far from the curve, which by sigma 32 has fallen well below it. What
        # the dashed line MEANS is stated once, in the key under the figure.


        # At full width the two operator values sit in a boxed legend. In one
        # column that box swallows the panel, so each line is labelled at its
        # own right-hand end instead. Which line is which is then given by the
        # words rather than by the colour, and the colour stays free to mean
        # what it has always meant here: whether the line clears the bar.
        handles = []
        ends = []
        for op in (d["best_tf"], d["best_ln"]):
            g = d["ops"][op]
            # Colour is the family, green for training-free and purple for
            # learned, which is what Fig. 1 already uses. It has to be the
            # family rather than the test result, because the pairs that come
            # closest together are always one of each family: on the
            # illumination change the two are 0.9 points apart and on lens blur
            # the learned line is 1.1 points under the oracle. Two greys or two
            # greens that close merge into one stroke whatever their style,
            # while a green beside a purple stays two lines.
            col = fs.GREEN if op in TRAINING_FREE else fs.PURPLE
            # A line can sit above the dashed one and still be grey, which is the
            # whole point of the test; n.s. marks that its interval spans the line.
            tag = " n.s." if d["ns"][op] else ""
            # Colour says whether the line clears the oracle, so it cannot also
            # say which family the line belongs to. Style does that. It matters
            # because two of these lines come within a point of each other, and
            # a third comes within a point of the dashed oracle: drawn in one
            # style they merge into a single stroke and the panel appears to
            # show one operator where it shows two.
            st = "-" if d["sig"][op] else DOTTED
            # No white outline. It was added when both families were drawn in
            # the same colour, to keep two adjacent lines apart. With the
            # families in different colours it does the opposite: on the
            # illumination change the purple line sits 0.9 points above the
            # green one, and a halo wide enough to be seen wiped the green out.
            _lw = 1.2 if ONE else (1.7 if op in TRAINING_FREE else 1.5)
            ax.axhline(g, color=col, lw=_lw, ls=st,
                       zorder=6 if op in TRAINING_FREE else 7)
            lbl = f"{PRETTY[op]} {g:+.1f}{tag}"
            handles.append(Line2D([], [], color=col, lw=2.2, ls=st, label=lbl))
            ends.append((g, col, lbl, st))
        ax.set_xlim(-0.8, 33)
        ax.set_ylim(*YLIM)
        # The labels are placed after the limits are fixed. Placed before them
        # the transform is still the autoscaled one, so every position measured
        # against the curve is measured in the wrong coordinates and the placer
        # reports that nothing fits.
        # The oracle's value used to sit at a fixed offset above the dashed
        # line, which put it across the training-free line on lens blur. It is
        # placed by measurement now, and against everything drawn in the panel:
        # the operator lines, the dashed line itself, and the noise curve. A
        # first attempt checked only the operator lines and moved the label
        # straight onto the curve instead.
        ax.set_xlim(-0.8, 33)
        ax.set_ylim(*YLIM)
        fig.canvas.draw()
        _rend = fig.canvas.get_renderer()
        _cur = [ax.transData.transform(q) for q in d["curve"]]
        _ok = False
        for _dy, _va in ((3.2, "bottom"), (-3.2, "top"), (8.4, "bottom"),
                         (-8.4, "top"), (13.6, "bottom"), (-13.6, "top"),
                         (19.0, "bottom"), (-19.0, "top")):
            _t = ax.text(32.4, d["pk_g"] + _dy, f"{d['pk_g']:+.1f}",
                         ha="right", va=_va, fontsize=F * 0.78, color=fs.RED,
                         fontweight="bold", zorder=7)
            fig.canvas.draw()
            _b = _t.get_window_extent(renderer=_rend)
            _pad = matplotlib.transforms.Bbox(
                [[_b.x0 - 2, _b.y0 - 3], [_b.x1 + 2, _b.y1 + 3]])
            _hitc = any(_seg_hits_box(_pad, _cur[m], _cur[m + 1])
                        for m in range(len(_cur) - 1))
            _ys = [ax.transData.transform((0.0, v))[1]
                   for v in [d["pk_g"]] + [g for g, *_ in ends]]
            _hitl = any(_pad.y0 - 2 <= v <= _pad.y1 + 2 for v in _ys)
            _ab = ax.get_window_extent(renderer=_rend)
            _in = _pad.y0 >= _ab.y0 and _pad.y1 <= _ab.y1
            if not _hitc and not _hitl and _in:
                _ok = True
                break
            _t.remove()
        if not _ok:
            raise SystemExit(f"ABORT: nowhere clear for the oracle value on "
                             f"{name}; it would sit on a line or the curve.")

        if ONE:
            LEGEND_LOC[name] = label_lines_above(ax, ends, F)
        else:
            LEGEND_LOC[name] = place_legend(ax, handles, F * 0.80)

        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_yticks([-30, -15, 0, 15, 30, 45])
        ax.tick_params(labelsize=TICK, length=2.5, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#999999")
        # Two lines. On one line "Darkening (illumination), degraded 59.8%" is
        # 3.35 in wide at this size and the panel is only 3.1 in, so it overflowed
        # the canvas. Height is free; width is not.
        # The internal keys run the words together. These are the same words
        # written out; nothing is shortened, and the kind of degradation moves
        # to the second line where there is room for it.
        _nm = {"GaussianBlur": "Gaussian blur", "LensBlur": "Lens blur"}.get(
            name, name)
        # In one column the kind of degradation does not fit beside the name,
        # and the caption is where it belongs anyway: the four panels are two
        # blurs, a veil and an illumination change, which is a statement about
        # the set rather than about any one panel.
        ax.set_title(f"{_nm}\ndegraded {d['raw']:.1f}%" if ONE
                     else f"{name} ({kind})\ndegraded {d['raw']:.1f}%",
                     fontsize=F * 0.86,
                     pad=(F * 2.05 if ONE else 4), linespacing=1.2)
        if i % 2 == 0:
            # In one column the label is written once for the whole left side
            # rather than once per row. Three short lines beside each panel
            # need more width than the margin has; two long lines beside both
            # panels need less, and they have the height of both rows to run in.
            if not ONE:
                ax.set_ylabel("accuracy gain over the\ndegraded image (points)",
                              fontsize=F * 0.76, labelpad=2, linespacing=1.2)
        if i // 2 == 1:
            ax.set_xlabel("sigma of the injected\nGaussian noise" if ONE
                          else "sigma of the injected Gaussian noise",
                          fontsize=F * 0.76, labelpad=2, linespacing=1.2)
        AXES.append(ax)

    if ONE:
        fig.text(0.012, 0.5 * (0.868 + 0.328),
                 "accuracy gain over the\ndegraded image (points)",
                 rotation=90, ha="left", va="center",
                 fontsize=F * 0.76, linespacing=1.2)

    # The key shows what the panels actually contain, and it is built from
    # them. It used to carry a black solid line and a black dash-dot line as
    # abstract samples of the two families, and neither appears anywhere in the
    # figure: the operator lines are green or grey, never black. A key with
    # swatches a reader cannot find is worse than a shorter one.
    seen = set()
    for _n, _k in PANELS:
        _d = data[_n]
        for _op in (_d["best_tf"], _d["best_ln"]):
            seen.add((_op in TRAINING_FREE, bool(_d["sig"][_op])))
    KEY_ONE, KEY_ONE_LAB = [
        Line2D([], [], color="#6E6E6E", lw=1.6, marker="o", ms=F * 0.26,
               mfc="white", mew=1.1),
        Line2D([], [], color=fs.RED, lw=1.5, ls="--", marker="*",
               ms=F * 0.95, mec="white", mew=0.5),
    ], [
        "accuracy after injecting noise of level sigma",
        "noise-injection oracle: the peak of that curve",
    ]
    for _tf in (True, False):
        for _sig in (True, False):
            if (_tf, _sig) not in seen:
                continue
            KEY_ONE.append(Line2D(
                [], [], color=fs.GREEN if _tf else fs.PURPLE,
                lw=1.4, ls="-" if _sig else DOTTED))
            KEY_ONE_LAB.append(
                ("green, a training-free operator: " if _tf
                 else "purple, a learned restorer: ")
                + ("above the oracle" if _sig else "not above it"))
    key = [Line2D([], [], color="#6E6E6E", lw=1.6, marker="o", ms=F * 0.26,
                  mfc="white", mew=1.1),
           Line2D([], [], color=fs.RED, lw=1.5, ls="--", marker="*",
                  ms=F * 0.95, mec="white", mew=0.5),
           Line2D([], [], color="#14532D", lw=2.2),
           Line2D([], [], color="#8A8A8A", lw=2.2)]
    # One column, one entry per line. Two columns of this wording measured wider
    # than the canvas on the previous figure; stacking them costs a quarter of an
    # inch of height, and the height is there.
    fig.legend(KEY_ONE if ONE else key,
               KEY_ONE_LAB if ONE else
               ["accuracy after injecting noise of level sigma",
                "noise-injection oracle: the peak of that curve",
                "significantly above the noise-injection oracle",
                "not significantly above it"],
               loc="lower center",
               bbox_to_anchor=(0.5, 0.066 if ONE else 0.098),
               ncol=1 if ONE else 2,
               frameon=False, fontsize=F * 0.78, handlelength=1.8,
               handletextpad=0.55, columnspacing=2.4, labelspacing=0.32)

    # Both lines are built from the peaks the audit just checked, so a change in
    # the data moves the words with it. They used to be typed.
    _blur = [data[n]["pk_g"] for n, k in PANELS if k == "blur"]
    _rest = [data[n]["pk_g"] for n, k in PANELS if k != "blur"]
    # One finding in the figure, as in Fig. 1. The supporting numbers, which
    # this line used to carry, are in the caption; a column has room for the
    # conclusion or for the evidence, and the conclusion is what a reader takes
    # from a glance.
    if not ONE:
        fig.text(0.5, 0.068,
                 f"Injected noise restores nothing: {min(_blur):.0f} to "
                 f"{max(_blur):.0f} points on blur, {min(_rest):.0f} to "
                 f"{max(_rest):.0f} on veil and illumination.",
                 ha="center", va="center", fontsize=F * 0.78, color=fs.INK,
                 linespacing=1.25)
    fig.text(0.5, 0.030 if ONE else 0.028,
             "The bar is set by the degradation, not the operator.\n"
             "On Gaussian blur nothing clears it." if ONE else
             "The bar is set by the degradation, not the operator. "
             "On Gaussian blur nothing clears it.",
             ha="center", va="center", fontsize=F * 0.78, color=fs.RED,
             linespacing=1.25)

    print("\n  legend placement (first position that misses the curve):")
    for k, v in LEGEND_LOC.items():
        print(f"    {k:14s} {v}")
    ok = fs.run_gates(fig, args.outdir, "fig02_noise_injection_oracle",
                      line_axes=AXES)
    fs.report_sizes(args.target, W, F, {
        "panel titles": 0.86, "panel legends": 0.80, "the key": 0.78,
        "oracle value": 0.78, "axis labels": 0.76, "tick labels": 0.76,
        "the two notes": 0.78})

    print("\nCAPTION:")
    for ln in [
        "Fig. 2. What the noise-injection oracle is, and why its height is a property",
        "of the degradation rather than of the operator. Each panel adds Gaussian",
        "noise of standard deviation sigma to the degraded image and plots the",
        "resulting change in classification accuracy; all four are at severity 4, and",
        "fixing the severity across panels keeps the figure from selecting its own",
        "evidence. Adding noise restores nothing, yet on the two blur challenges it",
        "returns 13 to 14 accuracy points, and on veil and illumination only 1 to 2.",
        "The dashed line marks the peak of that curve, the noise-injection oracle: an",
        "oracle because sigma is chosen with the test labels, which no deployed system",
        "has, and an oracle over noise levels only, so a restoration operator may",
        "exceed it. An operator must clear it before its gain can be attributed to",
        "restoration, because a manipulation that restores nothing already reaches",
        "that accuracy. Solid lines give the best training-free and the best learned",
        "operator on each panel (all six appear in Fig. 3); green where the operator",
        "clears the bar significantly and grey where it does not, on the same verdict",
        "that colours the cells of Fig. 3, with n.s. marking a margin whose interval",
        "spans the line. On Gaussian blur nothing clears it: the dark-channel prior",
        "lands 1.9 points above the line, but its interval runs from -1.0 to +4.7, so",
        "the margin is inside what chance allows and cannot be credited to",
        "restoration. On lens blur, the same degradation family, DCP clears it while",
        "AdaIR, which was trained to deblur, does not. This test bounds the benefit",
        "that may be attributed to restoration; it does not establish a mechanism.",
        "Scope: CompactCNN (145,291 parameters, 32x32 input) on CURE-TSR.",
    ]:
        print("  " + ln)
    if not ok or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
