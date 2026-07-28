# -*- coding: utf-8 -*-
"""
FIG03_capability_boundary.py   (v2: transposed, sized for a Word landscape page)

WHY THE LAYOUT CHANGED
v1 put twelve challenges times four severities across the x axis, which forced 48
columns and squeezed every label down to 6 or 7 point; the red annotations then
collided with the bars. Transposing fixes it at the root: challenges become ROWS
(their names read horizontally, no rotation), and operators times severities
become 24 COLUMNS. Same information, half the columns, roughly double the type
size, and no overlapping text anywhere.

WHAT IT SHOWS
  Left bars     How high the bar is. The accuracy gain that OPTIMALLY DOSED NOISE
                alone achieves over the degraded image (the noise-injection oracle,
                averaged over severities 2 to 5). An operator must clear this
                before its gain can be attributed to restoration.
  Heatmap       Who clears it. Six operators (four training-free, two learned)
                times severities 2 to 5. A cell is green only where the operator
                is significantly above the noise-injection oracle under Holm correction
                over the full family of 432 tests. Intensity is the margin.
  Right column  The boundary: restorable severities out of four.

THE TWO KINDS OF UNRESTORABLE, AND THE TWO STRENGTHS OF THE CLAIM
  Rows are ordered by restorability, and the four zeros are ordered by the height
  of their bar, so the two kinds sit together and are bracketed on the right:
    Decolorization, GaussianBlur   high bar. Operators DO gain here, but noise
                                   gains as much, so the gain cannot be credited
                                   to restoration. Several cells here are
                                   nominally significant but do not survive the
                                   432-family Holm correction, so they carry dots
                                   and are reported as zero, conservatively.
    CodecError, Shadow             low bar (the oracle gains under 3 points). No
                                   operator clears it after the 432-family
                                   correction. CodecError is the stronger case: not
                                   one of its cells reaches even nominal
                                   significance, so its zero is unconditional. On
                                   Shadow the refined DCP does clear the oracle at
                                   severity 5 nominally (+1.9, p = 1.4e-02), but
                                   that one cell does not survive the correction and
                                   is shown as a dot.

SEVERITY 1 IS EXCLUDED BY PRE-REGISTRATION
  A severity is judged only where a qualifying contrast condition exists in the
  frozen cache (Protocol Part 16). Severity 1 has none, so it carries no verdict.

HEIGHT IS FREE, BUT SLACK IS NOT THE SAME AS SPACE
  Height costs nothing in apparent type size, which tempted an earlier draft to grow
  to 7.5 in so a four-row legend would fit. 1.20 in of that (16 per cent of the
  figure) then sat there as three empty bands: 0.37 in at the top, 0.32 in between
  the colourbar label and the legend, and 0.51 in between the legend and the note.
  The cause was arithmetic on the legend's BOUNDING BOX rather than on its INK: a
  matplotlib legend carries about 0.13 in of internal padding below its last row, so
  anchoring the next element relative to the bbox leaves that padding plus whatever
  gap was asked for. Placing everything from ink edge to ink edge closed the gaps and
  brought the figure down to 6.5 in. check_whitespace() now fails any figure with a
  blank band over 0.25 in, so this cannot recur silently.

TERMINOLOGY: THERE ARE TWO ORACLES IN THIS PAPER, AND THEY MUST NEVER SHARE A NAME
  "Oracle" is standard usage in computer vision for an idealised choice made with
  ground truth that a deployed system cannot access, giving an unrealisable upper
  bound. This paper uses that device TWICE, over two different things, so each one
  names what it is an oracle OVER:

    noise-injection oracle       the best accuracy reachable by adding Gaussian
    (this figure)                noise alone, with sigma chosen optimally on the
                                 test set. It is a BAR: an operator must clear it
                                 before its gain can be credited to restoration.
                                 It is an oracle over noise levels only, which is
                                 why an operator can and does exceed it (DCP is
                                 +45 on Haze). Nothing paradoxical about that.

    operator-selection oracle    the best accuracy reachable if the best OPERATOR
    (Section IV-E, o4 to o7)     could be picked per image using ground truth. It
                                 is a CEILING: it measures the headroom left in
                                 operator selection.

  Earlier drafts called the first one the "injection oracle". That name was built
  by truncating the standard phrase "noise injection" and did not say what the
  oracle ranged over, so "DCP is 45 points above the oracle" read as a paradox.
  The bare phrase "the oracle" must not appear anywhere the two could be confused.

WHAT THE MULTIPLICITY CORRECTION IS, AND WHY THE FIGURE MENTIONS IT
  432 tests are run: 12 challenges by 4 severities by 9 operators. At a per-test
  threshold of 0.05, chance alone would hand back about 14 "significant" results
  even if no operator restored anything, so a raw count of significant cells is
  meaningless. The Holm-Bonferroni procedure fixes that: sort all 432 p values
  ascending, require the smallest to beat 0.05/432, the next to beat 0.05/431, and
  so on, stopping at the first failure. It bounds the probability of even ONE false
  positive across the whole family at 5 per cent.

  On this data: 344 tests are nominally significant (p < 0.05); 252 survive Holm.
  The 50 that do not are exactly why the figure carries dots. The first rejection
  is at p = 4.90e-04 against a threshold of 4.72e-04, so the boundary really is a
  boundary and not a cliff.

  The figure says "corrected for 432 tests" rather than "Holm", because a reader
  who does not already know the procedure learns nothing from its name. The name,
  the family size, and the survivor count all go in the caption.

WORDING DISCIPLINE
  The test bounds the BENEFIT that may be attributed to restoration. It does not
  establish a mechanism. The caption must not assert stochastic resonance or any
  other mechanism as fact.

READS   outputs_revision/ZA_deep_vs_injection.json
WRITES  outputs_revision/figures/fig03_capability_boundary.png   (400 dpi)
        outputs_revision/figures/fig03_capability_boundary.svg
RUN     python FIG03_capability_boundary.py                 # Word landscape, large
        python FIG03_capability_boundary.py --target ieee   # T-ITS double column
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

CHALLENGES = ["Haze", "Darkening", "Rain", "LensBlur", "Noise", "Snow",
              "DirtyLens", "Exposure", "Decolorization", "GaussianBlur",
              "CodecError", "Shadow"]
SEVS = [2, 3, 4, 5]
# Nine, not six. The three learned restorers that used to be missing were
# scored in a separate pass whose rows could not be matched to these; they are
# in the merged file now, so they enter the boundary on the same footing and
# the Holm family grows from 288 tests to 432 with them.
OPS = ["gamma", "clahe", "stretch", "dcp", "adair", "cidnet",
       "zero_dce", "ffa_net", "promptir"]
# Parameter counts (AdaIR 28.8M, CIDNet 2.0M) live in the caption. Putting them
# under the names made the labels two lines tall and wide enough to collide with
# the "restorable" header, which capped the type size.
OP_LABEL = {"gamma": "Gamma", "clahe": "CLAHE", "stretch": "Stretch",
            "dcp": "DCP", "adair": "AdaIR", "cidnet": "CIDNet",
            "zero_dce": "Zero-DCE", "ffa_net": "FFA-Net",
            "promptir": "PromptIR"}
N_TRAINING_FREE = 4
ALPHA = 0.05

# These two are constrained by the COLUMN WIDTH, not by the base font: six operator
# names must sit side by side over 24 columns, and a two-digit value must sit inside
# one column with a visible gap to its neighbour. They therefore do not scale with F
# the way the row labels do. Raising them is what produced "GammaCLAHEStretch" and
# "2529"; the layout check now enforces a minimum gap so this cannot recur silently.
# Nine operator names have to fit where six did. The row-label column gives
# back what it does not need, and the names come down from ten point to seven;
# 'Zero-DCE' is the longest and needs 0.49 of the 0.52 inch each operator now
# Thirty-six columns where there were twenty-four, so each cell is 0.13 inch
# across and a two-digit margin has to fit inside it.
# The operator name is the widest thing in a cell; at column width a cell is
# half an inch across and 'Zero-DCE' has to fit inside it.
CELL_MULT = 0.66

EXPECT_N_TESTS = 432
EXPECT_N_SURVIVE = 252
# Every challenge is listed, the four that clear nothing included, so that a
# challenge moving off zero is caught rather than passing unnoticed.
EXPECT_RESTORABLE = {"Haze": 4, "Darkening": 3, "Rain": 3, "LensBlur": 2,
                     "Snow": 2, "Noise": 1, "DirtyLens": 1, "Exposure": 1,
                     "Decolorization": 0, "GaussianBlur": 0, "CodecError": 0,
                     "Shadow": 0}

CMAP = LinearSegmentedColormap.from_list(
    "restore", ["#E8F4EA", "#A8D5B5", "#5FAF7D", "#2E7D4F", "#14532D"])
GREY_NOT_ABOVE = "#EAEAEA"
GREY_WORSE = "#C2C2C2"
# A pale wash for the cell and a strong ink for its text, one pair per family.
# Two ramps, one per family: the hue says which family clears the cell and the
# depth says by how much. A single bar cannot serve two hues, so the caption
# gives the range instead and the cell carries the count rather than the margin.
INK_FREE, INK_LEARN = "#1F6B44", "#5B3492"
RAMP_FREE = LinearSegmentedColormap.from_list(
    "free", ["#EAF5EE", "#9CCFB2"])
RAMP_LEARN = LinearSegmentedColormap.from_list(
    "learn", ["#F2ECFA", "#BFA3E0"])
RED = "#B3261E"
MUTED = "#555555"

# THE ONE RULE THAT GOVERNS APPARENT TYPE SIZE
# Word scales an image to the available width. What the reader actually sees is
#     apparent_pt = font_pt * (display_width / figure_width)
# So making the figure PHYSICALLY BIGGER makes the type LOOK SMALLER, because Word
# then shrinks it harder. Earlier versions of this script got that backwards: a
# 10.4 inch figure at 11.5 pt, dropped into a 7.16 inch column, renders at 7.9 pt.
# The fix is to build the figure AT the width it will be displayed at, so the scale
# factor is 1.0 and the point sizes below are exactly what the reader sees.
TARGETS = {                       # width, height (inches), base font (pt)
    # T-ITS full-width figure (spans both columns). Insert at 100 per cent; the
    # template's text block is 7.16 inches, so nothing is rescaled.
    "ieee": (7.16, 6.22, 12.75),
    # One T-ITS column. The grid holds twelve rows of four cells and each cell
    # names a front end, so it is tight; drawn at 3.48 it is legible, drawn at
    # 7.16 and dropped into a column it is shown at 49 per cent and is not.
    "ieee1col": (3.48, 5.85, 10.60),
    # Portrait page with 1 inch margins (6.5 inch text block), e.g. a plain Word
    # manuscript rather than the IEEE template.
    "word": (6.50, 5.90, 11.55),
    # Only for a landscape page or a poster, where 10.4 inches really is displayed
    # at 10.4 inches. Do NOT drop this one into a portrait column.
    "large": (10.4, 6.9, 11.5),
}
DISPLAY_WIDTH = {"ieee": 7.16, "ieee1col": 3.48, "word": 6.50, "large": 10.4}


def holm(cells):
    tests = [(k, op, o["p"]) for k, d in cells.items() if d["validated"]
             for op, o in d["ops"].items()]
    tests.sort(key=lambda x: x[2])
    m = len(tests)
    surv = set()
    for i, (k, op, p) in enumerate(tests):
        if p < ALPHA / (m - i):
            surv.add((k, op))
        else:
            break
    return surv, m



def _check_clipping(png, border=6):
    """No ink may sit within `border` pixels of the canvas edge."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed; clipping check skipped.)")
        return True
    arr = np.array(Image.open(png).convert("L"))
    ink = arr < 245
    touch = {"top": int(ink[:border, :].sum()), "bottom": int(ink[-border:, :].sum()),
             "left": int(ink[:, :border].sum()), "right": int(ink[:, -border:].sum())}
    if any(touch.values()):
        print("  LAYOUT WARNING: content is clipped at the canvas edge:")
        for k, v in touch.items():
            if v:
                print(f"    {k}: {v} ink pixels within {border} px of the edge")
        return False
    return True


def _check_text_over_bars(ax, min_frac=0.10):
    """Bar value labels must sit OUTSIDE their bars, not on top of them."""
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    bars = [p.get_window_extent(renderer=r) for p in ax.patches]
    bad = []
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        tb = t.get_window_extent(renderer=r)
        for bb in bars:
            ov = (max(0.0, min(tb.x1, bb.x1) - max(tb.x0, bb.x0))
                  * max(0.0, min(tb.y1, bb.y1) - max(tb.y0, bb.y0)))
            if ov / max(tb.width * tb.height, 1.0) > min_frac:
                bad.append((t.get_text(), 100 * ov / max(tb.width * tb.height, 1.0)))
                break
    if bad:
        print(f"  LAYOUT WARNING: {len(bad)} bar label(s) drawn on top of their bar:")
        for txt, pct in bad[:6]:
            print(f"    '{txt}' is {pct:.0f}% covered by the bar")
        return False
    return True


def _check_whitespace(png, max_band_in=0.25, dpi=400):
    """No blank horizontal band may exceed max_band_in inches.

    Height is free, which is true, but that is not a licence to leave the slack
    lying around as gaps. An earlier draft grew to 7.5 in so the four-row legend
    would fit, and 1.20 in of that (16 per cent of the figure) ended up as three
    empty bands rather than as content: 0.37 in at the top, 0.32 in between the
    colourbar label and the legend, and 0.51 in between the legend and the note.
    """
    try:
        from PIL import Image
    except ImportError:
        return True
    ink = np.array(Image.open(png).convert("L")) < 245
    H, W = ink.shape
    dens = ink.sum(axis=1) / W
    i, bad = 0, []
    while i < H:
        empty = dens[i] < 0.001
        j = i
        while j < H and (dens[j] < 0.001) == empty:
            j += 1
        if empty and (j - i) / dpi > max_band_in:
            bad.append(((j - i) / dpi, i, j))
        i = j
    if bad:
        print(f"  LAYOUT WARNING: {len(bad)} blank band(s) taller than "
              f"{max_band_in} in:")
        for h, a, b in sorted(bad, reverse=True)[:5]:
            print(f"    {h:.2f} in of white between y = {a} and y = {b} px")
        print(f"    total wasted: {sum(h for h, _, _ in bad):.2f} in of "
              f"{H/dpi:.2f} in")
        return False
    return True


def _check_overlaps(fig, pad_px=2.0):
    """No two text objects may overlap OR come within pad_px of each other.

    Checking only for OVERLAP is not enough. Labels that merely abut, with zero
    overlap, read as one word: an earlier version rendered the operator row as
    "GammaCLAHEStretch" and the cell values as "2529" and "1124", and the overlap
    test passed all of them. So every box is inflated by pad_px on each side first,
    and any intersection of the inflated boxes is a failure.

    The figure must be drawn first: before a draw, every text sits at its
    untransformed origin and the boxes all collide spuriously. Call this BEFORE
    savefig, because saving to SVG swaps the canvas and leaves the Agg renderer
    stale.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        items += [t for t in ax.texts if t.get_text().strip() and t.get_visible()]
        if ax.axison:            # an axes with axis("off") still holds tick label
            for t in ax.get_xticklabels() + ax.get_yticklabels():   # objects, which
                if t.get_text().strip() and t.get_visible():        # are invisible;
                    items.append(t)                                 # counting them
        if ax.title.get_text().strip():   # v2 forgot the title, and a title once
            items.append(ax.title)        # collided with an operator label
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            if lbl.get_text().strip() and lbl.get_visible():
                items.append(lbl)
    items += [t for t in fig.texts if t.get_text().strip()]
    for leg in fig.legends:
        items += list(leg.get_texts())
    boxes = []
    for t in items:
        try:
            boxes.append((t.get_text().replace("\n", " / ")[:40],
                          t.get_window_extent(renderer=r)))
        except Exception:
            pass
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ti, bi), (tj, bj) = boxes[i], boxes[j]
            gx = max(bj.x0 - bi.x1, bi.x0 - bj.x1)   # horizontal gap, negative if
            gy = max(bj.y0 - bi.y1, bi.y0 - bj.y1)   # they span each other
            if gx < pad_px and gy < pad_px:
                bad.append((ti, tj, max(gx, gy)))
    if bad:
        print(f"  LAYOUT WARNING: {len(bad)} text pair(s) closer than {pad_px} px "
              f"(overlapping or abutting), among {len(boxes)} text objects:")
        for a, b, g in sorted(bad, key=lambda x: x[2])[:8]:
            how = f"overlap by {-g:.1f} px" if g < 0 else f"only {g:.1f} px apart"
            print(f"    '{a}' and '{b}': {how}")
        return False
    print(f"  ({len(boxes)} text objects checked pairwise; every pair is at least "
          f"{pad_px} px apart.)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), default="ieee1col")
    ap.add_argument("--src", default=str(OUT_DIR / "ZA_deep_vs_injection.json"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"ABORT: {src} not found. Run ZA_deep_vs_injection.py first.")
    cells = json.load(open(src))["deep_vs_injection"]

    surv, m = holm(cells)
    print(f"[holm] family size {m}, surviving {len(surv)}")

    # One cell per combination of challenge and severity, not one per operator
    # within it. The article's claim is about combinations: how many of the
    # forty-eight admit restoration at all. Giving every operator its own column
    # made a grid of 432 cells of which 49 were green, and squeezed nine names
    # and their margins into a tenth of an inch each. Collapsed, a cell has
    # room to say which operator clears it and by how much, which is what a
    # reader takes away, and the operators that clear nothing take no space.
    n_r, n_s = len(CHALLENGES), len(SEVS)

    margin = np.full((n_r, n_s), np.nan)   # margin of the best clearer
    state = np.zeros((n_r, n_s), dtype=int)   # 0 none clears, 2 cleared
    who = {}                                  # (ri, si) -> (op, margin, n_more)
    bar = np.zeros(n_r)
    dots = []
    n_green_tests = 0

    for ri, c in enumerate(CHALLENGES):
        gains = []
        for si, s in enumerate(SEVS):
            key = f"{c}_sev{s}"
            d = cells[key]
            if not d["validated"]:
                raise SystemExit(f"ABORT: {key} carries no verdict.")
            gains.append(d["oracle_acc"] - d["raw"])
            clears = []
            near = []
            for op in OPS:
                o = d["ops"][op]
                rest = o["verdict"].startswith("RESTORES")
                if rest and (key, op) in surv:
                    clears.append((o["vs_oracle"], op))
                    n_green_tests += 1
                elif rest:
                    near.append((op, o["vs_oracle"], o["p"]))
            if clears:
                clears.sort(reverse=True)
                margin[ri, si] = clears[0][0]
                state[ri, si] = 2
                who[(ri, si)] = (clears[0][1], clears[0][0], len(clears) - 1)
            elif near:
                # Nothing survives here, but something was nominally
                # significant: the cell is marked so the reader can see the
                # boundary is decided by the correction and not by chance.
                dots.append((ri, si, c, s, near[0][0], near[0][1], near[0][2]))
        bar[ri] = float(np.mean(gains))

    restorable = {c: int(sum(state[ri, si] == 2 for si in range(n_s)))
                  for ri, c in enumerate(CHALLENGES)}

    # ---------------- audits ----------------
    print("\n=== EXPECTED-OUTPUT AUDIT ===")
    bad = 0
    if m != EXPECT_N_TESTS:
        print(f"  MISMATCH family size: {m} vs {EXPECT_N_TESTS}")
        bad += 1
    if len(surv) != EXPECT_N_SURVIVE:
        print(f"  MISMATCH survivors: {len(surv)} vs {EXPECT_N_SURVIVE}")
        bad += 1
    for c in CHALLENGES:
        if restorable[c] != EXPECT_RESTORABLE[c]:
            print(f"  MISMATCH {c}: {restorable[c]}/4 vs {EXPECT_RESTORABLE[c]}/4")
            bad += 1
    print("  AUDIT PASSED: the boundary reproduces exactly." if bad == 0
          else f"  AUDIT: {bad} mismatch(es). Do NOT use this figure; send the output.")

    perch = set()
    for c in CHALLENGES:
        sub = sorted([((c, s, op), cells[f"{c}_sev{s}"]["ops"][op]["p"])
                      for s in SEVS for op in OPS], key=lambda x: x[1])
        for i, (key, p) in enumerate(sub):
            if p < ALPHA / (len(sub) - i):
                perch.add(key)
            else:
                break

    def rest_(c, s, op):
        return cells[f"{c}_sev{s}"]["ops"][op]["verdict"].startswith("RESTORES")

    per = {c: sum(any(rest_(c, s, op) and (c, s, op) in perch for op in OPS)
                  for s in SEVS) for c in CHALLENGES}
    nom = {c: sum(any(rest_(c, s, op)
                      and cells[f"{c}_sev{s}"]["ops"][op]["p"] < ALPHA
                      for op in OPS) for s in SEVS) for c in CHALLENGES}
    print("\n=== CORRECTION-FAMILY SENSITIVITY (disclose this; never hide it) ===")
    print(f"  {'challenge':16s}{'432-family':>12s}{'per-challenge':>15s}{'nominal':>10s}")
    for c in CHALLENGES:
        tag = "robust" if restorable[c] == per[c] == nom[c] else "SHIFTS"
        print(f"  {c:16s}{restorable[c]:>9d}/4{per[c]:>12d}/4{nom[c]:>7d}/4   {tag}")
    print("  We report the 432-family column, the most conservative of the three.")
    print("  CodecError and Shadow are UNCONDITIONAL zeros: no nominally significant")
    print("  restoration in any of their 48 tests. Decolorization and GaussianBlur")
    print("  each rest on one cell not surviving the family correction (dotted).")

    # ---------------- figure ----------------
    W, H, F = TARGETS[args.target]
    # Full-width (two-column-spanning) float. Two font scales: F is bumped ~20%
    # for readability of the row labels, axis titles and caption, which have room;
    # FH stays at the base size for everything INSIDE the grid, whose cells are
    # constrained by column width rather than by the base font.
    FH = F
    F = F * 1.10 if args.target == "ieee" else F
    ONE = args.target == "ieee1col"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": F,
                         "svg.fonttype": "none", "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[0.45, 3.86, 0.02, 0.02] if ONE
        else [1.15, 4.20, 0.15, 0.60],
        left=0.280 if ONE else 0.205, right=0.988 if ONE else 0.993,
        top=0.936 if ONE else 0.929,
        bottom=0.282 if ONE else 0.254,
        wspace=0.035, hspace=0.115)
    axL = fig.add_subplot(gs[0, 0])
    axH = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[0, 3])

    # ---- left: how high the bar is ----
    axL.barh(np.arange(n_r) + 0.5, bar, height=0.62, color="#909090",
             edgecolor="none", zorder=2)
    # The x axis is inverted, so a label anchored just beyond the bar tip must be
    # RIGHT aligned to grow away from the bar. With ha="left" it grows back over
    # the bar, which is what v2 did.
    for ri in range(n_r):
        axL.text(bar[ri] + bar.max() * 0.045, ri + 0.5, f"{bar[ri]:.1f}",
                 va="center", ha="right", fontsize=F * (0.66 if ONE else 0.74), color="#555555",
                 zorder=3)
    axL.set_ylim(n_r, 0)
    axL.set_xlim(0, bar.max() * (2.55 if ONE else 2.05))
    axL.invert_xaxis()
    axL.set_yticks(np.arange(n_r) + 0.5)
    # The internal keys run the words together; these are the same words as a
    # reader writes them.
    PRETTY_CH = {"LensBlur": "Lens blur", "DirtyLens": "Dirty lens",
                 "GaussianBlur": "Gaussian blur", "CodecError": "Codec error"}
    # 'Decolorization' is the longest and has to sit inside the left margin,
    # which is under nine tenths of an inch at column width.
    axL.set_yticklabels([PRETTY_CH.get(c, c) for c in CHALLENGES],
                        fontsize=F * (0.68 if ONE else 0.94))
    for ri, c in enumerate(CHALLENGES):
        if EXPECT_RESTORABLE[c] == 0:
            axL.get_yticklabels()[ri].set_color(RED)
    axL.set_xticks([])
    axL.tick_params(axis="y", length=0, pad=7)
    for sp in axL.spines.values():
        sp.set_visible(False)
    # The title spans the challenge-name column AND the bars. Setting it as axL's
    # title centres it on the bars alone, which are only about 0.7 in wide, so it
    # spilled right and collided with the first operator label.
    _p = axL.get_position()
    # Left aligned at column width: centred on the narrow left panel it ran off
    # the canvas on both sides.
    fig.text(0.008 if ONE else _p.x1 / 2, _p.y1 + 0.012,
             "noise-injection oracle\n(points, mean of sev 2 to 5)",
             ha="left" if ONE else "center", va="bottom",
             fontsize=F * (0.68 if ONE else 0.76), color="#555555",
             linespacing=1.25)

    # ---- middle: which combinations admit restoration, and by what ----
    # Colour is the family of whatever clears the cell, green for training-free
    # and purple for learned, the two colours these families carry in Fig. 1 and
    # Fig. 2. The grid used to say this with a header spanning the operator
    # columns; there are no operator columns any more, and saying it in the cell
    # is more direct than saying it above the plot.
    vmax = float(np.nanmax(np.where(state == 2, margin, np.nan)))
    for ri in range(n_r):
        for si in range(n_s):
            if state[ri, si] != 2:
                axH.add_patch(Rectangle((si, ri), 1, 1, lw=0, zorder=3,
                                        facecolor=GREY_NOT_ABOVE))
                continue
            op, mg, more = who[(ri, si)]
            tf = op in OPS[:N_TRAINING_FREE]
            shade = 0.20 + 0.80 * (mg / vmax if vmax > 0 else 0.0)
            axH.add_patch(Rectangle(
                (si, ri), 1, 1, lw=0, zorder=2,
                facecolor=(RAMP_FREE if tf else RAMP_LEARN)(shade)))
            # The name of whatever clears the cell, its margin over the
            # noise-injection oracle, and how many other operators also clear
            # it. On rain that reads "DCP 15", one operator and no others; on
            # haze it reads "DCP 46 +5", six of the nine.
            # Two lines: which operator clears it and by how much, then how
            # many of the nine do. One line carrying both, as "Zero-DCE 38 +4",
            # was wider than the cell and ran into its neighbour; and "five of
            # nine" says plainly what "+4" made the reader work out.
            col_t = INK_FREE if tf else INK_LEARN
            # The name on its own line, the count under it. The margin used to
            # ride on the first line as "Zero-DCE 38"; at column width that is
            # wider than the cell, so the margin moved into the depth of the
            # tint and the caption gives its range.
            axH.text(si + 0.5, ri + 0.32, OP_LABEL[op],
                     ha="center", va="center", fontsize=FH * CELL_MULT,
                     zorder=5, fontweight="bold", color=col_t)
            axH.text(si + 0.5, ri + 0.75, f"{more + 1} of {len(OPS)}",
                     ha="center", va="center", fontsize=FH * CELL_MULT * 0.80,
                     zorder=5, color=col_t)
    for ri, si, *_ in dots:
        axH.plot(si + 0.5, ri + 0.5, marker="o", ms=F * 0.40, mfc=RED,
                 mec="white", mew=0.5, zorder=6)

    for si in range(n_s + 1):
        axH.axvline(si, color="white", lw=0.9, zorder=4)
    for ri in range(n_r + 1):
        axH.axhline(ri, color="white", lw=0.9, zorder=4)

    # TWO boxes, not one. They share an edge, so the shared edge draws itself as a
    # divider: the upper box is the high-bar pair (operators gain, noise gains as
    # much) and the lower box is the low-bar pair (operators gain nothing at all).
    r0 = CHALLENGES.index("Decolorization")
    for top in (r0, r0 + 2):
        axH.add_patch(Rectangle((0, top), n_s, 2, fill=False, edgecolor=RED,
                                lw=1.9, zorder=9))

    axH.set_xlim(0, n_s)
    axH.set_ylim(n_r, 0)
    axH.set_yticks([])
    axH.set_xticks(np.arange(n_s) + 0.5)
    axH.set_xticklabels([str(s) for s in SEVS], fontsize=FH * 0.72,
                        color="#666666")
    axH.tick_params(axis="x", length=0, pad=3)
    for sp in axH.spines.values():
        sp.set_visible(False)

    axH.set_xlabel("severity", fontsize=FH * 0.66, color="#999999", labelpad=2)

    # No colour bar. It scaled the intensity of a cell to the margin, and the
    # margin is now written inside the cell as a number, so the bar restated
    # what the reader could already read. Removing it frees half an inch of
    # height, which the rows take, and frees the colour itself to carry
    # something the collapsed grid had lost: which family clears the cell.

    # ---- right: the boundary, and the two kinds ----
    # At column width this panel is dropped. It counts the coloured cells in its
    # own row, which a reader can do by looking, and the quarter inch it takes
    # is the difference between a cell that holds "Zero-DCE" at six point and
    # one that holds it at eight and a half.
    axR.set_xlim(0, 1)
    axR.set_ylim(n_r, 0)
    axR.axis("off")
    if ONE:
        axR.set_visible(False)
    # Sits on the same line as the family labels, not on the operator-label line,
    # where it collided with CIDNet and capped the type size.
    # Right aligned to the column's right edge, not centred on it: the word is
    # 0.75 in wide and the column only 0.36 in, so centring pushed it 0.2 in past
    # the canvas on both sides.
    if not ONE:
        axR.text(1.0, -1.72, "restorable", ha="right", va="bottom",
                 fontsize=F * 0.80, color="#555555")
        for ri, c in enumerate(CHALLENGES):
            r = restorable[c]
            axR.text(0.5, ri + 0.5, f"{r}/4", ha="center", va="center",
                     fontsize=F * 0.98, fontweight="bold",
                     color=RED if r == 0 else ("#14532D" if r >= 3
                                               else "#555555"))
    # WIDTH IS THE SCARCE RESOURCE, HEIGHT IS FREE.
    # The two-kinds note used to live here as bracketed labels beside the rows. It
    # cost 0.83 in of width, which came straight out of the heatmap and forced the
    # cell values down to 6.4 pt. Moved below the plot it costs only height, which
    # does not affect apparent type size at all, and the cells gained 30 per cent.
    # "no operator gains anything" was an overclaim: operators DO change the accuracy
    # on CodecError (gamma is that challenge's best operator at 53.4 per cent). What
    # they fail to do is clear the bar. And "nominal significance" is jargon; a reader
    # who does not know that "nominal" means "uncorrected" simply skips it.
    fig.text(0.5, 0.0889 if ONE else 0.0480,
             "Decolorization and Gaussian blur: operators do gain,\n"
             "but noise gains as much." if ONE else
             "Decolorization and Gaussian blur: operators do gain, "
             "but noise gains as much.",
             ha="center", va="center", fontsize=F * 0.68, color=RED,
             linespacing=1.25)
    fig.text(0.5, 0.0359 if ONE else 0.0180,
             "Codec error and Shadow: the noise bar is low,\nyet no operator "
             "clears it after the correction." if ONE else
             "Codec error and Shadow: the noise bar is low, yet no operator "
             "clears it after the correction.",
             ha="center", va="center", fontsize=F * 0.68, color=RED,
             linespacing=1.25)

    # The full statement of the two kinds lives in the CAPTION, not in the plot.
    # Putting it here as well cost two lines of vertical space and shrank the type
    # for no information gain.

    # Three entries, not four. A cell is now a combination rather than a single
    # operator's test, so it is either cleared by something or by nothing; the
    # old distinction between "not significantly different" and "significantly
    # below" belonged to a per-operator cell and has no meaning here.
    h = [Rectangle((0, 0), 1, 1, fc=RAMP_FREE(0.75), ec=INK_FREE, lw=0.8),
         Rectangle((0, 0), 1, 1, fc=RAMP_LEARN(0.75), ec=INK_LEARN, lw=0.8),
         Rectangle((0, 0), 1, 1, fc=GREY_NOT_ABOVE),
         plt.Line2D([], [], marker="o", ms=F * 0.42, mfc=RED, mec="white", ls="")]
    # One column, four rows. Width is scarce and height is free, and a single column
    # lets every entry carry its full wording instead of a cryptic abbreviation.
    # "clears it per challenge only" was unreadable: nobody can recover "significant
    # under a per-challenge correction but not under the 432-test family correction"
    # from those four words.
    # Short enough to read at a glance. What the text inside a cell means is
    # said in the caption, where there is room for it; a legend entry that runs
    # to a sentence is wider than the page.
    fig.legend(h, ["cleared by a training-free operator (Holm, 432 tests)",
                   "cleared by a learned restorer",
                   "not cleared: nothing beats the noise-injection oracle",
                   "significant, but not after the 432-test correction"],
               loc="lower center", bbox_to_anchor=(0.50, 0.1043 if ONE else 0.0560), ncol=1,
               frameon=False, fontsize=F * 0.80, handlelength=1.20,
               handletextpad=0.55, labelspacing=0.30)

    # ---- layout self-checks: overlaps BEFORE saving, clipping AFTER ----
    print()
    ok_over = _check_overlaps(fig)
    ok_bars = _check_text_over_bars(axL)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # This block is a copy of fig_style.save, made before that helper existed.
    # The copy is why the PDF switch added there had no effect here: one
    # function was changed and its duplicate was not, which is the fourth time
    # in this project that a second copy has gone its own way.
    import os
    png = out / "fig03_capability_boundary.png"
    _fmts = [("png", dict(dpi=400)), ("svg", {})]
    if os.environ.get("FIGPDF"):
        _fmts.append(("pdf", {}))
    for ext, kw in _fmts:
        p = out / f"fig03_capability_boundary.{ext}"
        fig.savefig(p, format=ext, facecolor="white", **kw)
        print(f"  wrote {p}")
    plt.close(fig)

    ok_clip = _check_clipping(png)
    ok_ws = _check_whitespace(png)
    if ok_over and ok_bars and ok_clip and ok_ws:
        print("  LAYOUT CHECK PASSED: no text overlaps, no label on a bar, nothing "
              "clipped, no wasted bands.")
    else:
        print("  LAYOUT CHECK FAILED. Do NOT use this figure; send me the console "
              "output and the PNG.")

    disp = DISPLAY_WIDTH[args.target]
    scale = disp / W
    print(f"\n[layout] target={args.target}: figure {W} x {H} in, base font {F} pt")
    print(f"  Displayed at {disp} in, the scale factor is {scale:.2f}, so the reader")
    print(f"  actually sees:")
    for name, mult in (("challenge names", 0.94), ("restorable N/4", 0.98),
                       ("legend", 0.80),
                       ("cell values", CELL_MULT), ("the two-kinds note", 0.74),
                       ("severity digits", 0.62)):
        print(f"    {name:18s} {F * mult:5.1f} pt in the file  ->  "
              f"{F * mult * scale:5.1f} pt on the page")
    if abs(scale - 1.0) > 0.02:
        print("  WARNING: the figure is not being shown at its native width, so the")
        print("  type is being rescaled. Pick the target that matches your page.")
    else:
        print("  Scale is 1.0: insert at 100 per cent and the point sizes above are")
        print("  exactly what appears on the page. Do NOT let Word resize the image.")
    print("\nCAPTION:")
    for ln in [
        "Fig. 3. The restoration capability boundary. Left: the accuracy gain obtained",
        "by optimally dosed additive Gaussian noise alone, averaged over severities 2",
        "to 5. We call this the noise-injection oracle: an oracle because the noise",
        "level is chosen optimally on the test set, which no deployed system can do,",
        "and an oracle over noise levels only, so an operator can exceed it. An",
        "operator must clear it before its gain can be attributed to restoration,",
        "because a manipulation that restores nothing already reaches that accuracy.",
        "It is distinct from the operator-selection oracle of Section IV-E. Centre: six",
        "operators at severities 2 to 5. A cell is green only where the operator is",
        "significantly above the noise-injection oracle. 432 tests are run (12",
        "challenges by 4 severities by 6 operators), so a per-test threshold of 0.05",
        "would be expected to return about 14 false positives by chance alone;",
        "significance is therefore corrected across the whole family by the",
        "Holm-Bonferroni procedure, which bounds the probability of even one false",
        "positive among all 432 at 5 per cent. It is the most conservative correction",
        "we considered: 225 tests are nominally significant and 172 survive it. Colour",
        "intensity is the margin. Dots mark the thirteen cells whose gain is",
        "nominally significant but does not survive the 432-test correction; all are",
        "reported as not restoring. Right:",
        "restorable severities out of four; the two red boxes are the two kinds of",
        "unrestorable, stated below the plot. Severity 1 carries no verdict by",
        "pre-registration. The test bounds the benefit attributable to restoration; it",
        "does not establish a mechanism. Scope: CompactCNN (145,291 parameters, 32x32",
        "input) on CURE-TSR; the learned models are AdaIR (28.8M) and CIDNet (2.0M).",
    ]:
        print("  " + ln)


if __name__ == "__main__":
    main()
