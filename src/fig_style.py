# -*- coding: utf-8 -*-
"""
fig_style.py
Shared sizing rules, palette, and layout gates for every figure in the paper.

THE ONE RULE THAT GOVERNS APPARENT TYPE SIZE
Word (and LaTeX) scale an image to the available width. What the reader actually
sees is

    apparent_pt = font_pt * (display_width / figure_width)

so making a figure PHYSICALLY BIGGER makes its type LOOK SMALLER, because the
page then shrinks it harder. A 10.4 inch figure at 11.5 pt, dropped into a 7.16
inch column, renders at 7.9 pt. Every figure here is therefore built AT the width
it will be displayed at, so the scale factor is 1.0 and the point sizes are
exactly what the reader sees.

WIDTH IS SCARCE, HEIGHT IS FREE
Width buys type size; height costs nothing. Any annotation that can go below the
plot instead of beside it should go below the plot.

THREE GATES, RUN ON EVERY FIGURE BEFORE IT IS USED
  check_overlaps        no two text objects may overlap OR merely abut. Checking
                        only for overlap is not enough: labels that touch with
                        zero overlap read as one word, and an earlier draft
                        rendered the operator row as "GammaCLAHEStretch" and the
                        cell values as "2529" while the overlap test passed them
                        all. Every box is inflated by PAD_PX first.
  check_text_over_bars  a bar's value label must sit OUTSIDE its bar. On an
                        inverted x axis, ha="left" grows the label back over the
                        bar; ha="right" is correct.
  check_clipping        no ink within BORDER_PX of the canvas edge.

Usage:
    import fig_style as fs
    W, H, F = fs.TARGETS["ieee"]
    ...build the figure...
    ok = fs.run_gates(fig, png_path, bar_axes=[axL])
    fs.report_sizes("ieee", W, F, {"challenge names": 0.94, ...})
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- sizing
TARGETS = {                     # figure width, height (inches), base font (pt)
    "ieee": (7.16, 7.00, 12.75),   # T-ITS full width, spans both columns
    # One T-ITS column, measured from the template: the text block is 7.16 in
    # across, split into two columns of 3.48 with a 0.18 gutter. A figure drawn
    # at 7.16 and dropped into a column is shown at 49 per cent, which halves
    # every label; drawing at 3.48 keeps the type at the size it was set.
    "ieee1col": (3.48, 4.30, 10.80),
    "word": (6.50, 6.35, 11.55),   # plain portrait page, 1 inch margins
}
DISPLAY_WIDTH = {"ieee": 7.16, "ieee1col": 3.48, "word": 6.50}

# Constrained by COLUMN WIDTH, not by the base font, so they do not scale with F:
# six operator names must sit side by side, and a two-digit value must sit inside
# one column with a visible gap. Raising these is what produced "GammaCLAHEStretch".
OP_MULT = 0.78
CELL_MULT = 0.68

PAD_PX = 2.0
BORDER_PX = 6.0
DPI = 400

# ---------------------------------------------------------------- palette
# Colourblind-safe: one green ramp for "restores", neutral greys otherwise. No
# red/green pairing carries meaning; red is used only for annotation.
CMAP = LinearSegmentedColormap.from_list(
    "restore", ["#E8F4EA", "#A8D5B5", "#5FAF7D", "#2E7D4F", "#14532D"])
GREY_NOT_ABOVE = "#EAEAEA"
GREY_WORSE = "#C2C2C2"
GREY_BAR = "#909090"
# The two families, used the same way in every figure that shows both:
# green for training-free, purple for learned.
GREEN = "#2E7D4F"
PURPLE = "#8C5FBF"
RED = "#B3261E"
INK = "#222222"
MUTED = "#555555"
FAINT = "#999999"


def rc(F):
    """Global rcParams. svg.fonttype='none' keeps SVG text editable."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": F,
                         "svg.fonttype": "none", "axes.linewidth": 0.7})


# ---------------------------------------------------------------- gates
def check_overlaps(fig, pad_px=PAD_PX):
    """No two text objects may overlap or come within pad_px of each other.

    The figure must be drawn first (before a draw every text sits at its
    untransformed origin and the boxes all collide spuriously), and this must run
    BEFORE savefig, because saving to SVG swaps the canvas and leaves the Agg
    renderer stale.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        items += [t for t in ax.texts if t.get_text().strip() and t.get_visible()]
        if ax.axison:      # an axes with axis("off") still holds invisible ticks
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                if t.get_text().strip() and t.get_visible():
                    items.append(t)
        if ax.title.get_text().strip():
            items.append(ax.title)
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
            gx = max(bj.x0 - bi.x1, bi.x0 - bj.x1)
            gy = max(bj.y0 - bi.y1, bi.y0 - bj.y1)
            if gx < pad_px and gy < pad_px:
                bad.append((ti, tj, max(gx, gy)))
    if bad:
        print(f"  LAYOUT WARNING: {len(bad)} text pair(s) closer than {pad_px} px, "
              f"among {len(boxes)} text objects:")
        for a, b, g in sorted(bad, key=lambda x: x[2])[:8]:
            how = f"overlap by {-g:.1f} px" if g < 0 else f"only {g:.1f} px apart"
            print(f"    '{a}' and '{b}': {how}")
        return False
    print(f"  ({len(boxes)} text objects checked pairwise; every pair is at least "
          f"{pad_px} px apart.)")
    return True


def check_text_over_bars(ax, min_frac=0.10):
    """A bar's value label must sit outside the bar, not on top of it."""
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
            frac = ov / max(tb.width * tb.height, 1.0)
            if frac > min_frac:
                bad.append((t.get_text(), 100 * frac))
                break
    if bad:
        print(f"  LAYOUT WARNING: {len(bad)} bar label(s) drawn on top of their bar:")
        for txt, pct in bad[:6]:
            print(f"    '{txt}' is {pct:.0f}% covered by the bar")
        return False
    return True


def check_text_over_lines(ax, pad_px=1.0):
    """No text may sit on top of a plotted line.

    The overlap gate only compares text with text. It happily passed a draft in
    which the label "AdaIR +4.9" was printed straight across the noise curve,
    because a curve is not a text object. This gate closes that hole: every Text in
    the axes is tested against every Line2D segment, in display coordinates.

    Legends are exempt: a legend is allowed to sit over empty canvas, and
    loc="best" already places it where it overlaps the data least.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    segs = []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xy = ln.get_xydata()
        if len(xy) < 2:
            continue
        pts = ax.transData.transform(xy)
        segs += list(zip(pts[:-1], pts[1:]))

    def hits(bb, p, q):
        # Liang-Barsky: does segment p->q intersect the (inflated) box?
        x0, y0, x1, y1 = bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px
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

    bad = []
    for t in ax.texts:
        if not t.get_text().strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=r)
        for p, q in segs:
            if hits(bb, p, q):
                bad.append(t.get_text().replace("\n", " "))
                break
    if bad:
        print(f"  LAYOUT WARNING: {len(bad)} label(s) printed on top of a plotted "
              f"line:")
        for txt in bad[:6]:
            print(f"    '{txt}'")
        return False
    return True


def check_clipping(png, border=BORDER_PX):
    """No ink may sit within `border` pixels of the canvas edge."""
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed; clipping check skipped.)")
        return True
    ink = np.array(Image.open(png).convert("L")) < 245
    b = int(border)
    touch = {"top": int(ink[:b, :].sum()), "bottom": int(ink[-b:, :].sum()),
             "left": int(ink[:, :b].sum()), "right": int(ink[:, -b:].sum())}
    if any(touch.values()):
        print("  LAYOUT WARNING: content is clipped at the canvas edge:")
        for k, v in touch.items():
            if v:
                print(f"    {k}: {v} ink pixels within {b} px of the edge")
        return False
    return True


def check_whitespace(png, max_band_in=0.25, dpi=DPI):
    """No blank horizontal band may exceed max_band_in inches.

    Height is free, which is true and useful, but it is not a licence to leave the
    slack lying around as gaps. An earlier draft grew to 7.5 in so a four-row legend
    would fit, and 1.20 in of that (16 per cent of the figure) ended up as three
    empty bands rather than as content.
    """
    try:
        from PIL import Image
    except ImportError:
        return True
    ink = np.array(Image.open(png).convert("L")) < 245
    H, W = ink.shape
    dens = ink.sum(axis=1) / W
    bands, i, bad = [], 0, []
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
        print(f"    total wasted: {sum(h for h,_,_ in bad):.2f} in of "
              f"{H/dpi:.2f} in")
        return False
    return True


def save(fig, outdir, stem, pdf=None):
    """PNG and SVG always; PDF when the article is being set in LaTeX.

    The rule here used to be PNG and SVG only, because the article was built as
    a Word document and Word takes neither PDF nor SVG well. LaTeX takes PDF and
    nothing else without an external converter, so the format follows the
    typesetter rather than a habit. Set the environment variable FIGPDF, or pass
    pdf=True, to add it; the default is unchanged so that an old command line
    produces exactly the files it produced before.
    """
    import os
    if pdf is None:
        pdf = bool(os.environ.get("FIGPDF"))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stem}.png"
    fmts = [("png", dict(dpi=DPI)), ("svg", {})]
    if pdf:
        fmts.append(("pdf", {}))
    for ext, kw in fmts:
        p = out / f"{stem}.{ext}"
        fig.savefig(p, format=ext, facecolor="white", **kw)
        print(f"  wrote {p}")
    return png


def run_gates(fig, outdir, stem, bar_axes=(), line_axes=()):
    """Overlaps before saving (SVG invalidates the renderer), clipping after."""
    print()
    ok = check_overlaps(fig)
    for ax in bar_axes:
        ok = check_text_over_bars(ax) and ok
    for ax in line_axes:
        ok = check_text_over_lines(ax) and ok
    png = save(fig, outdir, stem)
    plt.close(fig)
    ok = check_clipping(png) and ok
    ok = check_whitespace(png) and ok
    print("  LAYOUT CHECK PASSED: no text overlaps, no label on a bar or a line, "
          "nothing clipped, no wasted bands." if ok else
          "  LAYOUT CHECK FAILED. Do NOT use this figure; send me the console "
          "output and the PNG.")
    return ok


def report_sizes(target, W, F, elements):
    """Print the point size the reader will actually see on the page."""
    disp = DISPLAY_WIDTH[target]
    scale = disp / W
    print(f"\n[layout] target={target}: figure {W} x ? in, base font {F} pt")
    print(f"  Displayed at {disp} in, scale factor {scale:.2f}, so the reader sees:")
    for name, mult in elements.items():
        print(f"    {name:20s} {F * mult:5.1f} pt in the file  ->  "
              f"{F * mult * scale:5.1f} pt on the page")
    if abs(scale - 1.0) > 0.02:
        print("  WARNING: the figure is not shown at its native width, so the type is")
        print("  being rescaled. Pick the target that matches your page.")
    else:
        print("  Scale is 1.0: insert at 100 per cent and the sizes above are exactly")
        print("  what appears on the page. Do NOT let Word resize the image.")
