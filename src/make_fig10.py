r"""
make_fig10.py  Figure 10: qualitative input/output of the two deep models
on real CURE-TSR degradations (darkening, haze, rain), at print resolution.

WHY A SEPARATE SCRIPT
  The earlier sample dumps were tiny thumbnails (about 15 px per panel) and
  are too low-resolution to print. This script re-runs AdaIR and HVI-CIDNet
  on full-resolution real CURE crops and lays out a clean 3x3 comparison.
  It reuses the exact enhancement code path of J_local_deep_eval.py, so the
  images shown are the same ones the accuracy tables were computed from.

WHERE TO PUT THIS FILE
  Place it in  D:\Project\traffic_sign\src\  (next to J_local_deep_eval.py).
  It reads the real CURE-TSR dataset from J's default location and writes
  PNG + SVG into  ..\outputs_revision\figures\ .
  If your project lives elsewhere, edit PROJECT_ROOT below.

RUN
  python make_fig10.py
  python make_fig10.py --sign 3               # restrict to one sign type
  python make_fig10.py --cidnet-weight D:\Project\traffic_sign\models\cidnet_hvi.safetensors

REQUIRES
  the same conda env used for the experiments (torch, opencv, matplotlib);
  models\adair5d.ckpt present; CIDNet weights in the HF cache or passed with
  --cidnet-weight. CPU is fine; three crops enhance in a few seconds.

OUTPUT
  ..\outputs_revision\figures\Fig10_deep_qualitative.png  (and .svg)
  plus a printed list of the exact crops chosen and their pixel sizes.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")     # <-- edit if needed
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import J_local_deep_eval as J                        # noqa: E402

FIGDIR = PROJECT_ROOT / "outputs_revision" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# challenge codes in CURE-TSR
CH = {"Darkening": 4, "Rain": 9, "Haze": 12}
ROWS = ["Darkening", "Haze", "Rain"]                 # display order
SEV = 5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def pick_crop(samples, ch_code, sign_filter, min_size):
    """Among sev-5 crops of this challenge, return the path of the crop with
    the largest short side (crispest for print), optionally restricted to a
    sign type and to crops at least min_size on the short side."""
    cands = [s for s in samples if s["ch"] == ch_code and s["sev"] == SEV
             and (sign_filter is None or s["sign"] == sign_filter)]
    best, best_short = None, -1
    for s in cands:
        im = cv2.imread(str(s["path"]))
        if im is None:
            continue
        short = min(im.shape[0], im.shape[1])
        if short < min_size:
            continue
        if short > best_short:
            best, best_short = s, short
    if best is None and min_size > 0:      # relax the size floor if needed
        return pick_crop(samples, ch_code, sign_filter, 0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", type=int, default=None,
                    help="restrict to one CURE sign type for visual "
                         "consistency across rows (optional)")
    ap.add_argument("--min-size", type=int, default=48,
                    help="preferred minimum short side in pixels")
    ap.add_argument("--cure-root", default=str(J.CURE_TSR_DIR_DEFAULT))
    ap.add_argument("--adair-weight",
                    default=str(PROJECT_ROOT / "models" / "adair5d.ckpt"))
    ap.add_argument("--cidnet-weight", default="",
                    help="local CIDNet weight; default uses the HF cache")
    ap.add_argument("--interp", default="bilinear",
                    help="imshow interpolation: bilinear (smooth) or nearest")
    args = ap.parse_args()

    samples, n_files = J.collect_samples(Path(args.cure_root))
    if not samples:
        sys.exit(f"[FATAL] no CURE crops under {args.cure_root}")
    print(f"scanned {n_files} files; {len(samples)} usable crops")

    chosen = {}
    for name in ROWS:
        s = pick_crop(samples, CH[name], args.sign, args.min_size)
        if s is None:
            sys.exit(f"[FATAL] no crop found for {name} at severity {SEV}")
        chosen[name] = s
        im = cv2.imread(str(s["path"]))
        print(f"  {name:10s} -> {s['filename']}  "
              f"({im.shape[1]}x{im.shape[0]} px, sign {s['sign']})")

    print("loading AdaIR ...")
    adair, _ = J.load_adair("cpu", args.adair_weight)
    print("loading HVI-CIDNet ...")
    cidnet, _ = J.load_cidnet("cpu", args.cidnet_weight)

    # enhance each chosen crop with both models
    panels = {}   # name -> (input_rgb, adair_rgb, cidnet_rgb)
    for name in ROWS:
        bgr = cv2.imread(str(chosen[name]["path"]))
        a = J.enhance_batch(adair, [bgr], "cpu")[0]
        c = J.enhance_batch(cidnet, [bgr], "cpu")[0]
        to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        panels[name] = (to_rgb(bgr), to_rgb(a), to_rgb(c))

    # ---- 3x3 figure, large labels, no overlap ----
    col_titles = ["Degraded input", "AdaIR", "HVI-CIDNet"]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.4))
    for r, name in enumerate(ROWS):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(panels[name][c], interpolation=args.interp)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333"); spine.set_linewidth(1.2)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=19, fontweight="bold",
                             pad=12)
            if c == 0:
                ax.set_ylabel(name, fontsize=18, fontweight="bold",
                              rotation=90, labelpad=14)
    fig.suptitle("Deep restoration on real CURE-TSR crops "
                 "(severity 5)", fontsize=20, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("png", "svg"):
        fig.savefig(FIGDIR / f"Fig10_deep_qualitative.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGDIR / "Fig10_deep_qualitative.png")


if __name__ == "__main__":
    main()
