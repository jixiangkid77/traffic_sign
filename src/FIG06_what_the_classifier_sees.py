# -*- coding: utf-8 -*-
"""
FIG06_what_the_classifier_sees.py
Paper figure 6. Script, output and paper number agree.

Figure 6: six crops, the front ends applied to them, and the label each one
produced. Failures first.

WHY EVERY CELL IS 32 x 32
The number printed under a cell is the class the classifier returned, and the
classifier returns it from a 32 x 32 array. Showing the crop at its own size
instead would put a different image next to that number: larger than the
classifier's input on four of these six crops, smaller on two. Each cell is
therefore the classifier's own input, produced with the same cv2.resize call the
pipeline uses, and magnified for print without interpolation so the reader sees
the pixels rather than a smoothed guess at them. The original crop size is
printed on the row, so nothing is hidden by the choice.

A SIZE FLOOR WAS CONSIDERED AND REJECTED
An earlier plan filtered the candidates to crops of at least 32 x 32 on the
grounds that smaller ones show the reader less than the classifier uses. That is
backwards: a crop of 11 x 19 is upscaled before classification, so those 209
pixels ARE everything the classifier has, while a crop of 61 x 58 is downscaled
and the classifier throws information away. Worse, small crops are the harder
ones, so a size floor would quietly drop the hardest cases from a figure whose
first three rows are failures. The floor was dropped.

HOW THE SIX WERE CHOSEN, WHICH IS NOT BY EYE
For each row the candidate set is every crop that shows that row's outcome, taken
from the first of the two occurrences a filename has under the pooled Real_Train
and Real_Test splits. Rows choose in order of how few sign classes their set offers,
most constrained first, and each row then takes the first candidate by filename
whose class no earlier row has used. Candidate sets held between 83 and 784
crops, so none of these is a rarity. Five of the six sign classes differ; class 9
appears twice, once under snow and once under haze, which lets the same sign be
compared across two degradations.

WHAT THE COLUMNS ARE
Left is the degraded crop with no enhancement. Middle is the best training-free
operator ON THAT CHALLENGE, right the best learned restorer on that challenge,
the same per-challenge choice Fig. 1 makes; that is why the middle column is
CLAHE on blur, gamma on darkening and the dark channel prior elsewhere.

READS   outputs_revision/fig06_samples/*.png       (EXPORT_fig06_samples.py)
        outputs_revision/fig06_samples/manifest.json
        outputs_revision/merged_per_image.csv      (to re-check the labels)
        outputs_revision/dcp_cure.csv
WRITES  outputs_revision/figures/fig06_what_the_classifier_sees.png   (400 dpi)
        outputs_revision/figures/fig06_what_the_classifier_sees.svg
RUN     python FIG06_what_the_classifier_sees.py
        python FIG06_what_the_classifier_sees.py --target word
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import fig_style as fs

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FIG_DIR = OUT_DIR / "figures"

INPUT_SIZE = 32
CH_NAME = {4: "darkening", 7: "Gaussian blur", 8: "noise", 9: "rain",
           11: "snow", 12: "haze"}
PRETTY = {"degraded": "no enhancement", "clahe": "CLAHE", "dcp": "DCP",
          "gamma": "Gamma", "stretch": "Stretch", "adair": "AdaIR",
          "cidnet": "CIDNet", "zero_dce": "Zero-DCE", "ffa_net": "FFA-Net",
          "promptir": "PromptIR"}

GREEN = "#2E7D4F"
RED = "#B3261E"

# Nothing about the six cases is written here. The manifest the export wrote
# carries the crop, its class, its size, the two front ends the challenge
# selects and what each cell classified as, and the export checked every one of
# those against the merged file before writing them. A second copy in this file
# could only disagree with the first, and when the pool of learned restorers
# grew from two to five that is exactly what the old copy did: it went on naming
# AdaIR on rain after FFA-Net had become the best learned restorer there.
ROWS = ["F1", "F2", "F3", "S1", "S2", "S3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(fs.TARGETS),
                    default="ieee1col")
    ap.add_argument("--samples", default=str(OUT_DIR / "fig06_samples"))
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(FIG_DIR))
    args = ap.parse_args()

    sdir = Path(args.samples)
    if not sdir.exists():
        raise SystemExit(f"ABORT: {sdir} not found. Run EXPORT_fig8_samples.py "
                         f"first.")
    man = {c["tag"]: c for c in json.load(open(sdir / "manifest.json"))}

    # ---- the labels on record, which the figure must not contradict ----
    rec = {}
    for r in csv.DictReader(open(args.merged, newline="", encoding="utf-8")):
        rec[(r["filename"], int(r["occ"]))] = dict(r)
    dcp_rec = {}
    for r in csv.DictReader(open(args.dcp, newline="", encoding="utf-8")):
        dcp_rec[(r["filename"], int(r["occ"]))] = int(r["pred_dcp"])

    print("=== AUDIT AGAINST THE RECORD ===")
    bad = 0
    cells = {}
    if [m["tag"] for m in json.load(open(sdir / "manifest.json"))] != ROWS:
        print(f"  MISMATCH: the manifest does not carry {ROWS} in that order")
        bad += 1
    for tag in ROWS:
        if tag not in man:
            print(f"  MISMATCH {tag}: not in manifest")
            bad += 1
            continue
        m = man[tag]
        key = (m["fn"], m["occ"])
        row = rec.get(key)
        if row is None:
            print(f"  MISMATCH {tag}: {key} is not in {args.merged}")
            bad += 1
            continue
        # the crop is the crop the manifest says it is
        if (int(row["ch"]), int(row["sev"]), int(row["true"])) != \
                (m["ch"], m["sev"], m["true"]):
            print(f"  MISMATCH {tag}: the merged file has challenge "
                  f"{row['ch']} severity {row['sev']} class {row['true']}, "
                  f"the manifest says {m['ch']}/{m['sev']}/{m['true']}")
            bad += 1
        # the three cells are the three the caption promises, in that order
        want = ["degraded", m["best_free"], m["best_learned"]]
        if [c["what"] for c in m["cells"]] != want:
            print(f"  MISMATCH {tag}: the cells are "
                  f"{[c['what'] for c in m['cells']]}, the caption promises "
                  f"{want}")
            bad += 1
        for cell, fname in zip(m["cells"], m["files"]):
            what, label = cell["what"], cell["pred"]
            png = sdir / fname
            if not png.exists():
                print(f"  MISMATCH {tag}: {fname} missing")
                bad += 1
                continue
            img = cv2.imread(str(png), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if [w, h] != m["size"]:
                print(f"  MISMATCH {tag} {what}: crop is {w}x{h}, the "
                      f"manifest says {m['size'][0]}x{m['size'][1]}")
                bad += 1
            # the label on record for this exact front end
            col = "passthrough" if what == "degraded" else what
            on_record = (dcp_rec[key] if col == "dcp"
                         else int(row[f"pred_{col}"]))
            if on_record != label:
                print(f"  MISMATCH {tag} {what}: the record says {on_record}, "
                      f"the figure would print {label}")
                bad += 1
            # the classifier's own input, made the way the pipeline makes it
            small = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            cells[(tag, what)] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # The outcome each row states, checked against the whole family and not
    # against the one member the row prints. A row that says only a learned
    # restorer recovers the sign is wrong if gamma does too, whether or not
    # gamma is on the page.
    FREE = ["gamma", "clahe", "stretch", "dcp"]
    LEARN = ["adair", "cidnet", "zero_dce", "ffa_net", "promptir"]
    per_row = {}
    for tag in ROWS:
        m = man[tag]
        got = set(m["recovered_by"])
        tf, ln = got & set(FREE), got & set(LEARN)
        want = {"none": not got, "learned": ln and not tf,
                "free": tf and not ln, "both": tf and ln}[m["outcome"]]
        if not want:
            print(f"  MISMATCH {tag}: outcome '{m['outcome']}' against "
                  f"training-free {sorted(tf) or 'none'} and learned "
                  f"{sorted(ln) or 'none'}")
            bad += 1
        per_row[tag] = sum(1 for c in m["cells"] if c["pred"] == m["true"])
    n_right = sum(per_row.values())
    print(f"  6 rows, 18 cells, {n_right} correct; per row {per_row}")
    print("  AUDIT PASSED: every printed label is the label on record."
          if bad == 0 else
          f"  AUDIT: {bad} mismatch(es). Do NOT use this figure.")

    # ---------------- figure ----------------
    W, H, F = fs.TARGETS[args.target]
    ONE = args.target == "ieee1col"
    F = F * 1.10 if args.target == "ieee" else F
    H = 5.75 if args.target == "ieee" else (7.72 if ONE else 5.50)
    fs.rc(F)
    fig = plt.figure(figsize=(W, H))

    # left is solved, not guessed: at this height and this gap a row can hold a
# square 0.626 inch on a side, so the three columns are made exactly that wide
# and none of the width is left over beside the pictures.
    left, right = (0.359 if ONE else 0.208), (0.985 if ONE else 0.995)
    top, bottom = (0.928 if ONE else 0.925), (0.093 if ONE else 0.135)
    n_r, n_c = len(ROWS), 3
    gap_x, gap_y = 0.016, (0.014 if ONE else 0.030)
    cw = (right - left - gap_x * (n_c - 1)) / n_c
    chh = (top - bottom - gap_y * (n_r - 1)) / n_r

    for i, tag in enumerate(ROWS):
        e = man[tag]
        y0 = top - (i + 1) * chh - i * gap_y
        # row label
        if ONE:
            # Four short lines, not two long ones. The column has to be as wide
            # as its longest line, and at two lines that was "class 12 . 11x19"
            # at sixteen characters, which left a third of the column blank
            # beside every shorter row. Broken up, the longest is thirteen, the
            # column narrows by a quarter inch and the pictures take it.
            for k, (txt, mult, col) in enumerate([
                    (CH_NAME[e["ch"]], 1.14, fs.INK),
                    (f"severity {e['sev']}", 1.00, fs.INK),
                    (f"class {e['true']}", 1.00, fs.MUTED),
                    (f"{e['size'][0]}x{e['size'][1]} crop", 1.00, fs.MUTED)]):
                fig.text(left - 0.014, y0 + chh * (0.88 - 0.195 * k), txt,
                         ha="right", va="center", fontsize=F * mult, color=col)
        else:
            fig.text(left - 0.014, y0 + chh * 0.62,
                     f"{CH_NAME[e['ch']]} {e['sev']}", ha="right", va="center",
                     fontsize=F * 0.72, color=fs.INK)
            fig.text(left - 0.014, y0 + chh * 0.30,
                     f"class {e['true']}, {e['size'][0]}x{e['size'][1]} crop",
                     ha="right", va="center", fontsize=F * 0.62,
                     color=fs.MUTED)
        for j, (what, label) in enumerate((c["what"], c["pred"])
                                          for c in e["cells"]):
            x0 = left + j * (cw + gap_x)
            if ONE:
                # The name and the class sit together under the picture they
                # belong to, the name first. On one line "Zero-DCE 13" comes to
                # under six point in a cell this wide; stacked, both are read at
                # full size, and the width stops being what limits them.
                ax = fig.add_axes([x0, y0 + chh * 0.36, cw, chh * 0.62])
                if what != "degraded":
                    fig.text(x0 + cw / 2, y0 + chh * 0.24, PRETTY[what],
                             ha="center", va="center", fontsize=F * 0.88,
                             color=fs.MUTED)
            else:
                ax = fig.add_axes([x0, y0 + chh * 0.24, cw, chh * 0.76])
            ax.imshow(cells[(tag, what)], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ok = label == e["true"]
            for sp in ax.spines.values():
                sp.set_color(GREEN if ok else "#CCCCCC")
                sp.set_linewidth(1.9 if ok else 0.8)
            fig.text(x0 + cw / 2, y0 + chh * (0.06 if ONE else 0.11),
                     f"{label}" if ONE else f"{PRETTY[what]}  {label}",
                     ha="center", va="center",
                     fontsize=F * (1.10 if ONE else 0.64),
                     color=GREEN if ok else fs.MUTED)

    for j, head in enumerate(["degraded", "training-\nfree", "learned"] if ONE
                             else ["degraded", "best training-free",
                                   "best learned"]):
        fig.text(left + j * (cw + gap_x) + cw / 2, top + 0.014, head,
                 ha="center", va="bottom", fontsize=F * (0.80 if ONE else 0.74),
                 color=fs.INK, linespacing=1.1)
    fig.text(left - 0.014, top + 0.014, "failures first", ha="right",
             va="bottom", fontsize=F * 0.72, color=RED)

    # Two lines at column width, not three. What each cell is and what the
    # border means are the two a reader needs before the first glance; that the
    # classes are GTSRB indices and the scope is one classifier on one data set
    # are said in the caption, where a longer sentence costs nothing.
    # One line at column width, not two. What each cell is and what is printed
    # under it are said in the caption, where a sentence can be a sentence; the
    # border is a visual code and has to be read off the figure itself, so it
    # keeps the space and gets the larger type.
    # This line stays on the figure. A reader meets six rows of blocky pictures
    # before meeting the caption, and the first thing it answers is whether the
    # figure is badly made. The caption cannot answer that in time; it is read
    # second. The line costs a sixth of an inch and no type size.
    fig.text(0.5, 0.0680 if ONE else 0.0880,
             "the 32 x 32 the classifier scored" if ONE else
             "Each cell is the 32 x 32 array the classifier scored, "
             "magnified without interpolation.",
             ha="center", va="center", fontsize=F * (0.78 if ONE else 0.68),
             color=fs.INK)
    fig.text(0.5, 0.0270 if ONE else 0.0540,
             f"green: returned the true class, {n_right} of 18" if ONE
             else f"A green border marks a cell that returned the true class: "
                  f"{n_right} of 18.",
             ha="center", va="center", fontsize=F * (0.84 if ONE else 0.68),
             color=GREEN)
    if not ONE:
        fig.text(0.5, 0.0200,
                 "Classes are GTSRB indices. Scope: CompactCNN on CURE-TSR.",
                 ha="center", va="center", fontsize=F * 0.62, color=fs.MUTED)

    ok_gate = fs.run_gates(fig, args.outdir, "fig06_what_the_classifier_sees")
    fs.report_sizes(args.target, W, F, {
        "column heads": 0.80, "row names": 1.14, "row detail": 1.00,
        "the front ends": 0.88, "cell labels": 1.10,
        "the 32 x 32 note": 0.78, "the green note": 0.84} if ONE else {
        "column heads": 0.74, "row names": 0.72, "row detail": 0.62,
        "cell labels": 0.64, "the two notes": 0.68, "the scope note": 0.62})

    print("\nCAPTION:")
    for ln in [
        "Fig. 6. Six crops, the front ends applied to them, and the class each one",
        "produced, failures first. Every cell is the 32 x 32 array the classifier",
        "actually scored, built with the same resize the pipeline uses and magnified",
        "for print without interpolation, so the picture beside a label is the picture",
        "that produced it; the original crop size is printed on each row. Left is the",
        "degraded crop with no enhancement, middle the best training-free operator on",
        "that challenge and right the best learned restorer on that challenge, the same",
        "per-challenge choice Fig. 1 makes, which is why the middle column is CLAHE on",
        "Gaussian blur, gamma on darkening and the dark channel prior elsewhere. A",
        "green border marks the cell that returned the true class. The first three rows",
        "are failures of the training-free route: on severe Gaussian blur nothing",
        "recovers the sign, and on noise and snow the learned restorer recovers it",
        "while the best training-free operator does not. The last three are the other",
        "direction: on haze and rain the dark channel prior recovers a sign that",
        "neither no enhancement nor AdaIR does, and on darkening gamma recovers one",
        "that the dark channel prior does not, which is the branch that Fig. 2 shows",
        "the routing rule earns its keep on. The six were chosen by rule and not by",
        "eye: for each row the candidates are every crop showing that row's outcome,",
        "taken from the first of the two occurrences a filename has under the pooled",
        "Real_Train and Real_Test splits, and rows choose in order of how few sign",
        "classes their candidate set offers, and each row takes the first candidate by",
        "filename whose class no earlier row has used. The candidate sets held between",
        "83 and 784 crops. Classes are GTSRB indices. Scope: CompactCNN (145,291",
        "parameters, 32 x 32 input) on CURE-TSR.",
    ]:
        print("  " + ln)
    if not ok_gate or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
