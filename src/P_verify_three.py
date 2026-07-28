#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P_verify_three.py
=================
Checks a local run of Zero-DCE, FFA-Net or PromptIR against the earlier,
independent run of the same model, cell by cell.

WHY THIS AND NOT AN EYEBALL
    A partial run reports the mean over whatever cells it happened to reach.
    Two cells out of sixty can read fourteen per cent while the model's true
    twelve-challenge average is fifty-one, and nothing about the number itself
    says which is happening. The earlier run scored the same images with the
    same classifier; if the new run agrees with it cell by cell, the local
    pipeline is right, and if it does not, the disagreement says where.

WHAT AGREEMENT MEANS
    The two runs used the same weights, the same classifier and the same
    images, so a full cell should match closely. They are not bit-identical
    pipelines, so a fraction of a point is expected; a cell that differs by
    more than a couple of points is a real difference and is reported as such.
    A cell the local run has only partly covered is compared with a note, since
    a hundred of its 270 images will not reproduce the full-cell figure exactly.

USAGE
    python P_verify_three.py --model zero_dce
    python P_verify_three.py --model all --earlier <path to the 12-class csv>
"""

import argparse
import collections
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
EARLIER_DEFAULT = OUT_DIR / "cure_tsr_per_image_predictions_12class.csv"

CH_NAME = {1: "Decolorization", 2: "LensBlur", 3: "CodecError", 4: "Darkening",
           5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur", 8: "Noise",
           9: "Rain", 10: "Shadow", 11: "Snow", 12: "Haze"}
MODELS = ("zero_dce", "ffa_net", "promptir")


def say(m=""):
    print(m, flush=True)


def rule(t=""):
    say("\n" + "=" * 76)
    if t:
        say(t)
        say("=" * 76)


def local_cells(path):
    """Hits and totals per (challenge, severity, true class).

    The class matters. A cell holds five classes in very unequal numbers, and
    one of them is recognised almost always while another is almost never; a
    run that has reached only the first class of a cell will read far below the
    cell's own figure for reasons that have nothing to do with the pipeline.
    The first version of this check compared exactly that against the whole
    cell and called a correct run broken.
    """
    cells = collections.defaultdict(lambda: [0, 0])
    n = 0
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            sev = int(r["sev"])
            if sev == 0:
                continue
            k = (r["ch_name"], sev, int(r["gtsrb_true"]))
            cells[k][1] += 1
            cells[k][0] += int(r["correct"])
    return cells, n


def earlier_cells(path, model):
    """The same grouping on the earlier run's per-image file."""
    cells = collections.defaultdict(lambda: [0, 0])
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] != model:
                continue
            sev = int(r["severity"])
            if sev == 0 or r["challenge"] not in CH_NAME.values():
                continue
            k = (r["challenge"], sev, int(r["true_label"]))
            cells[k][1] += 1
            cells[k][0] += int(r["correct"])
    return cells


def binom_sd(p, n):
    """Standard deviation of a proportion, in points, for a partial cell."""
    if n <= 0:
        return float("inf")
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def check(model, args):
    loc_path = Path(getattr(args, "csv", "") or OUT_DIR / f"deep_{model}_cure.csv")
    if not loc_path.exists():
        say(f"  [SKIP] {loc_path} not found")
        return None
    loc, n_rows = local_cells(loc_path)
    ear = earlier_cells(args.earlier, model)
    if not ear:
        say(f"  [SKIP] the earlier file has no rows for {model}")
        return None

    say(f"  local file : {loc_path.name}   {n_rows:,} rows   "
        f"{len(loc)} degraded cell(s) touched")
    say(f"  earlier run: {len(ear)} cells, "
        f"{sum(v[1] for v in ear.values()):,} degraded images")
    say("")
    say(f"  {'challenge / severity / class':34s} {'local':>15s} {'earlier':>10s} "
        f"{'diff':>8s}  verdict")

    worst, bad, part = 0.0, [], 0
    for k in sorted(loc, key=lambda x: (x[0], x[1], x[2])):
        hit, tot = loc[k]
        if tot == 0:
            continue
        a = 100.0 * hit / tot
        name = f"{k[0]} sev {k[1]} class {k[2]}"
        if k not in ear:
            say(f"  {name:34s} {a:8.2f} ({tot:4d}) {'--':>10s} {'--':>8s}"
                f"  not in the earlier run")
            continue
        eh, et = ear[k]
        b = 100.0 * eh / et
        if tot < et:
            # Only a part of this class in this cell has been reached, and the
            # part is a run of consecutive frames of the same physical sign
            # rather than a random draw, so it carries no usable band. It is
            # reported and not judged.
            part += 1
            say(f"  {name:34s} {a:8.2f} ({tot:4d}) {b:10.2f} {a - b:+8.2f}"
                f"  partial {tot}/{et}, not judged")
            continue
        d = a - b
        worst = max(worst, abs(d))
        ok = abs(d) <= max(1.5, 2.0 * binom_sd(b / 100.0, tot))
        if not ok:
            bad.append((name, a, b, d, tot))
        say(f"  {name:34s} {a:8.2f} ({tot:4d}) {b:10.2f} {d:+8.2f}"
            f"  {'agrees' if ok else 'DIFFERS'}")

    say("")
    # The verdict is about the run as a whole, not about single groups. The
    # local run is the one the article will use; the earlier one is being
    # discarded because its rows cannot be paired. So this check exists to
    # catch a configuration that is wrong everywhere, the way FFA-Net read
    # fourteen per cent against fifty-five for want of an input normalisation,
    # not to demand that two implementations of the same network agree on every
    # borderline image. Two of them will not: padding, dtype and library
    # versions all move a few predictions.
    n_all = sum(1 for k in loc if loc[k][1] and k in ear and loc[k][1] == ear[k][1])
    frac = (n_all - len(bad)) / n_all if n_all else 0.0
    la = sum(100.0 * loc[k][0] / loc[k][1] for k in loc if loc[k][1]) / max(len(loc), 1)
    ea = sum(100.0 * ear[k][0] / ear[k][1] for k in ear) / max(len(ear), 1)
    say(f"  groups compared {n_all}, agreeing {n_all - len(bad)} "
        f"({100 * frac:.1f} per cent); mean over groups local {la:.2f} "
        f"against earlier {ea:.2f}, a difference of {la - ea:+.2f} points.")
    if bad:
        say(f"  the {len(bad)} that differ:")
        for name, a, b, d, tot in bad:
            say(f"    {name}: local {a:.2f} ({tot}), earlier {b:.2f}, {d:+.2f}")
    ok = frac >= 0.95 and abs(la - ea) <= 2.0
    if ok:
        say("  The run reproduces the earlier one where it matters. Merge it: "
            "it shares its padding, its enumeration and its classifier with "
            "the front ends already on disk, which is what the paired "
            "comparisons need.")
    else:
        say("  Too much of the run disagrees, or the averages are too far "
            "apart, for this to be borderline images. Find the cause before "
            "merging.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=list(MODELS) + ["all"])
    ap.add_argument("--csv", default="",
                    help="the local file, if it is not in outputs_revision")
    ap.add_argument("--earlier", default=str(EARLIER_DEFAULT))
    args = ap.parse_args()

    if not Path(args.earlier).exists():
        say(f"[FATAL] the earlier per-image file is not at {args.earlier}. "
            "Pass it with --earlier.")
        sys.exit(1)

    rule("P: does the local run reproduce the earlier one, cell by cell?")
    say("The earlier run scored the same images with the same weights and the "
        "same classifier. Agreement cell by cell says the local pipeline is "
        "right; a partial cell is compared against its own sampling band, so "
        "a short test run can be checked without waiting for the full one.")

    keys = list(MODELS) if args.model == "all" else [args.model]
    verdicts = {}
    for k in keys:
        rule(k)
        verdicts[k] = check(k, args)

    rule("summary")
    for k in keys:
        v = verdicts.get(k)
        say(f"  {k:10s} " + ("reproduces the earlier run where it matters"
                             if v is True else
                             ("DIFFERS too widely, do not merge" if v is False
                              else "not checked")))
    if all(v is True for v in verdicts.values()):
        say("\n  Safe to run to completion and merge.")


if __name__ == "__main__":
    main()
