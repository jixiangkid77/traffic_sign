# -*- coding: utf-8 -*-
"""
AB_resolution_sensitivity.py
Closes the strongest available objection to the paper's headline finding.

THE OBJECTION
ZA showed that on the blur challenges the learned restoration models never beat
an oracle-selected noise injection, while CLAHE and DCP do. A reviewer will
immediately answer: the learned models were applied to a crop of about 28 by 28
pixels, whereas they were trained on patches of 128 by 128 or larger. Of course
they fail. Upsample the crop into their native regime, restore, and come back.

That objection has to be answered with data, not with an argument. If AdaIR's
blur performance improves substantially when it is given an input of the size it
was trained for, then the finding is an artefact of the protocol and the
manuscript must say so. If it does not improve, the finding is robust and the
strongest line of attack is closed.

THE TEST
The frozen protocol is unchanged; this is a sensitivity analysis alongside it.

    native (frozen)  : crop -> AdaIR -> resize 32 -> classify
    upsampled        : crop -> bicubic to 128 -> AdaIR -> resize 32 -> classify

Both are compared against the injection oracle for the same images, taken from
Z's noise scan, so the comparison is paired throughout.

A CONTROL IS INCLUDED, and it is what makes the test interpretable. Haze is a
challenge on which AdaIR demonstrably DOES restore in the native protocol
(+33.43 above the injection oracle at severity 4). If upsampling lifts blur but
not haze, the blur failure was resolution-bound. If it lifts both, or neither,
the interpretation changes accordingly, and each outcome is stated in advance.

PRE-REGISTERED READING (fixed before running)
  blur improves and now beats the oracle -> the native-resolution finding is an
        artefact of the protocol; the manuscript reports the upsampled result
        and withdraws the claim that learned models cannot restore blur.
  blur does not improve                  -> the finding is robust to the
        objection, and the manuscript says the failure is not a resolution
        artefact.
  haze degrades under upsampling         -> upsampling itself is harmful and the
        comparison is confounded; no conclusion is drawn.

DESIGN (frozen)
  challenges : GaussianBlur, LensBlur (the failure), Haze (the control)
  severities : 3, 4, 5
  per cell   : 150 images, seed 42, balanced
  models     : AdaIR (the strongest learned baseline)
  upsample   : bicubic to 128 x 128 (AdaIR's training patch size)
  audit      : the native-protocol predictions must equal the cached ones

Reads:  merged_per_image.csv (cached native AdaIR), Z's per-image csv (the
        injection oracle), CURE-TSR images
Writes: outputs_revision/AB_resolution.json
Run:    python AB_resolution_sensitivity.py            (~40-60 min)
        python AB_resolution_sensitivity.py --per-cell 60    (fast probe)
"""
import argparse, csv, json, os, sys, time
from math import comb, erfc, sqrt
from pathlib import Path

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import (            # noqa: E402
    load_model, build_transform, CHALLENGE_TYPES,
)
from J_local_deep_eval import (               # noqa: E402
    load_adair, enhance_batch, classify_batch,
)
from Q_dcp_branch import scan_images          # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
CHALLENGES = (7, 2, 12)             # GaussianBlur, LensBlur, Haze (control)
SEV = (3, 4, 5)
UPSAMPLE = 128                      # AdaIR's training patch size
B, SEED = 5000, 42


def mcn(a, b, t):
    x = int(np.sum((a == t) & (b != t)))
    y = int(np.sum((a != t) & (b == t)))
    n = x + y
    if n == 0:
        return 1.0
    if n <= 1000:
        return float(min(1.0, 2 * sum(comb(n, i)
                                      for i in range(min(x, y) + 1)) / 2 ** n))
    return float(erfc(abs(x - y) / sqrt(n) / sqrt(2)))


def boot(a, b, t):
    rng = np.random.default_rng(SEED)
    A = (a == t).astype(float)
    Bb = (b == t).astype(float)
    idx = rng.integers(0, len(A), (B, len(A)))
    d = 100 * (A[idx].mean(axis=1) - Bb[idx].mean(axis=1))
    return tuple(float(x) for x in np.percentile(d, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root",
                    default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model",
                    default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--adair-weight",
                    default=str(PROJECT_ROOT / "models" / "adair5d.ckpt"))
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--zfile", default=str(OUT_DIR / "Z_injection_per_image.csv"))
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    Mc = {}
    with open(args.merged, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Mc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = r
    Zc = {}
    with open(args.zfile, newline="", encoding="utf-8") as f:
        zr = csv.DictReader(f)
        sig_cols = [c for c in zr.fieldnames if c.startswith("pred_n")]
        for r in zr:
            Zc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = r
    print(f"loaded merged={len(Mc)} Z={len(Zc)}  noise levels in Z: "
          f"{[c[6:] for c in sig_cols]}")

    device = "cpu"
    model = load_model(args.model, device)
    tfm = build_transform()
    adair, _ = load_adair(device, args.adair_weight)

    samples = scan_images(Path(args.cure_root))
    rng = np.random.default_rng(SEED)
    picked = []
    for c in CHALLENGES:
        for v in SEV:
            cell = sorted([s for s in samples
                           if s["ch"] == c and s["sev"] == v],
                          key=lambda x: (x["filename"], x["occ"]))
            for i in sorted(rng.permutation(len(cell))[:args.per_cell]):
                k = (cell[i]["filename"], cell[i]["occ"], c, v)
                if k in Mc and k in Zc:
                    picked.append(cell[i])
    print(f"[plan] {len(picked)} images; AdaIR is run once per image at "
          f"{UPSAMPLE}x{UPSAMPLE} and once at native size")
    print(f"[plan] the native run is only for the audit; the cached native "
          f"prediction is the reference")

    rows = []
    t0 = time.time()
    for i in range(0, len(picked), args.batch):
        chunk = picked[i:i + args.batch]
        imgs, metas = [], []
        for s in chunk:
            im = cv2.imread(str(s["path"]))
            if im is not None:
                imgs.append(im); metas.append(s)
        if not imgs:
            continue
        # native: group by size (they may differ)
        nat = []
        for im in imgs:
            e = enhance_batch(adair, [im], device)[0]
            nat.append(e)
        p_nat, _ = classify_batch(model, nat, tfm, device)
        # upsampled
        ups = []
        for im in imgs:
            big = cv2.resize(im, (UPSAMPLE, UPSAMPLE),
                             interpolation=cv2.INTER_CUBIC)
            e = enhance_batch(adair, [big], device)[0]
            ups.append(e)
        p_ups, _ = classify_batch(model, ups, tfm, device)
        for j, s in enumerate(metas):
            k = (s["filename"], s["occ"], s["ch"], s["sev"])
            rows.append({"filename": k[0], "occ": k[1], "ch": k[2],
                         "sev": k[3], "true": s["true"],
                         "adair_native": int(p_nat[j]),
                         "adair_upsampled": int(p_ups[j]),
                         "adair_cached": int(Mc[k]["pred_adair"])})
        if len(rows) % 150 < args.batch:
            r = len(rows) / max(time.time() - t0, 1e-9)
            print(f"  {len(rows)}/{len(picked)}  {r:.2f} img/s  ETA "
                  f"{(len(picked)-len(rows))/max(r,1e-9)/60:.0f} min")
    print(f"[done] {len(rows)} images in {(time.time()-t0)/60:.1f} min")

    ch = np.array([r["ch"] for r in rows])
    sv = np.array([r["sev"] for r in rows])
    tr = np.array([r["true"] for r in rows])
    nat = np.array([r["adair_native"] for r in rows])
    ups = np.array([r["adair_upsampled"] for r in rows])
    cac = np.array([r["adair_cached"] for r in rows])

    print("\n=== AUDIT: the native run must reproduce the cache ===")
    agree = float(np.mean(nat == cac))
    verdict = ("PASS" if agree == 1.0 else
               ("WARN" if agree >= 0.995 else "FAIL"))
    print(f"  [{verdict}] {int(agree*len(nat))}/{len(nat)} "
          f"({100*agree:.2f}%) identical to merged_per_image.csv")
    if agree < 0.995:
        print("\n*** AUDIT FAILED: the native path differs from the cached "
              "run. Do NOT use these numbers. ***")
        return

    # the injection oracle for these same images, taken from Z
    sig_levels = sorted(int(c[6:]) for c in sig_cols)
    out = {"design": {"challenges": list(CHALLENGES), "severities": list(SEV),
                      "per_cell": args.per_cell, "upsample": UPSAMPLE},
           "cells": {}}
    print("\n" + "=" * 78)
    print("IS THE BLUR FAILURE A RESOLUTION ARTEFACT?")
    print("=" * 78)
    print(f"  {'cell':18s}{'raw':>7s}{'oracle':>8s}{'AdaIR nat':>11s}"
          f"{'AdaIR up':>10s}{'up - nat':>10s}   95% CI          "
          f"{'up - oracle':>12s}")
    for c in CHALLENGES:
        for v in SEV:
            m = (ch == c) & (sv == v)
            if m.sum() == 0:
                continue
            ks = [(rows[i]["filename"], rows[i]["occ"], c, v)
                  for i in np.where(m)[0]]
            t = tr[m]
            raw = 100 * float(np.mean(
                np.array([int(Zc[k]["pred_raw"]) for k in ks]) == t))
            best_a, best_s = raw, 0
            for sg in sig_levels:
                a = 100 * float(np.mean(
                    np.array([int(Zc[k][f"pred_n{sg}"]) for k in ks]) == t))
                if a > best_a:
                    best_a, best_s = a, sg
            orc = (np.array([int(Zc[k]["pred_raw"]) for k in ks])
                   if best_s == 0 else
                   np.array([int(Zc[k][f"pred_n{best_s}"]) for k in ks]))
            a_nat = 100 * float(np.mean(nat[m] == t))
            a_ups = 100 * float(np.mean(ups[m] == t))
            lo, hi = boot(ups[m], nat[m], t)
            p = mcn(ups[m], nat[m], t)
            lo2, hi2 = boot(ups[m], orc, t)
            beats = "BEATS ORACLE" if lo2 > 0 else (
                "below oracle" if hi2 < 0 else "ties oracle")
            name = f"{CHALLENGE_TYPES.get(c, c)}_s{v}"
            out["cells"][name] = {
                "raw": round(raw, 2), "oracle_sigma": best_s,
                "oracle": round(best_a, 2),
                "adair_native": round(a_nat, 2),
                "adair_upsampled": round(a_ups, 2),
                "up_minus_native": [round(a_ups - a_nat, 2), round(lo, 2),
                                     round(hi, 2), p],
                "up_minus_oracle": [round(a_ups - best_a, 2), round(lo2, 2),
                                     round(hi2, 2)],
                "verdict_vs_oracle": beats}
            print(f"  {name:18s}{raw:7.2f}{best_a:8.2f}{a_nat:11.2f}"
                  f"{a_ups:10.2f}{a_ups-a_nat:+10.2f}   "
                  f"[{lo:+6.2f},{hi:+6.2f}]  {a_ups-best_a:+8.2f} {beats}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "AB_resolution.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {Path(args.outdir) / 'AB_resolution.json'}")
    print("\nREADING GUIDE (pre-registered)\n"
          "  Haze is the control: AdaIR restores it at native size, so if "
          "upsampling degrades\n  haze, upsampling itself is harmful and "
          "nothing can be concluded. If upsampling\n  lifts blur above the "
          "injection oracle, the headline finding is a protocol artefact\n  "
          "and must be withdrawn. If it does not, the finding survives the "
          "strongest objection\n  available to a reviewer.")


if __name__ == "__main__":
    main()
