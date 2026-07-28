# -*- coding: utf-8 -*-
"""
Z_injection_definitive.py
Pre-registered experiment (Evaluation Protocol, Part 16). Supersedes the
sampling of Parts 14 and 15.

WHY THIS SUPERSEDES PARTS 14 AND 15
Part 15 (mild severities) failed its validity check, and the diagnosis is
disclosed here in full.

  (a) THE CONTRAST CONDITION WAS MIS-SPECIFIED. gamma on Darkening was used to
      validate the test. In the frozen cache its restoration effect at
      severities 1 and 2 is only +0.67 accuracy points, while the injection
      oracle there is worth +3.75. An operator that restores less than plain
      noise injects SHOULD lose to it; the test was working correctly and the
      validity check was simply not fit for purpose at that severity.

  (b) THE SAMPLING NOISE WAS OF THE SAME SIZE AS THE EFFECT. At 400 images per
      cell the standard error of an accuracy is about 1.5 points, which is
      larger than gamma's +0.67. The subsample duly reported -0.62 where the
      full cache gives +0.67, a pure sampling artefact confirmed by a
      bit-for-bit audit of the pipeline against the caches.

  (c) FULL DISCLOSURE. The contrast-selection criterion below was introduced
      AFTER Part 15 had run, and the pairs it selects had already been observed
      to pass. The criterion is computed from the frozen main cache alone and
      never from the injection test, but the reader is told that the adjustment
      was post hoc.

THE TWO REPAIRS
  1. NO SUBSAMPLING. Every image of every cell is used (1352 per cell), so the
     sampling noise is zero and the operator accuracies must reproduce the
     cached values exactly. This is verified as a determinism audit.
  2. A PRE-SPECIFIED CONTRAST CRITERION. A (challenge, operator) pair may serve
     as a validity check at a given severity only if, in the frozen cache, the
     operator exceeds passthrough by at least 5 accuracy points at that
     severity. The criterion is computed and PRINTED BEFORE any verdict. Only
     the two non-blur challenges may supply contrasts, because the blur
     challenges are what is under test. If no pair qualifies at a severity, the
     test cannot be validated there and no verdict is issued for it.

THE TEST ITSELF (unchanged from Part 14)
    acc(operator)   vs   max over sigma of  acc(raw + noise(sigma))
The best sigma is an oracle no deployed system could select, so the comparison
is conservative for the operator.

    operator not above the oracle -> its benefit does not exceed plain
                                     high-frequency injection.
    operator above the oracle     -> it restores beyond injection.

DESIGN (frozen)
  challenges : GaussianBlur, LensBlur (under test); Haze, Darkening (contrasts)
  severities : 1, 2, 3, 4, 5, every image of every cell
  operators  : gamma, clahe, stretch, dcp
  sigma grid : 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 (T/U/V seed formula)
  statistics : paired bootstrap B = 5000 seed 42, and exact McNemar, WITHIN each
               (challenge, severity) cell; no cell averaging, because the verdict
               is issued per cell
  audit      : the operator predictions must equal the cached ones bit for bit

Writes: outputs_revision/Z_injection_per_image.csv, Z_injection.json
Run:    python Z_injection_definitive.py           (~10-20 min, no deep model)
        python Z_injection_definitive.py --resume
"""
import argparse, csv, hashlib, json, os, sys, time
from math import comb, erfc, sqrt
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import (            # noqa: E402
    apply_gamma, apply_clahe, apply_stretch, load_model, build_transform,
    INPUT_SIZE, CHALLENGE_TYPES,
)
from J_local_deep_eval import classify_batch  # noqa: E402
from Q_dcp_branch import dcp_enhance, scan_images   # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
CONTRAST_SRC = (12, 4)              # Haze, Darkening: they supply the
                                    # validity check and are themselves tested
ALL_CHALLENGES = tuple(range(1, 13))
SEVERITIES = (1, 2, 3, 4, 5)
OPS = ("gamma", "clahe", "stretch", "dcp")
SIGMAS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
CONTRAST_MIN_GAIN = 5.0             # pre-specified, frozen
B, SEED = 5000, 42


def flat32(img_bgr):
    r = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (5, 5))
    mu2 = cv2.blur(g * g, (5, 5))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    return float(np.percentile(sd[2:-2, 2:-2], 10))


def noisy(img_bgr, sigma, key):
    if sigma == 0:
        return img_bgr
    h = hashlib.sha256(f"{key}|{sigma}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little") % (2 ** 32))
    return np.clip(img_bgr.astype(np.float32) +
                   rng.normal(0.0, float(sigma), img_bgr.shape),
                   0, 255).astype(np.uint8)


def apply_op(img_bgr, op):
    if op == "gamma":
        return apply_gamma(img_bgr)
    if op == "clahe":
        return apply_clahe(img_bgr)
    if op == "stretch":
        return apply_stretch(img_bgr)
    if op == "dcp":
        return dcp_enhance(img_bgr)[0]
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root",
                    default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model",
                    default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp-cache", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--challenges", default="7,2,12,4",
                    help="comma-separated challenge ids; 'all' runs all 12. "
                         "The Part 16 design is 7,2,12,4; running a different "
                         "set is a NEW pre-registered experiment (Part 17).")
    ap.add_argument("--out-name", default="Z_injection")
    args = ap.parse_args()
    challenges = (ALL_CHALLENGES if args.challenges.strip().lower() == "all"
                  else tuple(int(x) for x in args.challenges.split(",")))
    under_test = challenges

    # ---------- the contrast criterion, from the frozen cache ONLY ----------
    Dc = {}
    with open(args.dcp_cache, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Dc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = int(r["pred_dcp"])
    Mc = {}
    with open(args.merged, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Mc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = r
    CACHE_COL = {"raw": "pred_passthrough", "gamma": "pred_gamma",
                 "clahe": "pred_clahe", "stretch": "pred_stretch"}

    def cache_acc(c, v, tag):
        ks = [k for k in Mc if k[2] == c and k[3] == v]
        t = np.array([int(Mc[k]["true"]) for k in ks])
        if tag == "dcp":
            p = np.array([Dc[k] for k in ks])
        else:
            p = np.array([int(Mc[k][CACHE_COL[tag]]) for k in ks])
        return 100 * float(np.mean(p == t))

    print("=" * 78)
    print("CONTRAST CRITERION (from the frozen cache only; printed BEFORE any "
          "verdict)")
    print(f"  a pair qualifies if the operator beats passthrough by >= "
          f"{CONTRAST_MIN_GAIN} points at that severity")
    print("=" * 78)
    contrasts = {v: [] for v in SEVERITIES}
    for v in SEVERITIES:
        line = []
        for c in CONTRAST_SRC:
            base = cache_acc(c, v, "raw")
            for op in OPS:
                d = cache_acc(c, v, op) - base
                if d >= CONTRAST_MIN_GAIN:
                    contrasts[v].append((c, op))
                    line.append(f"{CHALLENGE_TYPES.get(c, c)}/{op} (+{d:.1f})")
        print(f"  severity {v}: " +
              (", ".join(line) if line else
               "NO qualifying contrast -> no verdict may be issued"))
    print()

    # ---------- run ----------
    device = "cpu"
    model = load_model(args.model, device)
    tfm = build_transform()
    samples = [s for s in scan_images(Path(args.cure_root))
               if s["ch"] in challenges and s["sev"] in SEVERITIES]
    samples.sort(key=lambda s: (s["ch"], s["sev"], s["filename"], s["occ"]))
    print(f"[plan] {len(samples)} images (every image of every cell; "
          f"no subsampling)")

    out_csv = Path(args.outdir) / f"{args.out_name}_per_image.csv"
    done = set()
    if args.resume and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["filename"], int(r["occ"]), int(r["ch"]),
                          int(r["sev"])))
        print(f"[resume] {len(done)} rows cached")
    os.makedirs(args.outdir, exist_ok=True)

    fields = (["filename", "occ", "ch", "sev", "true", "flat_raw"]
              + [f"flat_{o}" for o in OPS]
              + [f"flat_n{s}" for s in SIGMAS if s]
              + ["pred_raw"] + [f"pred_{o}" for o in OPS]
              + [f"pred_n{s}" for s in SIGMAS if s])
    mode = "a" if (args.resume and out_csv.exists()) else "w"
    f_out = open(out_csv, mode, newline="", encoding="utf-8")
    w = csv.DictWriter(f_out, fieldnames=fields)
    if mode == "w":
        w.writeheader()

    t0, n = time.time(), 0
    for smp in samples:
        k = (smp["filename"], smp["occ"], smp["ch"], smp["sev"])
        if k in done:
            continue
        img = cv2.imread(str(smp["path"]))
        if img is None:
            continue
        key = f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"
        rec = {"filename": k[0], "occ": k[1], "ch": k[2], "sev": k[3],
               "true": smp["true"], "flat_raw": round(flat32(img), 4)}
        batch, tags = [img], ["raw"]
        for op in OPS:
            enh = apply_op(img, op)
            rec[f"flat_{op}"] = round(flat32(enh), 4)
            batch.append(enh); tags.append(op)
        for sg in SIGMAS:
            if sg == 0:
                continue
            nz = noisy(img, sg, key)
            rec[f"flat_n{sg}"] = round(flat32(nz), 4)
            batch.append(nz); tags.append(f"n{sg}")
        p, _ = classify_batch(model, batch, tfm, device)
        for t, pv in zip(tags, p):
            rec[f"pred_{t}"] = int(pv)
        w.writerow(rec)
        n += 1
        if n % 2000 == 0:
            r = n / max(time.time() - t0, 1e-9)
            f_out.flush()
            print(f"  {n}/{len(samples)-len(done)}  {r:.1f} img/s  ETA "
                  f"{(len(samples)-len(done)-n)/max(r,1e-9)/60:.0f} min")
    f_out.close()
    print(f"[done] {n} images in {(time.time()-t0)/60:.1f} min")

    rows = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    ch = np.array([int(r["ch"]) for r in rows])
    sv = np.array([int(r["sev"]) for r in rows])
    tr = np.array([int(r["true"]) for r in rows])
    TAGS = ["raw"] + list(OPS) + [f"n{s}" for s in SIGMAS if s]
    PR = {t: np.array([int(r[f"pred_{t}"]) for r in rows]) for t in TAGS}

    # ---------- determinism audit ----------
    print("\n=== DETERMINISM AUDIT (full cells: must equal the caches "
          "bit for bit) ===")
    ok_all = True
    for tag in ["raw"] + list(OPS):
        good = tot = 0
        for i, r in enumerate(rows):
            k = (r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
            if k not in Mc or k not in Dc:
                continue
            tot += 1
            exp = Dc[k] if tag == "dcp" else int(Mc[k][CACHE_COL[tag]])
            good += int(int(r[f"pred_{tag}"]) == exp)
        ok = (good == tot)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag:9s} {good}/{tot} "
              f"({100*good/max(tot,1):.2f}%)")
    if not ok_all:
        print("\n*** AUDIT FAILED: the pipeline differs from the cached runs. "
              "Do NOT use these numbers. ***")
        return
    print("  -> pipeline identical to the main experiment")

    def acc(t, c, v):
        m = (ch == c) & (sv == v)
        return 100 * float(np.mean(PR[t][m] == tr[m]))

    def boot(a, b_, c, v):
        rng = np.random.default_rng(SEED)
        m = (ch == c) & (sv == v)
        A = (PR[a][m] == tr[m]).astype(float)
        Bb = (PR[b_][m] == tr[m]).astype(float)
        idx = rng.integers(0, len(A), (B, len(A)))
        d = 100 * (A[idx].mean(axis=1) - Bb[idx].mean(axis=1))
        return tuple(float(x) for x in np.percentile(d, [2.5, 97.5]))

    def mcn(a, b_, c, v):
        m = (ch == c) & (sv == v)
        x = int(np.sum((PR[a][m] == tr[m]) & (PR[b_][m] != tr[m])))
        y = int(np.sum((PR[a][m] != tr[m]) & (PR[b_][m] == tr[m])))
        n_ = x + y
        if n_ == 0:
            return 1.0
        return float(min(1.0, 2 * sum(comb(n_, i)
                                      for i in range(min(x, y) + 1)) / 2 ** n_)
                     if n_ <= 1000
                     else erfc(abs(x - y) / sqrt(n_) / sqrt(2)))

    out = {"design": {"challenges": list(challenges),
                      "severities": list(SEVERITIES),
                      "sigmas": list(SIGMAS),
                      "contrast_min_gain": CONTRAST_MIN_GAIN,
                      "subsampling": "none (all images of every cell)"},
           "contrasts": {str(v): [[CHALLENGE_TYPES.get(c, c), o]
                                  for c, o in contrasts[v]] for v in SEVERITIES},
           "cells": {}}

    print("\n" + "=" * 78)
    print("INJECTION ORACLE, per (challenge, severity), on full cells")
    print("=" * 78)
    for c in challenges:
        name = CHALLENGE_TYPES.get(c, str(c))
        under = c in under_test
        print(f"\n  {name} {'(UNDER TEST)' if under else '(contrast source)'}")
        for v in SEVERITIES:
            a_raw = acc("raw", c, v)
            scan = {s: acc(f"n{s}", c, v) for s in SIGMAS if s}
            scan[0] = a_raw
            bs = max(scan, key=scan.get)
            a_orc, otag = scan[bs], ("raw" if bs == 0 else f"n{bs}")
            valid = len(contrasts[v]) > 0
            print(f"    sev{v}: raw {a_raw:5.2f} | injection oracle "
                  f"{a_orc:5.2f} (sigma={bs:2d}, +{a_orc-a_raw:.2f})"
                  f"{'' if valid else '   [NO VALID CONTRAST -> no verdict]'}")
            cell = {"raw": round(a_raw, 2), "oracle_sigma": bs,
                    "oracle_acc": round(a_orc, 2),
                    "validated": bool(valid), "ops": {}}
            for op in OPS:
                a_op = acc(op, c, v)
                lo, hi = boot(op, otag, c, v)
                p = mcn(op, otag, c, v)
                is_con = (c, op) in contrasts[v]
                if not valid:
                    verdict = "no verdict (test not validated at this severity)"
                elif lo > 0:
                    verdict = "RESTORES beyond injection"
                elif hi < 0:
                    verdict = "WORSE than best injection"
                else:
                    verdict = "NOT above injection"
                cell["ops"][op] = {"acc": round(a_op, 2),
                                    "vs_raw": round(a_op - a_raw, 2),
                                    "vs_oracle": round(a_op - a_orc, 2),
                                    "lo": round(lo, 2), "hi": round(hi, 2),
                                    "p": p, "verdict": verdict,
                                    "is_contrast": is_con}
                flag = "  <<< CONTRAST" if is_con else ""
                if under or is_con:
                    print(f"        {op:8s} {a_op:6.2f}  vs oracle "
                          f"{a_op-a_orc:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  "
                          f"{verdict}{flag}")
            out["cells"][f"{name}_sev{v}"] = cell

    with open(Path(args.outdir) / f"{args.out_name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_name}_per_image.csv / {args.out_name}.json "
          f"to {args.outdir}")
    print("\nREADING GUIDE\n"
          "  A verdict is issued only at severities where a qualifying contrast "
          "exists, and the\n  contrast itself must show RESTORES. Where it does "
          "not, the test is not validated\n  there and the cell is reported as "
          "open.")


if __name__ == "__main__":
    main()
