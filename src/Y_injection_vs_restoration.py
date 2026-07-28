# -*- coding: utf-8 -*-
"""
Y_injection_vs_restoration.py
Pre-registered experiment (Evaluation Protocol, Part 14; registered before
execution).

THE QUESTION
On the blur challenges the classifier appears to gain as much from plain
additive noise as it gains from CLAHE. If an operator's benefit is nothing more
than the injection of high-frequency energy, then calling it "the matched
operator" for that degradation is wrong, and the capability-boundary table has
to say so.

DESIGN NOTE (why this is not a noise-matched control)
The first design searched, per image, for the sigma whose noisy image reaches
the operator's own high-frequency level. A pre-execution check killed it: on
small crops the flat-noise statistic is not monotone in sigma, because it is
partly driven by image structure rather than by noise, and some operators LOWER
it (gamma and CLAHE both did on a test crop). That search is unreliable, and is
replaced by a stronger, more robust comparison.

THE INJECTION ORACLE
Sweep a grid of noise levels, take the BEST one per challenge, and compare the
operator against it:

    acc(operator)   vs   max over sigma of  acc(raw + noise(sigma))

The best sigma is an oracle: no deployed system could select it. The comparison
is therefore CONSERVATIVE for the operator, which has to beat the strongest
injection plain noise can achieve.

PRE-REGISTERED DECISION RULE (fixed before running; all outcomes publishable)
    operator not above the oracle -> its benefit does not exceed plain
                                     high-frequency injection, and the boundary
                                     table must say so for that cell.
    operator above the oracle     -> it restores beyond injection, and the
                                     matched-operator claim stands.

CONTRAST CONDITIONS (they validate the test itself)
    DCP on Haze and gamma on Darkening are accepted physical inversions. If they
    do NOT beat the injection oracle, the test is unsound and no conclusion is
    drawn about the blur operators either.

DESIGN (frozen)
  challenges  : GaussianBlur, LensBlur (the suspicion), Haze, Darkening (the
                contrast conditions)
  severities  : 3, 4, 5; per-cell balanced, 400 per cell, seed 42
  operators   : gamma, clahe, stretch, dcp
  sigma grid  : 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32   (same per-image seed
                formula as T/U/V, so the levels are comparable across scripts)
  statistic   : flat-noise = 10th percentile of local 5x5 standard deviations at
                32x32, reported for corroboration only, never for matching
  aggregation : cell-averaged over (challenge, severity), the project-wide
                convention. Paired bootstrap B = 5000, seed 42, exact McNemar.

Writes: outputs_revision/Y_injection_per_image.csv, Y_injection.json
Run:    python Y_injection_vs_restoration.py       (minutes, no deep model)
"""
import argparse, csv, hashlib, json, os, sys
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
CHALLENGES = (7, 2, 12, 4)          # GaussianBlur, LensBlur, Haze, Darkening
SEV = (3, 4, 5)
OPS = ("gamma", "clahe", "stretch", "dcp")
SIGMAS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
B, SEED = 5000, 42


def flat32(img_bgr):
    """Edge-insensitive noise level at the classifier's input resolution."""
    r = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (5, 5))
    mu2 = cv2.blur(g * g, (5, 5))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    return float(np.percentile(sd[2:-2, 2:-2], 10))


def noisy(img_bgr, sigma, key):
    """Identical seed formula to T/U/V, so the levels are comparable."""
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
    ap.add_argument("--per-cell", type=int, default=400)
    ap.add_argument("--severities", default=",".join(str(v) for v in SEV),
                    help="comma-separated severities. Changing this "
                         "constitutes a NEW pre-registered experiment; the "
                         "Part 14 design is 3,4,5.")
    ap.add_argument("--out-name", default="Y_injection")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()
    sevs = tuple(int(x) for x in args.severities.split(","))

    device = "cpu"
    model = load_model(args.model, device)
    tfm = build_transform()

    samples = scan_images(Path(args.cure_root))
    rng = np.random.default_rng(SEED)
    picked = []
    for c in CHALLENGES:
        for v in sevs:
            cell = sorted([s for s in samples
                           if s["ch"] == c and s["sev"] == v],
                          key=lambda x: (x["filename"], x["occ"]))
            idx = sorted(rng.permutation(len(cell))[:args.per_cell])
            picked.extend([cell[i] for i in idx])
    print(f"[plan] {len(picked)} images "
          f"({len(CHALLENGES)} challenges x {len(sevs)} severities x "
          f"{args.per_cell} per cell); {len(OPS)} operators + "
          f"{len(SIGMAS)-1} noise levels")

    rows = []
    for n, smp in enumerate(picked):
        img = cv2.imread(str(smp["path"]))
        if img is None:
            continue
        key = f"{smp['filename']}|{smp['occ']}|{smp['ch']}|{smp['sev']}"
        rec = {"filename": smp["filename"], "occ": smp["occ"],
               "ch": smp["ch"], "sev": smp["sev"], "true": smp["true"],
               "flat_raw": round(flat32(img), 4)}
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
        rows.append(rec)
        if (n + 1) % 600 == 0:
            print(f"  {n+1}/{len(picked)}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / f"{args.out_name}_per_image.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ch = np.array([r["ch"] for r in rows])
    sv = np.array([r["sev"] for r in rows])
    tr = np.array([r["true"] for r in rows])
    NOISE_TAGS = [f"n{sg}" for sg in SIGMAS if sg > 0]
    PR = {t: np.array([r[f"pred_{t}"] for r in rows])
          for t in ["raw"] + list(OPS) + NOISE_TAGS}

    def cells(c): return [(ch == c) & (sv == v) for v in sevs]

    def acc(t, c):
        return 100 * float(np.mean([np.mean(PR[t][m] == tr[m])
                                    for m in cells(c)]))

    def boot(a, b_, c):
        rng2 = np.random.default_rng(SEED)
        A = [(PR[a][m] == tr[m]).astype(float) for m in cells(c)]
        Bb = [(PR[b_][m] == tr[m]).astype(float) for m in cells(c)]
        d = np.empty(B)
        for i in range(B):
            sa = sb = 0.0
            for x, y in zip(A, Bb):
                j = rng2.integers(0, len(x), len(x))
                sa += x[j].mean(); sb += y[j].mean()
            d[i] = 100 * (sa - sb) / len(A)
        return tuple(float(v) for v in np.percentile(d, [2.5, 97.5]))

    def mcn(a, b_, c):
        m = ch == c
        x = int(np.sum((PR[a][m] == tr[m]) & (PR[b_][m] != tr[m])))
        y = int(np.sum((PR[a][m] != tr[m]) & (PR[b_][m] == tr[m])))
        n_ = x + y
        if n_ == 0:
            return x, y, 1.0
        p = (min(1.0, 2 * sum(comb(n_, i)
                              for i in range(min(x, y) + 1)) / 2 ** n_)
             if n_ <= 1000 else erfc(abs(x - y) / sqrt(n_) / sqrt(2)))
        return x, y, float(p)

    out = {"design": {"challenges": list(CHALLENGES), "severities": list(sevs),
                      "per_cell": args.per_cell, "sigmas": list(SIGMAS)},
           "results": {}}
    print("\n" + "=" * 76)
    print("INJECTION ORACLE: can the operator beat the BEST plain-noise "
          "injection?")
    print("=" * 76)
    for c in CHALLENGES:
        name = CHALLENGE_TYPES.get(c, str(c))
        a_raw = acc("raw", c)
        scan = {sg: acc(f"n{sg}", c) for sg in SIGMAS if sg > 0}
        scan[0] = a_raw
        best_sg = max(scan, key=scan.get)
        a_orc = scan[best_sg]
        orc_tag = "raw" if best_sg == 0 else f"n{best_sg}"
        contrast = "dcp" if c == 12 else ("gamma" if c == 4 else None)
        print(f"\n  {name}   raw = {a_raw:.2f}")
        print("    noise scan: " +
              "  ".join(f"s{sg}:{scan[sg]:.1f}" for sg in sorted(scan)))
        print(f"    INJECTION ORACLE = {a_orc:.2f} at sigma = {best_sg} "
              f"(oracle over the grid; not selectable in deployment)")
        print(f"    {'operator':9s}{'flat:raw':>9s}{'->out':>8s}"
              f"{'acc(op)':>9s}{'vs raw':>9s}{'vs ORACLE':>11s}"
              f"   95% CI            verdict")
        out["results"][name] = {
            "raw_acc": round(a_raw, 2),
            "noise_scan": {str(k): round(v, 2) for k, v in sorted(scan.items())},
            "oracle_sigma": best_sg, "oracle_acc": round(a_orc, 2)}
        f_raw = float(np.median([r["flat_raw"] for r in rows if r["ch"] == c]))
        for op in OPS:
            f_op = float(np.median([r[f"flat_{op}"] for r in rows
                                    if r["ch"] == c]))
            a_op = acc(op, c)
            lo, hi = boot(op, orc_tag, c)
            _, _, p = mcn(op, orc_tag, c)
            if lo > 0:
                verdict = "RESTORES beyond injection"
            elif hi < 0:
                verdict = "WORSE than best injection"
            else:
                verdict = "NOT above injection"
            mark = "  <<< CONTRAST" if contrast == op else ""
            out["results"][name][op] = {
                "flat_out": round(f_op, 3), "acc": round(a_op, 2),
                "vs_raw": round(a_op - a_raw, 2),
                "vs_oracle": round(a_op - a_orc, 2), "lo": round(lo, 2),
                "hi": round(hi, 2), "p": p, "verdict": verdict,
                "is_contrast_condition": contrast == op}
            print(f"    {op:9s}{f_raw:9.2f}{f_op:8.2f}{a_op:9.2f}"
                  f"{a_op - a_raw:+9.2f}{a_op - a_orc:+11.2f}"
                  f"   [{lo:+6.2f},{hi:+6.2f}]  {verdict}{mark}")

    with open(Path(args.outdir) / f"{args.out_name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_name}_per_image.csv / {args.out_name}.json to "
          f"{args.outdir}")
    print("\nREADING GUIDE (pre-registered)\n"
          "  DCP on Haze and gamma on Darkening are the CONTRAST conditions. A "
          "genuine physical\n  inversion must beat the injection oracle. If it "
          "does not, the test is unsound and\n  no conclusion is drawn about "
          "the blur operators either.")


if __name__ == "__main__":
    main()
