# -*- coding: utf-8 -*-
"""
U_noise_selector.py
Closes a gap in the noise experiment (Protocol Part 12).

WHY THIS IS NEEDED
T_noise_robustness.py measured how each OPERATOR behaves under sensor noise.
It did not measure how the SELECTOR behaves, and those are different questions:
the routing statistics b, c, e are themselves computed from the noisy image, so
noise can change the routing decision (noise raises the contrast statistic c,
which may push an image out of the low-contrast branch). Any claim that the
selection policy is robust to noise therefore has to be measured, not inferred
from the operator table.

WHAT IT DOES
For every image and sigma in T's frozen subsample, it regenerates the identical
noise realisation (same per-image seed), recomputes b, c, e on the noisy image,
applies the frozen routing rule, and composes:

    VA   : the frozen 4-operator rule (branch operator as published)
    V-B  : the same rule with the low-contrast branch mapped to DCP

using the per-operator predictions already stored by T. No deep-model inference
is required, so the whole script runs in a few minutes.

It also reproduces every significance number from the noise experiment
(paired bootstrap, canonical per-comparison seed 42) with an expected-output
audit, so those numbers acquire manuscript status on the authoritative machine.

Reads:  outputs_revision/T_noise_per_image.csv, CURE-TSR images
Writes: outputs_revision/U_noise_selector.csv, U_noise_selector.json
Run:    python U_noise_selector.py
"""
import argparse, csv, hashlib, json, os, sys
from math import comb, erfc, sqrt
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import (            # noqa: E402
    compute_stats, route_decision, THRESHOLDS,
)
from Q_dcp_branch import scan_images          # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
SIGMAS = (0, 4, 8, 16)
SEV = (3, 4, 5)
CHN = {12: "Haze", 9: "Rain"}
B, SEED = 5000, 42

EXPECTED = {   # from the console output of T on the authoritative machine
    ("Haze", 0, "dcp_adair"): (+11.07, +8.80, +13.40),
    ("Haze", 4, "dcp_adair"): (+15.00, +12.73, +17.27),
    ("Haze", 8, "dcp_adair"): (+7.13, +5.13, +9.27),
    ("Haze", 16, "dcp_adair"): (-6.33, -8.40, -4.33),
    ("Rain", 0, "dcp_adair"): (+10.73, +8.20, +13.27),
    ("Rain", 4, "dcp_adair"): (+16.13, +13.67, +18.60),
    ("Rain", 8, "dcp_adair"): (+10.33, +8.00, +12.73),
    ("Rain", 16, "dcp_adair"): (-2.93, -5.40, -0.53),
}
EXPECTED_ACC = {  # operator accuracies, cell-averaged (T's console table)
    ("Haze", 0): {"passthrough": 29.53, "clahe": 49.47, "dcp": 68.07,
                  "adair": 57.00},
    ("Haze", 16): {"passthrough": 13.67, "clahe": 6.00, "dcp": 19.87,
                   "adair": 26.20},
    ("Rain", 16): {"passthrough": 26.07, "clahe": 2.13, "dcp": 21.73,
                   "adair": 24.67},
}


def noisy(img_bgr, sigma, key):
    """Identical to T_noise_robustness.noisy (same seed formula)."""
    if sigma == 0:
        return img_bgr
    h = hashlib.sha256(f"{key}|{sigma}".encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2 ** 32)
    rng = np.random.default_rng(seed)
    out = img_bgr.astype(np.float32) + rng.normal(0.0, float(sigma),
                                                  img_bgr.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--tnoise", default=str(OUT_DIR / "T_noise_per_image.csv"))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    T = list(csv.DictReader(open(args.tnoise, newline="", encoding="utf-8")))
    print(f"loaded T rows: {len(T)}")
    paths = {(s["filename"], s["occ"], s["ch"], s["sev"]): s["path"]
             for s in scan_images(Path(args.cure_root))}

    # ---- recompute routing on the noisy image ----
    rows = []
    cache = {}
    for i, r in enumerate(T):
        k = (r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
        sg = int(r["sigma"])
        if k not in cache:
            cache[k] = cv2.imread(str(paths[k]))
        img = cache[k]
        key = f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"
        b, c, e = compute_stats(noisy(img, sg, key))
        br = route_decision(b, c, e, THRESHOLDS)
        va = int(r[f"pred_{br}"])
        vb = int(r["pred_dcp"]) if br == "clahe" else va
        rows.append({**{kk: r[kk] for kk in
                        ("filename", "occ", "ch", "sev", "sigma", "true")},
                     "b": round(b, 5), "c": round(c, 5), "e": round(e, 5),
                     "branch": br, "pred_va": va, "pred_vb": vb,
                     "pred_dcp": int(r["pred_dcp"]),
                     "pred_adair": int(r["pred_adair"]),
                     "pred_clahe": int(r["pred_clahe"]),
                     "pred_passthrough": int(r["pred_passthrough"])})
        if (i + 1) % 3000 == 0:
            print(f"  routed {i+1}/{len(T)}")
    os.makedirs(args.outdir, exist_ok=True)
    outp = Path(args.outdir) / "U_noise_selector.csv"
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {outp}")

    ch = np.array([int(r["ch"]) for r in rows])
    sv = np.array([int(r["sev"]) for r in rows])
    sg = np.array([int(r["sigma"]) for r in rows])
    tr = np.array([int(r["true"]) for r in rows])
    br = np.array([r["branch"] for r in rows])
    PR = {o: np.array([int(r[f"pred_{o}"]) for r in rows])
          for o in ("va", "vb", "dcp", "adair", "clahe", "passthrough")}

    def cm(c, s): return [(ch == c) & (sv == v) & (sg == s) for v in SEV]
    def acc(o, c, s):
        return 100 * float(np.mean([np.mean(PR[o][m] == tr[m])
                                    for m in cm(c, s)]))
    def boot(a, b_, c, s):
        rng = np.random.default_rng(SEED)
        A = [(PR[a][m] == tr[m]).astype(float) for m in cm(c, s)]
        Bb = [(PR[b_][m] == tr[m]).astype(float) for m in cm(c, s)]
        d = np.empty(B)
        for i in range(B):
            sa = sb = 0.0
            for x, y in zip(A, Bb):
                j = rng.integers(0, len(x), len(x))
                sa += x[j].mean(); sb += y[j].mean()
            d[i] = 100 * (sa - sb) / len(A)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi)
    def mcn(a, b_, c, s):
        m = (ch == c) & (sg == s)
        x = int(np.sum((PR[a][m] == tr[m]) & (PR[b_][m] != tr[m])))
        y = int(np.sum((PR[a][m] != tr[m]) & (PR[b_][m] == tr[m])))
        n = x + y
        if n == 0:
            return x, y, 1.0
        p = (min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n)
             if n <= 1000 else erfc(abs(x - y) / sqrt(n) / sqrt(2)))
        return x, y, float(p)

    out = {"routing_shift": {}, "selector": {}, "operator_check": {}}

    print("\n=== 1. Does noise change the ROUTING itself? ===")
    print(f"{'challenge':10s}{'sigma':>6s}" +
          "".join(f"{b:>13s}" for b in
                  ("gamma", "clahe", "stretch", "passthrough")))
    for c in (12, 9):
        out["routing_shift"][CHN[c]] = {}
        for s in SIGMAS:
            m = (ch == c) & (sg == s)
            sh = {b_: 100 * float(np.mean(br[m] == b_))
                  for b_ in ("gamma", "clahe", "stretch", "passthrough")}
            out["routing_shift"][CHN[c]][str(s)] = sh
            print(f"{CHN[c]:10s}{s:6d}" +
                  "".join(f"{sh[b_]:12.1f}%" for b_ in
                          ("gamma", "clahe", "stretch", "passthrough")))

    print("\n=== 2. The SELECTOR under noise (this is C2's actual claim) ===")
    for c in (12, 9):
        print(f"\n  {CHN[c]}:")
        print(f"    {'sigma':>5s}{'VA':>8s}{'V-B':>8s}{'DCP':>8s}"
              f"{'AdaIR':>8s}   V-B - VA (95% CI)        V-B - AdaIR")
        out["selector"][CHN[c]] = {}
        for s in SIGMAS:
            a_va, a_vb = acc("va", c, s), acc("vb", c, s)
            a_dcp, a_ad = acc("dcp", c, s), acc("adair", c, s)
            lo1, hi1 = boot("vb", "va", c, s)
            lo2, hi2 = boot("vb", "adair", c, s)
            x1, y1, p1 = mcn("vb", "va", c, s)
            out["selector"][CHN[c]][str(s)] = {
                "va": a_va, "vb": a_vb, "dcp": a_dcp, "adair": a_ad,
                "vb_minus_va": [round(a_vb - a_va, 2), round(lo1, 2),
                                 round(hi1, 2), p1],
                "vb_minus_adair": [round(a_vb - a_ad, 2), round(lo2, 2),
                                    round(hi2, 2)]}
            print(f"    {s:5d}{a_va:8.2f}{a_vb:8.2f}{a_dcp:8.2f}{a_ad:8.2f}"
                  f"   {a_vb-a_va:+6.2f} [{lo1:+6.2f},{hi1:+6.2f}]"
                  f"   {a_vb-a_ad:+6.2f} [{lo2:+6.2f},{hi2:+6.2f}]")

    print("\n=== 3. AUDIT: reproduce T's operator numbers ===")
    ok = True
    def chk(n, got, exp, tol):
        nonlocal ok
        p = abs(got - exp) <= tol
        ok &= p
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {got:.2f} vs {exp}")
    for (cn, s), d in EXPECTED_ACC.items():
        c = 12 if cn == "Haze" else 9
        for o, v in d.items():
            chk(f"{cn} sigma{s} {o}", acc(o, c, s), v, 0.02)
    for (cn, s, _), (pt, lo, hi) in EXPECTED.items():
        c = 12 if cn == "Haze" else 9
        g = acc("dcp", c, s) - acc("adair", c, s)
        l_, h_ = boot("dcp", "adair", c, s)
        chk(f"{cn} sigma{s} DCP-AdaIR", g, pt, 0.02)
        chk(f"{cn} sigma{s} lo", l_, lo, 0.03)
        chk(f"{cn} sigma{s} hi", h_, hi, 0.03)
    out["operator_check"] = {"passed": bool(ok)}

    with open(Path(args.outdir) / "U_noise_selector.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nALL CHECKS PASSED -- the noise numbers and the selector-under-"
          "noise numbers may be quoted." if ok else
          "\n*** MISMATCH: send me this output. ***")


if __name__ == "__main__":
    main()
