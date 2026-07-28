# -*- coding: utf-8 -*-
"""
V_robust_routing.py
Pre-registered experiment (Evaluation Protocol, Part 13; registered
2026-07-09, before execution).

THE PROBLEM (established by U_noise_selector.py)
The routing statistics are computed from the observed image, so sensor noise
corrupts the routing itself. Additive noise of standard deviation s inflates the
contrast statistic, because variances add: std(signal + noise)^2 = std_signal^2
+ s^2. Measured consequence: the low-contrast branch share on Haze falls from
74.3% (clean) to 42.1% (sigma = 16), and on Rain from 39.5% to 4.9%. The edge
statistic is inflated in the same way by spurious Canny responses, which is why
the stretch branch empties completely. The published rule therefore loses its
haze benefit under noise and falls below doing nothing.

THE FIX, DERIVED RATHER THAN TUNED
Variance additivity is exactly invertible. Estimate the noise level with the
parameter-free estimator of Immerkaer (1996), subtract only the EXCESS above the
sensor's own noise floor (so that clean images are untouched), and take the
square root:

    sigma_hat  = Immerkaer(gray)                       [parameter free]
    var_added  = max(sigma_hat^2 - sigma_floor^2, 0)   [variances add;
                                                        sigma_floor from the
                                                        clean split, label free]
    c_robust   = sqrt(max(c^2 - var_added/128^2, 0))

At zero added noise var_added is zero and c_robust is identically c, so the
published thresholds T1..T4 remain valid by construction. No new tunable
parameter is introduced.

TWO PRE-SPECIFIED VARIANTS
  V-C1  analytic compensation of the contrast statistic only (above).
        Identity at sigma = 0 by construction; does not repair the edge
        statistic, so the stretch branch remains noise fragile. Disclosed.
  V-C2  a 3x3 median prefilter before computing all three statistics.
        Repairs the contrast and the edge statistic together, but is not
        identity at sigma = 0, so its threshold compatibility must be measured.

2x2 FACTORIAL (isolates the two fixes)
  routing in {published, robust}  x  low-contrast operator in {CLAHE, DCP}
  The published rule with CLAHE is VA; the published rule with DCP is V-B.

PRE-REGISTERED CRITERIA
  Threshold compatibility: at sigma = 0 a routing variant must agree with the
  published routing on at least 95% of images; a variant that fails is reported
  as rejected and its accuracy is not promoted to a claim.
  Primary endpoint: accuracy of the robust-routing selector against V-B at
  sigma = 4 and sigma = 8.
  Both directions are informative. If the correction does not restore the branch
  shares, the manuscript reports the routing statistics as intrinsically noise
  fragile and the limitation stands.

Reads:  outputs_revision/T_noise_per_image.csv, CURE-TSR images
Writes: outputs_revision/V_robust_routing.csv, V_robust_routing.json
Run:    python V_robust_routing.py            (a few minutes, no deep inference)
"""
import argparse, csv, hashlib, json, os, sys
from math import comb, erfc, pi, sqrt
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import route_decision, THRESHOLDS   # noqa: E402
from Q_dcp_branch import scan_images                          # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
SIGMAS = (0, 4, 8, 16)
SEV = (3, 4, 5)
CHN = {12: "Haze", 9: "Rain"}
B, SEED = 5000, 42
AGREE_MIN = 0.95          # pre-registered threshold-compatibility criterion

IMMERKAER_K = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])


def immerkaer_sigma(gray):
    """Fast noise standard deviation estimate (Immerkaer 1996), in [0,255]."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float64), -1, IMMERKAER_K,
                        borderType=cv2.BORDER_REPLICATE)[1:-1, 1:-1]
    return float(np.sum(np.abs(conv)) * sqrt(pi / 2.0) /
                 (6.0 * (w - 2) * (h - 2)))


def stats_from_gray(gray):
    b = float(gray.mean()) / 255.0
    c = float(gray.std()) / 128.0
    e = float(cv2.Canny(gray, 50, 150).mean()) / 255.0
    return b, c, e


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

    samples = scan_images(Path(args.cure_root))
    paths = {(s["filename"], s["occ"], s["ch"], s["sev"]): s["path"]
             for s in samples}

    # ---- sigma_floor: label-free, from the clean (ChallengeFree) split ----
    clean = [s for s in samples if s["ch"] == 0]
    fl = []
    for s in clean:
        im = cv2.imread(str(s["path"]))
        if im is None:
            continue
        fl.append(immerkaer_sigma(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))
    sigma_floor = float(np.median(fl))
    print(f"[floor] Immerkaer noise floor of the clean split: "
          f"{sigma_floor:.3f}/255  (n={len(fl)}; label-free, frozen)")

    T = list(csv.DictReader(open(args.tnoise, newline="", encoding="utf-8")))
    print(f"[data] T rows: {len(T)}")

    rows, cache = [], {}
    for i, r in enumerate(T):
        k = (r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
        sg = int(r["sigma"])
        if k not in cache:
            cache[k] = cv2.imread(str(paths[k]))
        key = f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"
        img = noisy(cache[k], sg, key)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b, c, e = stats_from_gray(gray)                    # published
        br_pub = route_decision(b, c, e, THRESHOLDS)

        sh = immerkaer_sigma(gray)
        # variances add: sigma_added^2 = sigma_hat^2 - sigma_floor^2
        var_added = max(sh * sh - sigma_floor * sigma_floor, 0.0)
        c_rob = sqrt(max(c * c - var_added / (128.0 ** 2), 0.0))
        br_c1 = route_decision(b, c_rob, e, THRESHOLDS)    # V-C1

        med = cv2.medianBlur(gray, 3)
        b2, c2, e2 = stats_from_gray(med)
        br_c2 = route_decision(b2, c2, e2, THRESHOLDS)     # V-C2

        def compose(branch, low_op):
            op = low_op if branch == "clahe" else branch
            return int(r[f"pred_{op}"])

        rows.append({
            "filename": k[0], "occ": k[1], "ch": k[2], "sev": k[3],
            "sigma": sg, "true": int(r["true"]),
            "sigma_hat": round(sh, 3), "sigma_added": round(sqrt(var_added), 3),
            "c_raw": round(c, 5), "c_rob": round(c_rob, 5),
            "br_pub": br_pub, "br_c1": br_c1, "br_c2": br_c2,
            # 2x2 factorial
            "pred_VA": compose(br_pub, "clahe"),
            "pred_VB": compose(br_pub, "dcp"),
            "pred_C1_clahe": compose(br_c1, "clahe"),
            "pred_C1_dcp": compose(br_c1, "dcp"),
            "pred_C2_clahe": compose(br_c2, "clahe"),
            "pred_C2_dcp": compose(br_c2, "dcp"),
            "pred_dcp": int(r["pred_dcp"]),
            "pred_adair": int(r["pred_adair"]),
            "pred_passthrough": int(r["pred_passthrough"]),
        })
        if (i + 1) % 3000 == 0:
            print(f"  routed {i+1}/{len(T)}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "V_robust_routing.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ch = np.array([r["ch"] for r in rows]); sv = np.array([r["sev"] for r in rows])
    sg = np.array([r["sigma"] for r in rows]); tr = np.array([r["true"] for r in rows])
    BR = {v: np.array([r[f"br_{v}"] for r in rows]) for v in ("pub", "c1", "c2")}
    KEYS = ("VA", "VB", "C1_clahe", "C1_dcp", "C2_clahe", "C2_dcp",
            "dcp", "adair", "passthrough")
    PR = {kk: np.array([r[f"pred_{kk}"] for r in rows]) for kk in KEYS}

    def cm(c, s): return [(ch == c) & (sv == v) & (sg == s) for v in SEV]
    def acc(kk, c, s):
        return 100 * float(np.mean([np.mean(PR[kk][m] == tr[m])
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
        return tuple(float(v) for v in np.percentile(d, [2.5, 97.5]))

    out = {"sigma_floor": sigma_floor, "threshold_compat": {},
           "branch_share": {}, "factorial": {}}

    print("\n=== PRE-REGISTERED CRITERION: threshold compatibility at sigma=0 ===")
    m0 = (sg == 0)
    for v in ("c1", "c2"):
        agree = float(np.mean(BR[v][m0] == BR["pub"][m0]))
        ok = agree >= AGREE_MIN
        out["threshold_compat"][v] = {"agreement": agree, "passed": ok}
        print(f"  V-{v.upper()}: routing agrees with the published rule on "
              f"{100*agree:.2f}% of clean images -> "
              f"{'PASS' if ok else 'REJECTED (>=95% required)'}")

    print("\n=== Low-contrast branch share (the failure being repaired) ===")
    print(f"{'challenge':10s}{'sigma':>6s}{'published':>11s}{'V-C1':>8s}{'V-C2':>8s}")
    for c in (12, 9):
        out["branch_share"][CHN[c]] = {}
        for s in SIGMAS:
            m = (ch == c) & (sg == s)
            sh_ = {v: 100 * float(np.mean(BR[v][m] == "clahe"))
                   for v in ("pub", "c1", "c2")}
            out["branch_share"][CHN[c]][str(s)] = sh_
            print(f"{CHN[c]:10s}{s:6d}{sh_['pub']:10.1f}%{sh_['c1']:7.1f}%"
                  f"{sh_['c2']:7.1f}%")

    print("\n=== 2x2 FACTORIAL: routing fix vs operator swap ===")
    for c in (12, 9):
        print(f"\n  {CHN[c]}  (rows: accuracy; the two fixes are separable)")
        print(f"    {'sigma':>5s} | {'VA':>7s}{'V-B':>8s} | {'C1+CLAHE':>9s}"
              f"{'C1+DCP':>8s} | {'C2+CLAHE':>9s}{'C2+DCP':>8s} | "
              f"{'AdaIR':>7s}{'passthr':>8s}")
        out["factorial"][CHN[c]] = {}
        for s in SIGMAS:
            a = {kk: acc(kk, c, s) for kk in KEYS}
            lo, hi = boot("C1_dcp", "VB", c, s)
            out["factorial"][CHN[c]][str(s)] = {
                **a, "C1dcp_minus_VB": [round(a["C1_dcp"] - a["VB"], 2),
                                         round(lo, 2), round(hi, 2)]}
            print(f"    {s:5d} | {a['VA']:7.2f}{a['VB']:8.2f} | "
                  f"{a['C1_clahe']:9.2f}{a['C1_dcp']:8.2f} | "
                  f"{a['C2_clahe']:9.2f}{a['C2_dcp']:8.2f} | "
                  f"{a['adair']:7.2f}{a['passthrough']:8.2f}")
        print(f"    primary endpoint  (C1+DCP) - (V-B):")
        for s in SIGMAS:
            d, lo, hi = out["factorial"][CHN[c]][str(s)]["C1dcp_minus_VB"]
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
            print(f"      sigma={s:2d}: {d:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {sig}")

    with open(Path(args.outdir) / "V_robust_routing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote V_robust_routing.csv / .json to {args.outdir}")
    print("\nREADING GUIDE (pre-registered): a variant that fails the "
          "sigma=0 threshold-compatibility\ncriterion is reported as rejected. "
          "If the surviving variant does not restore the branch\nshares, the "
          "routing statistics are reported as intrinsically noise fragile and "
          "the\nlimitation stands.")


if __name__ == "__main__":
    main()
