# -*- coding: utf-8 -*-
"""
R_dcp_significance.py
Reproduces on the authoritative machine EVERY DCP-related significance number
destined for the paper, under the CANONICAL bootstrap convention:

    each comparison uses an independently seeded generator (seed 42),
    cell-stratified paired percentile bootstrap, B = 5000,
    so any single interval is exactly reproducible in isolation
    (Evaluation Protocol, Part 6).

Reads:  outputs_revision/merged_per_image.csv   (12-class aligned family)
        outputs_revision/dcp_cure.csv           (Q output)
Writes: outputs_revision/R_dcp_significance.json
Run:    python R_dcp_significance.py            (~3-6 min CPU)

Ends with an EXPECTED OUTPUT AUDIT; every line must PASS before any of these
numbers is quoted in the manuscript.
"""
import argparse, csv, json, os
from math import comb, erfc, sqrt
import numpy as np

PROJECT_ROOT = r"D:\Project\traffic_sign"
B = 5000
SEED = 42

EXPECTED = {
    "points": {"dcp_deg": 58.89, "vb_deg": 58.21, "dcp_f1": 39.82,
               "dcp_cf": 79.36, "vb_cf": 80.77, "oracle5": 70.52,
               "adair_unique_vs_5pool": 0.65, "exploratory_vb2": 59.01},
    "comparisons": {   # name: (point, lo, hi, mcnemar_b, mcnemar_c)
        "DCP-AdaIR":             (+1.11, +0.87, +1.35, 5614, 4713),
        "DCP-CIDNet":            (+2.58, +2.31, +2.85, 7966, 5873),
        "VB-VA":                 (+1.57, +1.46, +1.67, 1610,  340),
        "VB-AdaIR":              (+0.43, +0.24, +0.61, 3352, 3004),
        "DCP-AdaIR Haze":        (+6.23, +5.22, +7.23,  834,  413),
        "DCP-AdaIR Rain":        (+8.76, +7.65, +9.88, 1061,  469),
        "DCP-AdaIR LensBlur":    (+3.36, +2.29, +4.39,  778,  551),
        "DCP-AdaIR GaussBlur":   (+4.97, +3.93, +6.02,  859,  523),
        "DCP-AdaIR Darkening":   (-8.27, -9.11, -7.41,  239,  798),
        "gamma-AdaIR Darkening": (+4.23, +3.39, +5.07,  578,  292),
    },
    "cf_comparison": (-1.41, -2.74, +0.00, 35, 54),   # DCP-passthrough on CF
    "haze_sev_diff": [-0.3, -1.9, -1.4, +11.7, +23.1],
    "degeneracy": {"Darkening_std_pct": 20.5, "Haze_clip_pct": 13.8,
                   "Exposure_clip_pct": 6.8},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=os.path.join(
        PROJECT_ROOT, "outputs_revision", "merged_per_image.csv"))
    ap.add_argument("--dcp", default=os.path.join(
        PROJECT_ROOT, "outputs_revision", "dcp_cure.csv"))
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: directory of --merged)")
    args = ap.parse_args()

    D, CLIP, OSTD = {}, {}, {}
    with open(args.dcp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
            D[k] = int(r["pred_dcp"])
            CLIP[k] = float(r["t_clip_frac"]); OSTD[k] = float(r["out_std"])
    M = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    print(f"loaded merged={len(M)} dcp={len(D)}")

    def col(n): return np.array([int(r[n]) for r in M])
    ch, sev, tru = col("ch"), col("sev"), col("true")
    pp, pg, pc, ps = (col("pred_passthrough"), col("pred_gamma"),
                      col("pred_clahe"), col("pred_stretch"))
    va, ad, cd = col("pred_va_rule"), col("pred_adair"), col("pred_cidnet")
    br = np.array([r["rule_branch"] for r in M])
    keys = [(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
            for r in M]
    dcp = np.array([D[k] for k in keys])
    clipf = np.array([CLIP[k] for k in keys])
    ostd = np.array([OSTD[k] for k in keys])
    DEG = list(range(1, 13)); deg = np.isin(ch, DEG)
    CELLS = [(c, s) for c in DEG for s in range(1, 6)]
    cm = {cs: (ch == cs[0]) & (sev == cs[1]) for cs in CELLS}
    cf = (ch == 0)
    vb = np.where(br == "clahe", dcp, va)
    vb2 = np.where(np.isin(br, ["clahe", "stretch"]), dcp, va)

    def degavg(p, restrict=None):
        cs = [c for c in CELLS if restrict is None or c[0] == restrict]
        return 100 * np.mean([np.mean(p[cm[c]] == tru[cm[c]]) for c in cs])

    classes = sorted(set(tru.tolist()))
    def macro_f1_cells(p):
        vals = []
        for c in CELLS:
            t_, p_ = tru[cm[c]], p[cm[c]]
            f1s = []
            for cls in classes:
                tp = np.sum((t_ == cls) & (p_ == cls))
                fp = np.sum((t_ != cls) & (p_ == cls))
                fn = np.sum((t_ == cls) & (p_ != cls))
                pr = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
            vals.append(100 * np.mean(f1s))
        return float(np.mean(vals))

    def boot(pA, pB, restrict=None):
        rng = np.random.default_rng(SEED)          # canonical: fresh per call
        cs = [c for c in CELLS if restrict is None or c[0] == restrict]
        okA = [(pA[cm[c]] == tru[cm[c]]).astype(np.float64) for c in cs]
        okB = [(pB[cm[c]] == tru[cm[c]]).astype(np.float64) for c in cs]
        diffs = np.empty(B)
        for b in range(B):
            da_ = db_ = 0.0
            for a, bb in zip(okA, okB):
                idx = rng.integers(0, len(a), len(a))
                da_ += a[idx].mean(); db_ += bb[idx].mean()
            diffs[b] = 100 * (da_ - db_) / len(cs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return float(lo), float(hi)

    def mcn(pA, pB, mask):
        b = int(np.sum((pA[mask] == tru[mask]) & (pB[mask] != tru[mask])))
        c = int(np.sum((pA[mask] != tru[mask]) & (pB[mask] == tru[mask])))
        n = b + c
        if n == 0:
            return b, c, 1.0
        if n <= 1000:
            p = min(1.0, 2 * sum(comb(n, i)
                                 for i in range(min(b, c) + 1)) / 2 ** n)
        else:
            p = erfc(abs(b - c) / sqrt(n) / sqrt(2))
        return b, c, float(p)

    out = {"points": {}, "comparisons": {}, "cf_comparison": None,
           "haze_sev_diff": [], "degeneracy": {}}

    # ---- points ----
    out["points"] = {
        "dcp_deg": degavg(dcp), "vb_deg": degavg(vb),
        "dcp_f1": macro_f1_cells(dcp),
        "dcp_cf": 100 * float(np.mean(dcp[cf] == tru[cf])),
        "vb_cf": 100 * float(np.mean(vb[cf] == tru[cf])),
        "exploratory_vb2": degavg(vb2),
    }
    o5 = np.mean([100 * np.mean((pp[cm[c]] == tru[cm[c]]) |
                                 (pg[cm[c]] == tru[cm[c]]) |
                                 (pc[cm[c]] == tru[cm[c]]) |
                                 (ps[cm[c]] == tru[cm[c]]) |
                                 (dcp[cm[c]] == tru[cm[c]])) for c in CELLS])
    uni = 100 * np.mean(((ad == tru) & ~((pp == tru) | (pg == tru) |
            (pc == tru) | (ps == tru) | (dcp == tru)))[deg])
    out["points"]["oracle5"] = float(o5)
    out["points"]["adair_unique_vs_5pool"] = float(uni)

    # ---- comparisons ----
    spec = [("DCP-AdaIR", dcp, ad, None), ("DCP-CIDNet", dcp, cd, None),
            ("VB-VA", vb, va, None), ("VB-AdaIR", vb, ad, None),
            ("DCP-AdaIR Haze", dcp, ad, 12), ("DCP-AdaIR Rain", dcp, ad, 9),
            ("DCP-AdaIR LensBlur", dcp, ad, 2),
            ("DCP-AdaIR GaussBlur", dcp, ad, 7),
            ("DCP-AdaIR Darkening", dcp, ad, 4),
            ("gamma-AdaIR Darkening", pg, ad, 4)]
    print("\n=== canonical comparisons (independent seed 42 each) ===")
    for nm, a, b_, rc in spec:
        d = degavg(a, rc) - degavg(b_, rc)
        lo, hi = boot(a, b_, rc)
        m = deg if rc is None else (deg & (ch == rc))
        bb, cc, p = mcn(a, b_, m)
        out["comparisons"][nm] = {"point": round(d, 2), "lo": round(lo, 2),
                                   "hi": round(hi, 2), "b": bb, "c": cc, "p": p}
        print(f"  {nm:24s} {d:+.2f} [{lo:+.2f},{hi:+.2f}]  b={bb} c={cc} "
              f"p={p:.2e}")

    # ---- CF clean-cost comparison (single-cell bootstrap) ----
    rng = np.random.default_rng(SEED)
    okD = (dcp[cf] == tru[cf]).astype(float)
    okP = (pp[cf] == tru[cf]).astype(float)
    idx = rng.integers(0, cf.sum(), (B, int(cf.sum())))
    ds = 100 * (okD[idx].mean(axis=1) - okP[idx].mean(axis=1))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    bb, cc, p = mcn(dcp, pp, cf)
    out["cf_comparison"] = {"point": round(float(okD.mean()*100 -
        okP.mean()*100), 2), "lo": round(float(lo), 2),
        "hi": round(float(hi), 2), "b": bb, "c": cc, "p": p}
    print(f"\nCF: DCP-passthrough {out['cf_comparison']['point']:+.2f} "
          f"[{lo:+.2f},{hi:+.2f}] b={bb} c={cc} p={p:.2e}  "
          f"(clean cost is NOT established; selector removes it "
          f"structurally: 0% low-contrast routing on CF)")

    # ---- Haze severity trend ----
    print("\n=== Haze severity trend (DCP - AdaIR) ===")
    for s in range(1, 6):
        m = cm[(12, s)]
        d = 100 * (np.mean(dcp[m] == tru[m]) - np.mean(ad[m] == tru[m]))
        out["haze_sev_diff"].append(round(float(d), 1))
        print(f"  sev{s}: {d:+.1f}")

    # ---- degeneracy hotspots ----
    def chmask(c): return deg & (ch == c)
    out["degeneracy"] = {
        "Darkening_std_pct": round(100 * float(np.mean(
            ostd[chmask(4)] < 5 / 255)), 1),
        "Haze_clip_pct": round(100 * float(np.mean(
            clipf[chmask(12)] >= 0.8)), 1),
        "Exposure_clip_pct": round(100 * float(np.mean(
            clipf[chmask(6)] >= 0.8)), 1),
    }
    print(f"\ndegeneracy hotspots: {out['degeneracy']} "
          f"(all far below the 50% pre-registered trigger)")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.merged))
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "R_dcp_significance.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outp}")

    # ---- EXPECTED OUTPUT AUDIT ----
    print("\n=== EXPECTED OUTPUT AUDIT ===")
    ok = True
    def chk(name, got, exp, tol):
        nonlocal ok
        passed = abs(got - exp) <= tol
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {got} vs {exp}")
    for k, v in EXPECTED["points"].items():
        chk(f"point {k}", round(out["points"][k], 2), v, 0.01)
    for nm, (pt, lo, hi, bb, cc) in EXPECTED["comparisons"].items():
        g = out["comparisons"][nm]
        chk(f"{nm} point", g["point"], pt, 0.01)
        chk(f"{nm} lo", g["lo"], lo, 0.02)
        chk(f"{nm} hi", g["hi"], hi, 0.02)
        chk(f"{nm} b", g["b"], bb, 0); chk(f"{nm} c", g["c"], cc, 0)
    pt, lo, hi, bb, cc = EXPECTED["cf_comparison"]
    g = out["cf_comparison"]
    chk("CF point", g["point"], pt, 0.01); chk("CF lo", g["lo"], lo, 0.02)
    chk("CF hi", g["hi"], hi, 0.02); chk("CF b", g["b"], bb, 0)
    chk("CF c", g["c"], cc, 0)
    for i, v in enumerate(EXPECTED["haze_sev_diff"]):
        chk(f"haze sev{i+1} diff", out["haze_sev_diff"][i], v, 0.05)
    for k, v in EXPECTED["degeneracy"].items():
        chk(f"degeneracy {k}", out["degeneracy"][k], v, 0.1)
    print("\nALL CHECKS PASSED -- these numbers may be quoted in the "
          "manuscript." if ok else "\n*** MISMATCH: send me this output; do "
          "NOT quote any DCP significance number until resolved. ***")


if __name__ == "__main__":
    main()
