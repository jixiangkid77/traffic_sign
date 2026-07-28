# -*- coding: utf-8 -*-
"""
S_veil_ablation.py
Locks down the veil-rendering confound analysis on the authoritative machine.

BACKGROUND (why this exists)
The CURE-TSD generation recipe (project GitHub) renders Haze as a light-gray
solid layer (#CECECE) alpha-composited at 10/20/30/40/50% for severities 1-5,
plus a depth-following radial gradient (#D6D6D6). That is the atmospheric
scattering model I = J(1-alpha) + A*alpha with a near-constant transmission --
exactly the forward model the dark channel prior inverts. Rain likewise carries
a blue-ish gradient veil. DCP's advantage on these two challenges may therefore
be inflated by a generator/prior match, the same circularity the manuscript
itself invokes in Sec. IV-B to exclude synthetic-fog comparisons. This script
quantifies how much of every DCP-related claim survives when the veil-rendered
challenges are removed.

Computes
  1. Veil ablation: DCP-AdaIR and V-B-AdaIR over all 12 / minus Haze /
     minus Haze+Rain, with canonical per-comparison bootstrap CIs (seed 42).
  2. Oracle ablation: oracle-4 / oracle-5 (+DCP) / oracle-6 (+2 deep) under the
     same exclusions -- tests whether "operator space >= model space" survives.
  3. Confirmatory cascade on the DCP-augmented selector (Protocol Part 10):
     the FROZEN nested 5-fold protocol of Part 3 applied unchanged to a new base
     predictor. Gate, tau grid and endpoint are identical to the primary cascade;
     no selection among alternatives is performed.

Reads:  outputs_revision/merged_per_image.csv, outputs_revision/dcp_cure.csv
Writes: outputs_revision/S_veil_ablation.json
Run:    python S_veil_ablation.py      (~4-8 min CPU)
"""
import argparse, csv, json, os
import numpy as np

PROJECT_ROOT = r"D:\Project\traffic_sign"
B, SEED = 5000, 42
TAUS = np.round(np.arange(0.0, 0.6, 0.025), 3)
EXCL = {"all12": (), "ex_haze": (12,), "ex_haze_rain": (12, 9)}

EXPECTED = {
    "points": {
        "all12":        {"dcp": 58.89, "adair": 57.78, "vb": 58.21, "va": 56.65},
        "ex_haze":      {"dcp": 57.69, "adair": 57.04, "vb": 57.25, "va": 56.45},
        "ex_haze_rain": {"dcp": 58.88, "adair": 59.05, "vb": 58.99, "va": 58.68},
    },
    "diffs": {   # (point, lo, hi)
        ("all12", "DCP-AdaIR"):        (+1.11, +0.87, +1.35),
        ("all12", "VB-AdaIR"):         (+0.43, +0.24, +0.61),
        ("ex_haze", "DCP-AdaIR"):      (+0.65, +0.39, +0.89),
        ("ex_haze", "VB-AdaIR"):       (+0.21, +0.01, +0.39),
        ("ex_haze_rain", "DCP-AdaIR"): (-0.17, -0.42, +0.08),
        ("ex_haze_rain", "VB-AdaIR"):  (-0.06, -0.24, +0.12),
    },
    "oracles": {   # o4, o5, o6
        "all12":        (66.42, 70.52, 69.37),
        "ex_haze":      (66.42, 69.70, 68.79),
        "ex_haze_rain": (68.94, 71.24, 71.18),
    },
    "cascade_on_vb": {"heldout": 58.34, "vb_base": 58.21, "adair": 57.78,
                      "paired_diff": 0.13, "invocation": 18.9},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=os.path.join(
        PROJECT_ROOT, "outputs_revision", "merged_per_image.csv"))
    ap.add_argument("--dcp", default=os.path.join(
        PROJECT_ROOT, "outputs_revision", "dcp_cure.csv"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    D = {}
    with open(args.dcp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            D[(r["filename"], int(r["occ"]), int(r["ch"]),
               int(r["sev"]))] = int(r["pred_dcp"])
    M = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    print(f"loaded merged={len(M)} dcp={len(D)}")

    def col(n): return np.array([int(r[n]) for r in M])
    ch, sev, tru = col("ch"), col("sev"), col("true")
    pp, pg, pc, ps = (col("pred_passthrough"), col("pred_gamma"),
                      col("pred_clahe"), col("pred_stretch"))
    va, ad, cd = col("pred_va_rule"), col("pred_adair"), col("pred_cidnet")
    br = np.array([r["rule_branch"] for r in M])
    pa = np.array([float(r["prob_adair"]) for r in M])
    dcp = np.array([D[(r["filename"], int(r["occ"]), int(r["ch"]),
                       int(r["sev"]))] for r in M])
    vb = np.where(br == "clahe", dcp, va)          # V-B selector

    CM = {(c, s): (ch == c) & (sev == s)
          for c in range(1, 13) for s in range(1, 6)}
    def cells(ex): return [(c, s) for c in range(1, 13) if c not in ex
                           for s in range(1, 6)]
    def da(p, ex=()):
        return 100 * float(np.mean([np.mean(p[CM[k]] == tru[CM[k]])
                                    for k in cells(ex)]))
    def boot(pA, pB, ex=()):
        rng = np.random.default_rng(SEED)          # canonical: fresh per call
        ks = cells(ex)
        A = [(pA[CM[k]] == tru[CM[k]]).astype(float) for k in ks]
        Bb = [(pB[CM[k]] == tru[CM[k]]).astype(float) for k in ks]
        d = np.empty(B)
        for b in range(B):
            sa = sb = 0.0
            for a, bb in zip(A, Bb):
                i = rng.integers(0, len(a), len(a))
                sa += a[i].mean(); sb += bb[i].mean()
            d[b] = 100 * (sa - sb) / len(ks)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi)
    def orc(preds, ex=()):
        v = []
        for k in cells(ex):
            m = CM[k]; ok = np.zeros(int(m.sum()), bool)
            for p in preds:
                ok |= (p[m] == tru[m])
            v.append(100 * ok.mean())
        return float(np.mean(v))

    out = {"points": {}, "diffs": {}, "oracles": {}, "cascade_on_vb": {}}

    print("\n=== 1. VEIL ABLATION (canonical CIs, seed 42 per comparison) ===")
    print(f"{'scope':14s}{'comparison':12s}{'diff':>8s}   95% CI")
    for nm, ex in EXCL.items():
        out["points"][nm] = {"dcp": da(dcp, ex), "adair": da(ad, ex),
                             "vb": da(vb, ex), "va": da(va, ex)}
        for lbl, a, b_ in [("DCP-AdaIR", dcp, ad), ("VB-AdaIR", vb, ad)]:
            d = da(a, ex) - da(b_, ex)
            lo, hi = boot(a, b_, ex)
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (tied)"
            out["diffs"][f"{nm}|{lbl}"] = {"point": round(d, 2),
                                            "lo": round(lo, 2),
                                            "hi": round(hi, 2), "sig": sig}
            print(f"{nm:14s}{lbl:12s}{d:+8.2f}   [{lo:+.2f},{hi:+.2f}]  {sig}")

    print("\n=== 2. ORACLE ABLATION (does 'operator space >= model space' "
          "survive?) ===")
    print(f"{'scope':14s}{'oracle4':>9s}{'oracle5':>9s}{'+DCP':>8s}"
          f"{'oracle6':>9s}{'+2deep':>8s}")
    for nm, ex in EXCL.items():
        o4 = orc([pp, pg, pc, ps], ex)
        o5 = orc([pp, pg, pc, ps, dcp], ex)
        o6 = orc([pp, pg, pc, ps, ad, cd], ex)
        out["oracles"][nm] = {"o4": o4, "o5": o5, "o6": o6,
                              "gain_one_operator": round(o5 - o4, 2),
                              "gain_two_deep": round(o6 - o4, 2)}
        print(f"{nm:14s}{o4:9.2f}{o5:9.2f}{o5-o4:+8.2f}{o6:9.2f}{o6-o4:+8.2f}")

    print("\n=== 3. CONFIRMATORY cascade on the DCP-augmented selector "
          "(Protocol Part 10) ===")
    deg = np.isin(ch, list(range(1, 13)))
    rng = np.random.default_rng(SEED)
    fold = np.full(len(M), -1)
    for c in range(1, 13):
        for s in range(1, 6):
            idx = np.where(CM[(c, s)])[0]
            rng.shuffle(idx)
            for k, sp in enumerate(np.array_split(idx, 5)):
                fold[sp] = k
    gate = np.isin(br, ["clahe", "stretch"])
    def da_on(mask, p):
        v = []
        for k in cells(()):
            m = CM[k] & mask
            if m.sum():
                v.append(np.mean(p[m] == tru[m]))
        return 100 * float(np.mean(v))
    H, AA, VV, UU, TT = [], [], [], [], []
    for k in range(5):
        dev, tst = deg & (fold != k), deg & (fold == k)
        bt, ba = 0.0, -1.0
        for t in TAUS:
            a = da_on(dev, np.where(gate & (pa > t), ad, vb))
            if a > ba:
                ba, bt = a, float(t)
        m = gate & (pa > bt)
        H.append(da_on(tst, np.where(m, ad, vb)))
        AA.append(da_on(tst, ad)); VV.append(da_on(tst, vb))
        UU.append(100 * float(np.mean(gate[tst]))); TT.append(bt)
    H, AA, VV, UU = map(np.array, (H, AA, VV, UU))
    out["cascade_on_vb"] = {
        "tau_per_fold": TT, "heldout_mean": float(H.mean()),
        "heldout_std": float(H.std()), "vb_base_mean": float(VV.mean()),
        "adair_mean": float(AA.mean()),
        "paired_diff_mean": float((H - VV).mean()),
        "paired_diff_std": float((H - VV).std()),
        "invocation_pct": float(UU.mean()),
    }
    print(f"  tau per fold: {TT}")
    print(f"  cascade-on-V-B held-out : {H.mean():.2f} +/- {H.std():.2f}")
    print(f"  V-B base (no deep model): {VV.mean():.2f} +/- {VV.std():.2f}")
    print(f"  AdaIR alone             : {AA.mean():.2f} +/- {AA.std():.2f}")
    print(f"  cascade - V-B (paired)  : {(H-VV).mean():+.2f} +/- "
          f"{(H-VV).std():.2f}  at {UU.mean():.1f}% deep invocation")
    print("  READING: on the 4-operator rule the same cascade bought +1.13 "
          "(matching AdaIR);\n           on the 5-operator selector it buys "
          "+0.13. The marginal value of deep\n           escalation collapses "
          "as the operator pool is better matched.")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.merged))
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "S_veil_ablation.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outp}")

    print("\n=== EXPECTED OUTPUT AUDIT ===")
    ok = True
    def chk(n, got, exp, tol=0.02):
        nonlocal ok
        p = abs(got - exp) <= tol
        ok &= p
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {got:.2f} vs {exp}")
    for nm in EXCL:
        for kk, v in EXPECTED["points"][nm].items():
            chk(f"{nm} {kk}", out["points"][nm][kk], v, 0.01)
    for (nm, lbl), (pt, lo, hi) in EXPECTED["diffs"].items():
        g = out["diffs"][f"{nm}|{lbl}"]
        chk(f"{nm} {lbl} point", g["point"], pt, 0.01)
        chk(f"{nm} {lbl} lo", g["lo"], lo)
        chk(f"{nm} {lbl} hi", g["hi"], hi)
    for nm, (o4, o5, o6) in EXPECTED["oracles"].items():
        chk(f"{nm} oracle4", out["oracles"][nm]["o4"], o4, 0.01)
        chk(f"{nm} oracle5", out["oracles"][nm]["o5"], o5, 0.01)
        chk(f"{nm} oracle6", out["oracles"][nm]["o6"], o6, 0.01)
    e = EXPECTED["cascade_on_vb"]
    chk("cascade-on-VB heldout", out["cascade_on_vb"]["heldout_mean"],
        e["heldout"], 0.01)
    chk("cascade-on-VB vb_base", out["cascade_on_vb"]["vb_base_mean"],
        e["vb_base"], 0.01)
    chk("cascade-on-VB paired", out["cascade_on_vb"]["paired_diff_mean"],
        e["paired_diff"], 0.01)
    chk("cascade-on-VB invocation", out["cascade_on_vb"]["invocation_pct"],
        e["invocation"], 0.05)
    print("\nALL CHECKS PASSED -- the ablation numbers may be quoted."
          if ok else "\n*** MISMATCH: send me this output. ***")


if __name__ == "__main__":
    main()
