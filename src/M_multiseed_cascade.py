# -*- coding: utf-8 -*-
"""
M_multiseed_cascade.py
Closes the last outstanding disclosed limitation, now correctly identified.

WHAT THE LIMITATION ACTUALLY IS
It was recorded as "single-seed router", which was wrong. The routing thresholds
T1..T4 are not derived from any sample: F_master_sweep_cache states in its own
header that they are the paper's values, and a percentile check confirms it (in
the clean split of the router-training data they sit at 12.83, 12.05, 12.40 and
69.22 per cent, which is no round quantile of anything). There is therefore no
seed in the thresholds to vary.

The one place a seed genuinely enters a reported number is the CASCADE. The
nested five-fold protocol assigns folds with a single generator seeded at 42,
and every confirmatory cascade result rests on that one assignment: tau = 0.2
selected in all five folds, a held-out estimate of 57.80 +/- 0.38 on the frozen
four-operator rule, and +0.13 +/- 0.04 on the DCP-augmented selector.

WHAT THIS SCRIPT DOES
It repeats the whole nested five-fold calibration under many different fold
assignments and reports how much the answer moves. Nothing is re-inferred: every
prediction is already cached, so only the fold assignment changes.

  base = the frozen four-operator rule   -> is "the cascade matches AdaIR" stable?
  base = the DCP-augmented selector V-B  -> is "escalation buys +0.13" stable?

For each seed: stratified five-fold assignment within every (challenge,
severity) cell; tau selected on the four development folds by cell-averaged
degraded accuracy; the held-out fold evaluated at that tau. Reported per seed
and then aggregated across seeds.

Writes: outputs_revision/M_multiseed_cascade.json
Run:    python M_multiseed_cascade.py                    (a few minutes)
        python M_multiseed_cascade.py --seeds 42,1,2,3   (fewer seeds)
"""
import argparse, csv, json, os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
DEG = list(range(1, 13))
TAUS = np.round(np.arange(0.0, 0.6, 0.025), 3)      # frozen grid (Part 3)
GATE_BRANCHES = ("clahe", "stretch")                # frozen gate
K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--seeds", default="42,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    Dc = {}
    with open(args.dcp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Dc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = int(r["pred_dcp"])
    M = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    print(f"loaded merged={len(M)} dcp={len(Dc)}")

    def col(n): return np.array([int(r[n]) for r in M])
    ch, sev, tru = col("ch"), col("sev"), col("true")
    va, ad = col("pred_va_rule"), col("pred_adair")
    br = np.array([r["rule_branch"] for r in M])
    pa = np.array([float(r["prob_adair"]) for r in M])
    dcp = np.array([Dc[(r["filename"], int(r["occ"]), int(r["ch"]),
                        int(r["sev"]))] for r in M])
    vb = np.where(br == "clahe", dcp, va)            # the V-B selector

    deg = np.isin(ch, DEG)
    CELLS = [(c, s) for c in DEG for s in range(1, 6)]
    CM = {cs: (ch == cs[0]) & (sev == cs[1]) for cs in CELLS}
    gate = np.isin(br, GATE_BRANCHES)

    def da_on(mask, p):
        v = [np.mean(p[CM[c] & mask] == tru[CM[c] & mask])
             for c in CELLS if (CM[c] & mask).sum()]
        return 100 * float(np.mean(v))

    BASES = {"frozen 4-operator rule": va, "DCP-augmented selector (V-B)": vb}
    out = {"seeds": seeds, "bases": {}}

    for bname, base in BASES.items():
        print(f"\n{'=' * 74}\nBASE: {bname}\n{'=' * 74}")
        print(f"  {'seed':>5s}{'taus per fold':>34s}"
              f"{'held-out':>11s}{'base':>9s}{'AdaIR':>8s}{'paired':>9s}")
        rows = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            fold = np.full(len(M), -1)
            for c in CELLS:                       # stratified within each cell
                idx = np.where(CM[c])[0]
                rng.shuffle(idx)
                for k, part in enumerate(np.array_split(idx, K)):
                    fold[part] = k
            H, BB, AA, TT = [], [], [], []
            for k in range(K):
                dev, tst = deg & (fold != k), deg & (fold == k)
                best_t, best_a = 0.0, -1.0
                for t in TAUS:
                    a = da_on(dev, np.where(gate & (pa > t), ad, base))
                    if a > best_a:
                        best_a, best_t = a, float(t)
                m = gate & (pa > best_t)
                H.append(da_on(tst, np.where(m, ad, base)))
                BB.append(da_on(tst, base))
                AA.append(da_on(tst, ad))
                TT.append(best_t)
            H, BB, AA = map(np.array, (H, BB, AA))
            rows.append({"seed": sd, "taus": TT,
                         "heldout": float(H.mean()),
                         "base": float(BB.mean()),
                         "adair": float(AA.mean()),
                         "paired": float((H - BB).mean())})
            print(f"  {sd:5d}{str(TT):>34s}{H.mean():11.2f}{BB.mean():9.2f}"
                  f"{AA.mean():8.2f}{(H-BB).mean():+9.2f}")

        h = np.array([r["heldout"] for r in rows])
        p = np.array([r["paired"] for r in rows])
        allt = [t for r in rows for t in r["taus"]]
        uniq, cnt = np.unique(allt, return_counts=True)
        print(f"\n  across {len(seeds)} seeds:")
        print(f"    held-out          : {h.mean():.2f} +/- {h.std():.2f}  "
              f"(range {h.min():.2f} to {h.max():.2f})")
        print(f"    cascade minus base: {p.mean():+.2f} +/- {p.std():.2f}  "
              f"(range {p.min():+.2f} to {p.max():+.2f})")
        print(f"    tau chosen ({len(allt)} folds): " +
              ", ".join(f"{u}x{c}" for u, c in zip(uniq, cnt)))
        print(f"    AdaIR alone       : "
              f"{np.mean([r['adair'] for r in rows]):.2f}")
        print(f"    invocation rate   : {100*np.mean(gate[deg]):.1f}%  "
              f"(gate is severity- and seed-independent)")
        out["bases"][bname] = {
            "per_seed": rows,
            "heldout_mean": round(float(h.mean()), 2),
            "heldout_std_across_seeds": round(float(h.std()), 3),
            "paired_mean": round(float(p.mean()), 2),
            "paired_std_across_seeds": round(float(p.std()), 3),
            "tau_counts": {str(u): int(c) for u, c in zip(uniq, cnt)},
            "invocation_pct": round(100 * float(np.mean(gate[deg])), 1),
        }

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "M_multiseed_cascade.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {Path(args.outdir) / 'M_multiseed_cascade.json'}")
    print("\nREADING GUIDE\n"
          "  The dispersion ACROSS SEEDS is the quantity the limitation asked "
          "for. If it is small\n  relative to the effect, the cascade result "
          "does not depend on the fold assignment and\n  the limitation is "
          "discharged. If it is large, the manuscript must report the spread "
          "and\n  not the single-seed number.")


if __name__ == "__main__":
    main()
