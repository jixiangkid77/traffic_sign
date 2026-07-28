# -*- coding: utf-8 -*-
"""
O_cascade_kfold.py
Cascade policy verification: baselines, P1 branch-gate, tau sweep,
Oracle Cascade, and 5-fold stratified stability curves.

Reads:  outputs_revision/merged_per_image.csv
Writes: outputs_revision/O_cascade_kfold.results.json
        outputs_revision/O_cascade_curves.csv

Run:    python O_cascade_kfold.py
        python O_cascade_kfold.py --csv <path-to-merged_per_image.csv>

Deterministic: fixed seed 42; identical input must reproduce the
EXPECTED OUTPUT block printed at the end to the last digit.
Requires: numpy >= 1.17 (uses np.random.default_rng).
"""
import argparse, csv, json, os
import numpy as np

PROJECT_ROOT = r"D:\Project\traffic_sign"
DEG_CH = [4, 8, 9, 11, 12]          # darkening, noise, rain, snow, haze
GATE_BRANCHES = ("clahe", "stretch") # P1 escalation gate
TAUS = np.round(np.arange(0.0, 1.0, 0.05), 2)
SEED = 42
K = 5

# reference values every rerun must reproduce (2 d.p. unless noted)
EXPECTED = {
    "va_degavg": 57.32, "adair_degavg": 58.73,
    "p1_degavg": 58.63, "p1_escalation_pct": 17.6,
    "p4_tau020_degavg": 59.40,
    "oracle_cascade_degavg": 63.43,
    "oracle_needed_escalation_pct": 42.7,
    "gate_precision_pct": 68.4, "gate_recall_pct": 28.3,
    "fold_sizes": [6775, 6775, 6750, 6750, 6750],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        PROJECT_ROOT, "outputs_revision", "merged_per_image.csv"))
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: directory of --csv)")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    ch  = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    va  = np.array([int(r["pred_va_rule"]) for r in rows])
    ad  = np.array([int(r["pred_adair"]) for r in rows])
    br  = np.array([r["rule_branch"] for r in rows])
    pa  = np.array([float(r["prob_adair"]) for r in rows])
    deg = np.isin(ch, DEG_CH)
    print(f"loaded {len(rows)} rows; degraded rows: {int(deg.sum())}")

    def degavg(pred, subset=None):
        m0 = deg if subset is None else (deg & subset)
        accs = []
        for cc in DEG_CH:
            for ss in range(1, 6):
                m = m0 & (ch == cc) & (sev == ss)
                if m.sum() == 0:
                    continue
                accs.append(float(np.mean(pred[m] == tru[m])))
        return 100.0 * float(np.mean(accs))

    out = {}

    # ---------- baselines ----------
    out["va_degavg"] = degavg(va)
    out["adair_degavg"] = degavg(ad)

    # ---------- P1: branch-gated escalation ----------
    m1 = np.isin(br, list(GATE_BRANCHES))
    p1 = np.where(m1, ad, va)
    out["p1_degavg"] = degavg(p1)
    out["p1_escalation_pct"] = 100.0 * float(np.mean(m1[deg]))

    # ---------- full-data tau sweep (P4 family) ----------
    sweep = []
    for tau in TAUS:
        m4 = m1 & (pa > tau)
        p4 = np.where(m4, ad, va)
        sweep.append({
            "tau": float(tau),
            "degavg": degavg(p4),
            "adair_pred_used_pct": 100.0 * float(np.mean(m4[deg])),
        })
    out["tau_sweep_full"] = sweep
    out["p4_tau020_degavg"] = next(
        s["degavg"] for s in sweep if abs(s["tau"] - 0.20) < 1e-9)

    # ---------- Oracle Cascade ----------
    need = (va != tru)                       # escalation is useful only here
    oracle_pred = np.where(need, ad, va)
    out["oracle_cascade_degavg"] = degavg(oracle_pred)
    out["oracle_needed_escalation_pct"] = 100.0 * float(np.mean(need[deg]))
    tp = int(np.sum(m1[deg] & need[deg]))
    fp = int(np.sum(m1[deg] & ~need[deg]))
    fn = int(np.sum(~m1[deg] & need[deg]))
    out["gate_precision_pct"] = 100.0 * tp / (tp + fp)
    out["gate_recall_pct"] = 100.0 * tp / (tp + fn)
    out["gate_counts"] = {"tp": tp, "fp": fp, "fn": fn,
                          "degraded_total": int(deg.sum())}

    # ---------- 5-fold stratified stability curves ----------
    rng = np.random.default_rng(SEED)
    fold = np.full(len(rows), -1, dtype=int)
    for cc in DEG_CH:
        for ss in range(1, 6):
            idx = np.where((ch == cc) & (sev == ss))[0]
            rng.shuffle(idx)
            for k, s in enumerate(np.array_split(idx, K)):
                fold[s] = k
    out["fold_sizes"] = [int(np.sum((fold == k) & deg)) for k in range(K)]

    curves = np.zeros((K, len(TAUS)))
    for k in range(K):
        sub = (fold == k)
        for i, tau in enumerate(TAUS):
            m4 = m1 & (pa > tau)
            p4 = np.where(m4, ad, va)
            curves[k, i] = degavg(p4, subset=sub)
    mean_c, std_c = curves.mean(axis=0), curves.std(axis=0)
    out["kfold"] = {
        "K": K, "seed": SEED, "taus": [float(t) for t in TAUS],
        "per_fold_curves": curves.tolist(),
        "mean_curve": mean_c.tolist(), "std_curve": std_c.tolist(),
        "mean_peak_tau": float(TAUS[int(np.argmax(mean_c))]),
        "per_fold_peak_tau": [float(TAUS[int(np.argmax(curves[k]))])
                              for k in range(K)],
    }

    # ---------- write artifacts ----------
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "O_cascade_kfold.results.json")
    cpath = os.path.join(outdir, "O_cascade_curves.csv")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau"] + [f"fold{k}" for k in range(K)] + ["mean", "std"])
        for i, tau in enumerate(TAUS):
            w.writerow([f"{tau:.2f}"] + [f"{curves[k, i]:.4f}" for k in range(K)]
                       + [f"{mean_c[i]:.4f}", f"{std_c[i]:.4f}"])
    print(f"wrote {jpath}\nwrote {cpath}\n")

    # ---------- console table ----------
    hdr = "  tau | " + " | ".join(f"fold{k}" for k in range(K)) + " |  mean   std"
    print(hdr)
    for i, tau in enumerate(TAUS):
        row = " | ".join(f"{curves[k, i]:5.2f}" for k in range(K))
        print(f" {tau:4.2f} | {row} | {mean_c[i]:6.2f}  {std_c[i]:4.2f}")

    # ---------- expected-output audit ----------
    print("\n=== EXPECTED OUTPUT AUDIT (all must PASS) ===")
    ok = True
    for key, ref in EXPECTED.items():
        got = out[key]
        if isinstance(ref, list):
            passed = (got == ref)
            shown = got
        else:
            passed = abs(got - ref) < 0.005 if key.endswith("degavg") \
                     else abs(got - ref) < 0.05
            shown = f"{got:.2f}" if isinstance(got, float) else got
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {key}: got {shown}, expected {ref}")
    print("\nALL CHECKS PASSED" if ok else "\n*** MISMATCH: investigate before "
          "using any cascade number ***")

if __name__ == "__main__":
    main()
