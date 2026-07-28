# -*- coding: utf-8 -*-
"""
AD_deployment_arithmetic.py
Post-hoc analysis (Evaluation Protocol, Part 17.6; declared POST HOC).

THE PROBLEM THIS FIXES
Section V-D must state an unfavourable fact: always-DCP reaches a higher
degraded-average accuracy (58.89) than the selector V-B (58.21). Left there, the
paper invites the obvious reply: "then just always run DCP, and drop the
selector". That reply is wrong, but the manuscript has to SHOW why rather than
assert it. This script computes the three quantities that show it. All three are
table lookups and arithmetic over the already-locked per-image predictions. No
new experiment, no new model, no new data.

  BEAM 1  CLEAN-TRAFFIC IMMUNITY
          On the ChallengeFree split the base rule routes every image to
          passthrough, so V-B is bit-identical to no enhancement (80.77), while
          always-DCP pays 1.41 points for enhancing images that need nothing.

  BEAM 2  MIXTURE BREAK-EVEN
          A deployed camera does not see the degraded split; it sees a mixture.
          Write w for the fraction of clean frames. The mixture accuracy of a
          method is w * CF + (1 - w) * deg-avg. V-B overtakes always-DCP when
              w * 1.41  >  (1 - w) * 0.68,
          that is, above a break-even fraction reported below. The manuscript
          states the weighting convention explicitly and calls this an arithmetic
          corollary of Table II, NOT a new benchmark.

  BEAM 3  STRICT DOMINANCE AND THE LATENCY BILL
          Against always-AdaIR, V-B is higher on BOTH axes (clean and degraded),
          so it wins at every mixture w, with no break-even to argue about. And
          the 11.1 ms operator is invoked only on the frames the router sends to
          the low-contrast branch, whereas always-AdaIR pays 333 ms on every
          frame.

HONEST SCOPE (printed again at the end; it belongs in the manuscript)
  * The mixture curve is arithmetic over two measured endpoints. It assumes the
    deployed clean frames resemble ChallengeFree and the deployed degraded frames
    resemble the CURE mixture with equal weight per challenge and severity. It is
    an illustration of the trade-off, not a field measurement.
  * Latency figures are per-image on the stated CPU and are reported with that
    scope, not as an embedded-hardware claim.
  * Everything here inherits the study's scope: one classifier (CompactCNN,
    145,291 parameters, 32x32), one synthetic-rendered benchmark.

READS   outputs_revision/merged_per_image.csv     (K_merge_results.py)
        outputs_revision/dcp_cure.csv             (Q_dcp_branch.py)
WRITES  outputs_revision/AD_deployment_arithmetic.json
        outputs_revision/AD_mixture_curve.csv     (for the Fig. 7 inset)
RUN     python AD_deployment_arithmetic.py        (seconds; pure lookup)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"

SEVS = (1, 2, 3, 4, 5)
CHS = tuple(range(1, 13))

# Measured latencies, per image, CPU, single thread (scope as stated in Table I).
LAT_ROUTER_MS = 0.2      # routing statistics + branch dispatch, upper bound
LAT_DCP_MS = 11.1
LAT_ADAIR_MS = 333.0

# Expected values, locked 2026-07-10 from the authoritative per-image files.
EXPECT = {
    "cf_passthrough": 80.77, "cf_va": 80.77, "cf_vb": 80.77,
    "cf_dcp": 79.36, "cf_adair": 79.51, "cf_cidnet": 80.10,
    "deg_passthrough": 54.49, "deg_va": 56.65, "deg_vb": 58.21,
    "deg_dcp": 58.89, "deg_adair": 57.78, "deg_cidnet": 56.31,
    "share_deg": 11.9, "share_cf": 0.0, "breakeven_pct": 32.7,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    M = list(csv.DictReader(open(args.merged, newline="", encoding="utf-8")))
    D = {}
    for r in csv.DictReader(open(args.dcp, newline="", encoding="utf-8")):
        D[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = int(r["pred_dcp"])

    ch = np.array([int(r["ch"]) for r in M])
    sev = np.array([int(r["sev"]) for r in M])
    tru = np.array([int(r["true"]) for r in M])
    branch = np.array([r["rule_branch"] for r in M])
    P = {k: np.array([int(r["pred_" + k]) for r in M])
         for k in ["passthrough", "gamma", "clahe", "stretch", "va_rule",
                   "adair", "cidnet"]}
    P["dcp"] = np.array([D[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))]
                         for r in M])
    # V-B: the low-contrast branch (CLAHE) is served by DCP; everything else is
    # bit-identical to the base rule. Zero new tunable parameters.
    P["vb"] = np.where(branch == "clahe", P["dcp"], P["va_rule"])

    m_cf = (ch == 0)
    print(f"[data] rows={len(M)}  clean(ChallengeFree)={int(m_cf.sum())}  "
          f"degraded={int((~m_cf).sum())}")

    def cf(op):
        return 100.0 * float(np.mean(P[op][m_cf] == tru[m_cf]))

    def deg(op):
        return float(np.mean([100.0 * np.mean(P[op][(ch == c) & (sev == s)] ==
                                              tru[(ch == c) & (sev == s)])
                              for c in CHS for s in SEVS]))

    METHODS = ["passthrough", "va_rule", "vb", "dcp", "adair", "cidnet"]
    A = {op: (cf(op), deg(op)) for op in METHODS}

    print("\n=== the two endpoints every beam is built from ===")
    print(f"{'method':14s}{'clean (CF)':>12s}{'degraded avg':>14s}")
    for op in METHODS:
        tag = "  <- selector" if op == "vb" else ""
        print(f"{op:14s}{A[op][0]:12.2f}{A[op][1]:14.2f}{tag}")

    # ---------------- BEAM 1: clean-traffic immunity ----------------
    share_deg = 100.0 * float(np.mean(branch[~m_cf] == "clahe"))
    share_cf = 100.0 * float(np.mean(branch[m_cf] == "clahe"))
    cf_branches = {b: 100.0 * float(np.mean(branch[m_cf] == b))
                   for b in ("passthrough", "gamma", "clahe", "stretch")}
    identical = bool(np.array_equal(P["vb"][m_cf], P["passthrough"][m_cf]))

    print("\n=== BEAM 1: clean-traffic immunity ===")
    print("  branch shares on the clean split: " +
          "  ".join(f"{b} {v:.2f}%" for b, v in cf_branches.items()))
    print(f"  V-B predictions on clean split are bit-identical to no enhancement: "
          f"{identical}")
    print(f"  clean accuracy   V-B {A['vb'][0]:.2f}   always-DCP {A['dcp'][0]:.2f}"
          f"   (cost of always enhancing: {A['vb'][0]-A['dcp'][0]:+.2f})")

    # ---------------- BEAM 2: mixture break-even ----------------
    d_cf = A["vb"][0] - A["dcp"][0]     # V-B advantage on clean  (positive)
    d_dg = A["dcp"][1] - A["vb"][1]     # DCP advantage on degraded (positive)
    w_star = d_dg / (d_cf + d_dg)

    print("\n=== BEAM 2: mixture break-even against always-DCP ===")
    print(f"  V-B gains {d_cf:+.2f} on clean frames and gives up {d_dg:.2f} on "
          f"degraded frames.")
    print(f"  Mixture accuracy = w * CF + (1 - w) * deg-avg, w = fraction of clean "
          f"frames.")
    print(f"  Break-even: w = {d_dg:.2f} / ({d_cf:.2f} + {d_dg:.2f}) = "
          f"{100*w_star:.1f}%  ->  V-B is ahead whenever clean frames exceed "
          f"{100*w_star:.1f}% of traffic.")
    print(f"\n  {'clean %':>8s}{'V-B':>9s}{'always-DCP':>12s}{'always-AdaIR':>14s}"
          f"{'V-B minus DCP':>15s}")
    curve = []
    for w in [i / 100.0 for i in range(0, 101, 10)]:
        mix = {op: w * A[op][0] + (1 - w) * A[op][1] for op in METHODS}
        curve.append({"clean_fraction": round(w, 2),
                      **{f"mix_{op}": round(mix[op], 2) for op in METHODS},
                      "vb_minus_dcp": round(mix["vb"] - mix["dcp"], 2),
                      "vb_minus_adair": round(mix["vb"] - mix["adair"], 2)})
        if int(w * 100) % 20 == 0:
            print(f"  {int(100*w):8d}{mix['vb']:9.2f}{mix['dcp']:12.2f}"
                  f"{mix['adair']:14.2f}{mix['vb']-mix['dcp']:+15.2f}")

    # ---------------- BEAM 3: strict dominance + latency ----------------
    print("\n=== BEAM 3: strict dominance and the latency bill ===")
    for rival in ("adair", "cidnet", "passthrough"):
        dc, dd = A["vb"][0] - A[rival][0], A["vb"][1] - A[rival][1]
        strict = dc > 0 and dd > 0
        print(f"  V-B vs always-{rival:11s}: clean {dc:+6.2f}  degraded {dd:+6.2f}  "
              f"-> {'STRICTLY DOMINATES at every mixture' if strict else 'mixture dependent'}")

    lat_vb_deg = LAT_ROUTER_MS + LAT_DCP_MS * share_deg / 100.0
    lat_vb_cf = LAT_ROUTER_MS + LAT_DCP_MS * share_cf / 100.0
    print(f"\n  the 11.1 ms operator is invoked on {share_deg:.1f}% of degraded "
          f"frames and {share_cf:.2f}% of clean frames")
    print(f"  average front-end latency   V-B: {lat_vb_deg:.2f} ms (degraded), "
          f"{lat_vb_cf:.2f} ms (clean)")
    print(f"                       always-DCP: {LAT_DCP_MS:.2f} ms on every frame")
    print(f"                     always-AdaIR: {LAT_ADAIR_MS:.1f} ms on every frame "
          f"({LAT_ADAIR_MS/max(lat_vb_deg,1e-9):.0f}x the selector on degraded traffic)")

    out = {"endpoints": {op: {"clean": round(A[op][0], 2), "degraded": round(A[op][1], 2)}
                         for op in METHODS},
           "beam1": {"clean_branch_shares_pct": {k: round(v, 2) for k, v in cf_branches.items()},
                     "vb_bit_identical_to_passthrough_on_clean": identical,
                     "cost_of_always_dcp_on_clean": round(A["dcp"][0] - A["vb"][0], 2)},
           "beam2": {"clean_gain": round(d_cf, 2), "degraded_loss": round(d_dg, 2),
                     "breakeven_clean_fraction_pct": round(100 * w_star, 1),
                     "convention": "mixture = w * CF + (1-w) * deg-avg; arithmetic "
                                   "corollary of Table II, not a new benchmark"},
           "beam3": {"dominates_adair_at_every_mixture":
                     bool(A["vb"][0] > A["adair"][0] and A["vb"][1] > A["adair"][1]),
                     "dcp_invocation_pct_degraded": round(share_deg, 1),
                     "dcp_invocation_pct_clean": round(share_cf, 2),
                     "avg_frontend_latency_ms": {"vb_degraded": round(lat_vb_deg, 2),
                                                 "vb_clean": round(lat_vb_cf, 2),
                                                 "always_dcp": LAT_DCP_MS,
                                                 "always_adair": LAT_ADAIR_MS}}}

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir) / "AD_deployment_arithmetic.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(Path(args.outdir) / "AD_mixture_curve.csv", "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        wr.writeheader()
        wr.writerows(curve)

    # ---------------- expected-output audit ----------------
    print("\n=== EXPECTED-OUTPUT AUDIT (against values locked 2026-07-10) ===")
    got = {f"cf_{op}": A[op][0] for op in METHODS}
    got.update({f"deg_{op}": A[op][1] for op in METHODS})
    got["cf_va"] = A["va_rule"][0]
    got["deg_va"] = A["va_rule"][1]
    got.update(share_deg=share_deg, share_cf=share_cf, breakeven_pct=100 * w_star)
    bad = 0
    for k, exp in EXPECT.items():
        if k not in got:
            continue
        tol = 0.15 if k in ("share_deg", "breakeven_pct") else 0.05
        if abs(got[k] - exp) > tol:
            print(f"  MISMATCH {k}: got {got[k]:.2f}, expected {exp:.2f}")
            bad += 1
    if not identical:
        print("  MISMATCH: V-B is not bit-identical to passthrough on the clean split; "
              "beam 1 does not hold on this data.")
        bad += 1
    print("  AUDIT PASSED: every value reproduces." if bad == 0
          else f"  AUDIT: {bad} mismatch(es) above. Send me the console output.")

    print("\nSCOPE, to be carried into the manuscript with the numbers:")
    print("  The mixture curve is arithmetic over two measured endpoints, not a field")
    print("  measurement; it assumes deployed clean frames resemble ChallengeFree and")
    print("  deployed degraded frames resemble the CURE mixture with equal weight per")
    print("  challenge and severity. Latency is per-image on the stated CPU. All of it")
    print("  inherits the single-classifier, single-benchmark scope of the study.")
    print(f"\nwrote AD_deployment_arithmetic.json / AD_mixture_curve.csv to {args.outdir}")


if __name__ == "__main__":
    main()
