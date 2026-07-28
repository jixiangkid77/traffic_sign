# -*- coding: utf-8 -*-
"""
AC_winner_margins.py
Post-hoc analysis (Evaluation Protocol, Part 17.6; declared POST HOC, not
pre-registered: the question was raised on 2026-07-10 after the boundary results
were already known).

WHAT THIS ANSWERS
Section IV-A claims operator heterogeneity: "twelve challenges, six different
winning operators". A winner is an argmax, and an argmax is a descriptive
statistic, not evidence. A reviewer will ask whether the winner is actually
better than the runner-up, or whether the ranking is noise. This script answers
that, and it also computes the one heterogeneity measure that needs no ranking
at all: the oracle gap.

THREE OUTPUTS
  1. Per challenge: winner, runner-up, margin, paired bootstrap 95% CI, verdict.
     A challenge whose CI includes zero is a STATISTICAL TIE and must be
     labelled as such in the manuscript. The claim in the text becomes
     "six different operators attain the maximum, and in N of twelve the maximum
     is significantly above the runner-up".
  2. The oracle gap: acc(oracle over the 7-operator pool) minus acc(best single
     operator). This is the assumption-free measure of heterogeneity: if one
     operator were universally best, the gap would be zero.
  3. A wins matrix (which operator attains the maximum on which challenge),
     written to CSV for the Fig. 1 winner-distribution panel.

BOOTSTRAP CONVENTION (identical to K_merge_results.py; do not change)
  Per-cell paired resampling with a single rng: for each of the B replicates and
  for each (challenge, severity) cell, draw one index vector and apply it to BOTH
  methods, so the pairing is preserved. Cell accuracies are then averaged with
  equal weight (5 severities), matching how deg-avg is defined everywhere else in
  this study. B = 5000, seed = 42.

  NOTE ON MULTIPLICITY: twelve winner-vs-runner-up tests are reported. They are
  NOT Holm corrected, because they are descriptive companions to a table rather
  than a family of confirmatory hypotheses; the manuscript reports the raw CIs
  and says so. Holm correction IS applied, separately, to the 288 restorability
  tests in Z_all12/ZA, which are the confirmatory family.

READS   outputs_revision/merged_per_image.csv     (K_merge_results.py)
        outputs_revision/dcp_cure.csv             (Q_dcp_branch.py)
WRITES  outputs_revision/AC_winner_margins.csv
        outputs_revision/AC_winner_margins.json
RUN     python AC_winner_margins.py               (about 1 to 2 minutes, CPU)
        python AC_winner_margins.py --boot 2000   (faster pass)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"

OPS = ["passthrough", "gamma", "clahe", "stretch", "dcp", "adair", "cidnet"]
FAMILY = {"passthrough": "training-free", "gamma": "training-free",
          "clahe": "training-free", "stretch": "training-free",
          "dcp": "training-free", "adair": "learned", "cidnet": "learned"}
NAMES = {1: "Decolorization", 2: "LensBlur", 3: "CodecError", 4: "Darkening",
         5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur", 8: "Noise",
         9: "Rain", 10: "Shadow", 11: "Snow", 12: "Haze"}
SEVS = (1, 2, 3, 4, 5)

# Expected values, computed 2026-07-10 from the locked per-image files.
# Any deviation means the inputs are not the authoritative ones.
EXPECT_ROWS = 82472
EXPECT_CELL = 1352
EXPECT_WINNER = {
    1: ("cidnet", 72.0), 2: ("clahe", 48.9), 3: ("gamma", 53.4),
    4: ("gamma", 76.1), 5: ("adair", 78.1), 6: ("passthrough", 54.3),
    7: ("clahe", 41.3), 8: ("adair", 54.5), 9: ("dcp", 45.7),
    10: ("cidnet", 76.6), 11: ("cidnet", 68.4), 12: ("dcp", 72.2),
}
# Locked at the standard convention B = 5000, seed = 42. Darkening is a genuine
# borderline case: its margin is +0.80 with CI [-0.01, +1.61] at B = 5000 (tie),
# but [+0.03, +1.66] at B = 2000 (significant). It is reported as a tie, because
# B = 5000 is the convention used everywhere else in this study. The manuscript
# reports the CI, not only the label.
EXPECT_TIES = {"Darkening", "DirtyLens", "Exposure", "Shadow"}   # CI includes zero
EXPECT_N_SIG = 8
EXPECT_ORACLE7 = 72.17
EXPECT_BEST_SINGLE = 58.89                          # always-DCP


def load(merged, dcp_csv):
    M = list(csv.DictReader(open(merged, newline="", encoding="utf-8")))
    D = {}
    for r in csv.DictReader(open(dcp_csv, newline="", encoding="utf-8")):
        D[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = int(r["pred_dcp"])
    if len(M) != EXPECT_ROWS:
        print(f"  WARNING: merged has {len(M)} rows, expected {EXPECT_ROWS}")
    miss = [k for k in ((r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
                        for r in M) if k not in D]
    if miss:
        raise SystemExit(f"ABORT: {len(miss)} merged rows have no DCP prediction; "
                         f"the two files are not aligned.")
    ch = np.array([int(r["ch"]) for r in M])
    sev = np.array([int(r["sev"]) for r in M])
    tru = np.array([int(r["true"]) for r in M])
    P = {k: np.array([int(r["pred_" + k]) for r in M])
         for k in ["passthrough", "gamma", "clahe", "stretch", "adair", "cidnet"]}
    P["dcp"] = np.array([D[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))]
                         for r in M])
    return ch, sev, tru, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ch, sev, tru, P = load(args.merged, args.dcp)
    print(f"[data] rows={len(ch)}  operators={len(OPS)}  B={args.boot} seed={args.seed}")

    cells = {c: [(ch == c) & (sev == s) for s in SEVS] for c in NAMES}
    n_bad = [(c, s) for c in NAMES for s, m in zip(SEVS, cells[c]) if m.sum() != EXPECT_CELL]
    if n_bad:
        print(f"  WARNING: {len(n_bad)} cells do not have {EXPECT_CELL} images: {n_bad[:5]}")

    def acc(op, c):
        return 100.0 * float(np.mean([np.mean(P[op][m] == tru[m]) for m in cells[c]]))

    rng = np.random.default_rng(args.seed)
    rows, out = [], {"per_challenge": {}, "convention": "per-cell paired bootstrap, "
                     "single rng, 5 severities equally weighted; NOT Holm corrected "
                     "(descriptive companion, see header)"}
    n_sig = 0

    print("\n=== winner vs runner-up (paired bootstrap 95% CI) ===")
    print(f"{'challenge':16s}{'winner':>12s}{'acc':>7s}{'runner-up':>12s}{'acc':>7s}"
          f"{'margin':>8s}{'95% CI':>18s}  verdict")
    for c in sorted(NAMES):
        a = {op: acc(op, c) for op in OPS}
        order = sorted(a, key=a.get, reverse=True)
        w, r = order[0], order[1]

        A = [(P[w][m] == tru[m]).astype(float) for m in cells[c]]
        B = [(P[r][m] == tru[m]).astype(float) for m in cells[c]]
        d = np.empty(args.boot)
        for i in range(args.boot):
            sa = sb = 0.0
            for x, y in zip(A, B):
                j = rng.integers(0, len(x), len(x))
                sa += x[j].mean()
                sb += y[j].mean()
            d[i] = 100.0 * (sa - sb) / len(A)
        lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
        sig = lo > 0.0
        n_sig += sig
        verdict = "SIGNIFICANT" if sig else "STATISTICAL TIE"

        print(f"{NAMES[c]:16s}{w:>12s}{a[w]:7.1f}{r:>12s}{a[r]:7.1f}"
              f"{a[w]-a[r]:+8.2f}  [{lo:+6.2f},{hi:+6.2f}]  {verdict}")
        rows.append({"challenge": NAMES[c], "winner": w, "winner_family": FAMILY[w],
                     "winner_acc": round(a[w], 2), "runner_up": r,
                     "runner_up_acc": round(a[r], 2),
                     "margin": round(a[w] - a[r], 2), "ci_lo": round(lo, 2),
                     "ci_hi": round(hi, 2), "significant": int(sig),
                     **{f"acc_{op}": round(a[op], 2) for op in OPS}})
        out["per_challenge"][NAMES[c]] = rows[-1]

    winners = sorted({r["winner"] for r in rows})
    ties = sorted(r["challenge"] for r in rows if not r["significant"])
    out.update(n_distinct_winners=len(winners), winners=winners,
               n_significant=int(n_sig), ties=ties)

    # ---- oracle gap: heterogeneity without any ranking ----
    def oracle(pool, c):
        vals = []
        for m in cells[c]:
            ok = np.zeros(int(m.sum()), bool)
            for op in pool:
                ok |= (P[op][m] == tru[m])
            vals.append(ok.mean())
        return 100.0 * float(np.mean(vals))

    o7 = float(np.mean([oracle(OPS, c) for c in NAMES]))
    single = {op: float(np.mean([acc(op, c) for c in NAMES])) for op in OPS}
    best_op = max(single, key=single.get)
    gap = o7 - single[best_op]
    out.update(oracle7=round(o7, 2), best_single=best_op,
               best_single_acc=round(single[best_op], 2), oracle_gap=round(gap, 2))

    print(f"\n=== heterogeneity without ranking ===")
    print(f"  oracle over all 7 operators        : {o7:.2f}")
    print(f"  best single operator ({best_op:^11s}) : {single[best_op]:.2f}")
    print(f"  oracle gap                         : {gap:.2f}"
          f"   (zero if one operator were universally best)")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir) / "AC_winner_margins.csv", "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(Path(args.outdir) / "AC_winner_margins.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- expected-output audit ----
    print("\n=== EXPECTED-OUTPUT AUDIT (against values locked 2026-07-10) ===")
    bad = 0
    for c in sorted(NAMES):
        ew, ea = EXPECT_WINNER[c]
        gw, ga = out["per_challenge"][NAMES[c]]["winner"], \
            out["per_challenge"][NAMES[c]]["winner_acc"]
        if gw != ew or abs(ga - ea) > 0.1:
            print(f"  MISMATCH {NAMES[c]}: got {gw} {ga:.1f}, expected {ew} {ea:.1f}")
            bad += 1
    if len(winners) != 6:
        print(f"  MISMATCH distinct winners: got {len(winners)}, expected 6")
        bad += 1
    got_ties = set(ties)
    if got_ties != EXPECT_TIES:
        print(f"  NOTE ties differ: got {sorted(got_ties)}, expected {sorted(EXPECT_TIES)}")
        print( "       (borderline challenges can flip with a different B; if the only"
               " difference\n        is a borderline one, report the CI, not the label)")
        bad += 1
    for nm, got, exp in [("oracle7", o7, EXPECT_ORACLE7),
                         ("best single", single[best_op], EXPECT_BEST_SINGLE)]:
        if abs(got - exp) > 0.05:
            print(f"  MISMATCH {nm}: got {got:.2f}, expected {exp:.2f}")
            bad += 1
    if n_sig != EXPECT_N_SIG:
        print(f"  MISMATCH n_significant: got {n_sig}, expected {EXPECT_N_SIG}")
        bad += 1
    print("  AUDIT PASSED: every value reproduces." if bad == 0
          else f"  AUDIT: {bad} mismatch(es) above. Send me the console output.")

    print(f"\n[summary] {len(winners)} distinct winners across 12 challenges; "
          f"{n_sig} of 12 significantly above the runner-up; "
          f"{len(ties)} statistical tie(s): {', '.join(ties) if ties else 'none'}")
    print("\nMANUSCRIPT WORDING (use verbatim, it is the only claim this supports):")
    print("  Six different operators attain the maximum across the twelve challenges,")
    print(f"  and in {n_sig} of the twelve the maximum is significantly above the")
    print("  runner-up; the remaining challenges are statistical ties and are marked")
    print("  as such in Table III. Attaining the maximum is not evidence of")
    print("  restoration; see Section IV-B.")
    print("\n  The assumption-free companion (no ranking involved):")
    print(f"  An oracle over the seven-operator pool reaches {o7:.2f}, against "
          f"{single[best_op]:.2f}")
    print(f"  for the best single operator, a gap of {gap:.1f} points. If one "
          "operator were")
    print("  universally best, that gap would be zero.")
    print("  (Report the gap to one decimal: subtracting the rounded table values "
          "gives\n   a last-digit artefact.)")
    print(f"\nwrote AC_winner_margins.csv / .json to {args.outdir}")


if __name__ == "__main__":
    main()
