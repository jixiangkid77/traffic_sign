# -*- coding: utf-8 -*-
"""
ZA_deep_vs_injection.py
Closes the one gap that could change C1's central claim, and fills in a
confidence interval that was promised and never delivered.

THE GAP
Z_injection_definitive.py tests the four training-free operators against the
noise-injection oracle. It does NOT test the learned restoration models. The question
a reviewer will certainly ask has therefore never been answered:

    does a LEARNED restoration model actually restore severe blur, or does its
    apparent benefit also amount to high-frequency injection?

The two answers point in opposite directions.
  If AdaIR does not beat the noise-injection oracle either, then no method in the pool
  restores information-loss degradation, which is a strong and clean statement
  of where the capability boundary lies.
  If AdaIR does beat it, then the learned model possesses a restoration ability
  the training-free operators lack, and C1 must say so plainly.

It costs nothing to answer. AdaIR and CIDNet were already evaluated on every
CURE image by J and are cached in merged_per_image.csv; the noise-injection oracle
comes from the noise scan that Z has already computed. The two are joined per
image, so every comparison is paired on identical images.

ALSO FILLED IN HERE
The bootstrap interval for the DCP row of the method table. Every other method
carries one (passthrough 54.49 [54.19, 54.80] and so on); DCP never got one.
It is computed with the main table's convention: one seeded generator, per-cell
draws shared across methods, B = 5000, seed 42, cell-averaged over the 60
(challenge, severity) cells.

Reads:  outputs_revision/Z_injection_per_image.csv   (from Z, full cells)
        outputs_revision/merged_per_image.csv        (AdaIR, CIDNet)
        outputs_revision/dcp_cure.csv                (DCP)
Writes: outputs_revision/ZA_deep_vs_injection.json
Run:    python ZA_deep_vs_injection.py          (seconds; no inference at all)

TERMINOLOGY NOTE (registered 2026-07-11, Evaluation Protocol Part 17.9)
  What this script computes is now called the NOISE-INJECTION ORACLE: the best
  accuracy reachable by adding Gaussian noise alone, with sigma chosen optimally
  on the test set. Earlier drafts called it the "noise-injection oracle", a name built by
  truncating the standard phrase "noise injection", which did not say what the
  oracle ranged over. It is an oracle over NOISE LEVELS only, so a restoration
  operator can and does exceed it; that is not a paradox. It is distinct from the
  OPERATOR-SELECTION ORACLE (o4 to o7), which is the best accuracy reachable if the
  best operator could be picked per image. The bare phrase "the oracle" must not
  appear where the two could be confused.

  The JSON field names (oracle_acc, vs_oracle, deep_vs_injection) and this file's
  name keep their original spelling ON PURPOSE: renaming them would force a re-run
  over 81,120 images for no benefit, and they never appear in the manuscript. This
  change is to comments and printed strings only. No computation is touched and no
  number changes.
"""
import argparse, csv, json, os, sys
from math import comb, erfc, sqrt
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
CONTRAST_SRC = (12, 4)              # Haze, Darkening (the validity check)
SEVERITIES = (1, 2, 3, 4, 5)
CLASSICAL = ("gamma", "clahe", "stretch", "dcp")
DEEP = ("adair", "cidnet", "zero_dce", "ffa_net", "promptir")
SIGMAS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
CONTRAST_MIN_GAIN = 5.0
B, SEED = 5000, 42
NAMES = {1: "Decolorization", 2: "LensBlur", 3: "CodecError",
         4: "Darkening", 5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur",
         8: "Noise", 9: "Rain", 10: "Shadow", 11: "Snow", 12: "Haze"}


def mcn(a, b_, t):
    x = int(np.sum((a == t) & (b_ != t)))
    y = int(np.sum((a != t) & (b_ == t)))
    n = x + y
    if n == 0:
        return 1.0
    if n <= 1000:
        return float(min(1.0, 2 * sum(comb(n, i)
                                      for i in range(min(x, y) + 1)) / 2 ** n))
    return float(erfc(abs(x - y) / sqrt(n) / sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zfile", default=str(OUT_DIR / "Z_injection_per_image.csv"),
                    help="Z's per-image output; the challenge set is derived "
                         "from it, so no separate flag is needed")
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    Dc = {}
    with open(args.dcp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Dc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = int(r["pred_dcp"])
    Mrows = list(csv.DictReader(open(args.merged, newline="",
                                     encoding="utf-8")))
    Mc = {(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"])): r
          for r in Mrows}
    Z = list(csv.DictReader(open(args.zfile, newline="", encoding="utf-8")))
    print(f"loaded Z={len(Z)} merged={len(Mrows)} dcp={len(Dc)}")

    # ---------------- part 1: the deep models against the oracle -------------
    keys = [(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
            for r in Z]
    miss = sum(1 for k in keys if k not in Mc)
    if miss:
        print(f"[FATAL] {miss} of Z's rows are absent from merged; "
              f"the join is unsound. Stopping.")
        return
    ch = np.array([k[2] for k in keys])
    challenges = tuple(sorted(set(int(x) for x in ch)))
    print(f"[scope] challenge set derived from Z: "
          f"{[NAMES.get(c, c) for c in challenges]}")
    sv = np.array([k[3] for k in keys])
    tr = np.array([int(r["true"]) for r in Z])
    # sanity: the true labels must agree between the two files
    tr_m = np.array([int(Mc[k]["true"]) for k in keys])
    if not np.array_equal(tr, tr_m):
        print("[FATAL] the true labels disagree between Z and merged. "
              "Stopping.")
        return
    print("[join] keys and labels agree on every row")

    PR = {"raw": np.array([int(r["pred_raw"]) for r in Z])}
    for o in CLASSICAL:
        # The dark channel prior is taken from its own file, not from the
        # injection sweep's copy of it. The sweep was run before the prior was
        # refined and its pred_dcp column still holds the older one, which
        # scores 58.89 against the 60.72 every other figure and table in the
        # article reports. Reading the sweep's column here would give this
        # figure a different prior from the rest of the paper.
        if o == "dcp":
            PR[o] = np.array([Dc[k] for k in keys])
        else:
            PR[o] = np.array([int(r[f"pred_{o}"]) for r in Z])
    for s in SIGMAS:
        if s:
            PR[f"n{s}"] = np.array([int(r[f"pred_n{s}"]) for r in Z])
    # All five learned restorers, not two. The three that used to be missing
    # were scored in a separate pass whose rows could not be matched to these;
    # they are in the merged file now, so the comparison against the
    # noise-injection oracle can include them and "the best learned restorer"
    # means the best of five rather than the best of whichever two were aligned.
    for _m in ("adair", "cidnet", "zero_dce", "ffa_net", "promptir"):
        col = f"pred_{_m}"
        if col not in next(iter(Mc.values())):
            sys.exit(f"[FATAL] {col} missing from the merged file. Run "
                     f"K_merge_results.py on a set that includes it.")
        PR[_m] = np.array([int(Mc[k][col]) for k in keys])

    # audit: Z's classical predictions must equal the caches (Z checks this too,
    # but the join is what this script depends on, so it is re-checked here)
    # dcp is deliberately absent: it is read from its own file above, so a
    # comparison with the sweep's stale column would fail by construction. The
    # size of that difference is printed instead, so it cannot pass unnoticed.
    CACHE = {"raw": "pred_passthrough", "gamma": "pred_gamma",
             "clahe": "pred_clahe", "stretch": "pred_stretch"}
    _stale = sum(1 for i, r in enumerate(Z)
                 if int(r["pred_dcp"]) != PR["dcp"][i])
    print(f"[note] the sweep's own pred_dcp differs from the refined prior on "
          f"{_stale:,} of {len(Z):,} rows ({100 * _stale / len(Z):.1f} per "
          f"cent); the refined one is used, as everywhere else.")
    bad = 0
    for i, k in enumerate(keys):
        for tag, col in CACHE.items():
            bad += int(PR[tag][i] != int(Mc[k][col]))
        bad += int(PR["dcp"][i] != Dc[k])
    print(f"[audit] Z-vs-cache disagreements across 5 operators: {bad} "
          f"({'PASS' if bad == 0 else 'FAIL -- do not use these numbers'})")
    if bad:
        return

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

    # contrast criterion, from the frozen cache alone (identical to Z)
    def cache_acc(c, v, tag):
        ks = [k for k in Mc if k[2] == c and k[3] == v]
        t = np.array([int(Mc[k]["true"]) for k in ks])
        if tag == "dcp":
            p = np.array([Dc[k] for k in ks])
        else:
            p = np.array([int(Mc[k][CACHE[tag]]) for k in ks])
        return 100 * float(np.mean(p == t))

    contrasts = {}
    for v in SEVERITIES:
        q = []
        for c in CONTRAST_SRC:
            base = cache_acc(c, v, "raw")
            for op in CLASSICAL:
                if cache_acc(c, v, op) - base >= CONTRAST_MIN_GAIN:
                    q.append((c, op))
        contrasts[v] = q

    out = {"deep_vs_injection": {}, "dcp_table_ci": {}}
    print("\n" + "=" * 78)
    print("DO THE LEARNED MODELS RESTORE, OR DO THEY ALSO ONLY INJECT?")
    print("  (full cells; the oracle sigma is taken from Z's noise scan)")
    print("=" * 78)
    for c in challenges:
        print(f"\n  {NAMES[c]}")
        for v in SEVERITIES:
            a_raw = acc("raw", c, v)
            scan = {s: acc(f"n{s}", c, v) for s in SIGMAS if s}
            scan[0] = a_raw
            bs = max(scan, key=scan.get)
            a_orc = scan[bs]
            otag = "raw" if bs == 0 else f"n{bs}"
            valid = len(contrasts[v]) > 0
            print(f"    sev{v}: raw {a_raw:5.2f} | noise-injection oracle "
                  f"{a_orc:5.2f} (sigma={bs})"
                  f"{'' if valid else '   [no valid contrast: no verdict]'}")
            cell = {"raw": round(a_raw, 2), "oracle_sigma": bs,
                    "oracle_acc": round(a_orc, 2), "validated": bool(valid),
                    "ops": {}}
            for op in CLASSICAL + DEEP:
                a_op = acc(op, c, v)
                lo, hi = boot(op, otag, c, v)
                m = (ch == c) & (sv == v)
                p = mcn(PR[op][m], PR[otag][m], tr[m])
                if not valid:
                    verdict = "no verdict"
                elif lo > 0:
                    verdict = "RESTORES beyond injection"
                elif hi < 0:
                    verdict = "WORSE than best injection"
                else:
                    verdict = "NOT above injection"
                cell["ops"][op] = {"acc": round(a_op, 2),
                                    "vs_oracle": round(a_op - a_orc, 2),
                                    "lo": round(lo, 2), "hi": round(hi, 2),
                                    "p": p, "verdict": verdict}
                star = "  <<< LEARNED" if op in DEEP else ""
                print(f"        {op:8s} {a_op:6.2f}  vs oracle "
                      f"{a_op-a_orc:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  "
                      f"{verdict}{star}")
            out["deep_vs_injection"][f"{NAMES[c]}_sev{v}"] = cell

    # ---------------- part 2: the missing DCP interval ----------------------
    print("\n" + "=" * 78)
    print("THE DCP ROW OF THE METHOD TABLE (the interval that was never "
          "computed)")
    print("  main-table convention: one generator, per-cell draws shared "
          "across methods")
    print("=" * 78)
    CELLS = [(c, v) for c in range(1, 13) for v in range(1, 6)]
    cell_idx = {}
    for c, v in CELLS:
        ks = [k for k in Mc if k[2] == c and k[3] == v]
        cell_idx[(c, v)] = ks
    rng = np.random.default_rng(SEED)
    # Every front end the accuracy table lists, so the table has one source and
    # not two. The three learned restorers that used to be missing were carried
    # in the table from an earlier run, and stayed there after they were rerun
    # here: the wording about that run was deleted from the article, the numbers
    # were not. Selector V-B is included for the same reason.
    meth = ["passthrough", "gamma", "clahe", "stretch", "va_rule", "vb",
            "cidnet", "adair", "zero_dce", "ffa_net", "promptir", "dcp"]
    accs = {m: np.zeros(B) for m in meth}
    point = {m: [] for m in meth}
    for c, v in CELLS:
        ks = cell_idx[(c, v)]
        t = np.array([int(Mc[k]["true"]) for k in ks])
        preds = {}
        for m in meth:
            if m == "dcp":
                preds[m] = np.array([Dc[k] for k in ks])
            elif m == "vb":
                # The selector is the rule with the prior on the CLAHE branch.
                preds[m] = np.array(
                    [Dc[k] if Mc[k]["rule_branch"] == "clahe"
                     else int(Mc[k]["pred_va_rule"]) for k in ks])
            else:
                preds[m] = np.array([int(Mc[k][f"pred_{m}"]) for k in ks])
            point[m].append(100 * float(np.mean(preds[m] == t)))
        idx = rng.integers(0, len(t), (B, len(t)))
        tt = t[idx]
        for m in meth:
            accs[m] += 100.0 * (preds[m][idx] == tt).mean(axis=1)
    print(f"\n  {'method':14s}{'deg-acc':>9s}{'95% CI':>20s}")
    for m in meth:
        a = accs[m] / len(CELLS)
        lo, hi = np.percentile(a, [2.5, 97.5])
        pt = float(np.mean(point[m]))
        out["dcp_table_ci"][m] = {"deg_acc": round(pt, 2),
                                   "lo": round(float(lo), 2),
                                   "hi": round(float(hi), 2)}
        star = "   <<< the missing row" if m == "dcp" else ""
        print(f"  {m:14s}{pt:9.2f}   [{lo:5.2f}, {hi:5.2f}]{star}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "ZA_deep_vs_injection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {Path(args.outdir) / 'ZA_deep_vs_injection.json'}")
    print("\nREADING GUIDE\n"
          "  If AdaIR does not beat the noise-injection oracle on the blur cells "
          "either, then no method\n  in the pool restores information-loss "
          "degradation, and the boundary says so. If it\n  does, the learned "
          "model has a capability the training-free operators lack, and C1 "
          "must\n  state that plainly.")


if __name__ == "__main__":
    main()
