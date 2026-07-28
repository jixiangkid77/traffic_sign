# -*- coding: utf-8 -*-
r"""
TABLES_main.py
The three tables the article prints, computed from the files that hold the
results rather than assembled by hand.

WHY THIS SCRIPT EXISTS IN THIS FORM
It used to emit four tables as IEEEtran source, and the article no longer reads
that source: the tables are rendered by build_sec3.js and build_sec5.js from two
JSON files. Those two files were being maintained by hand, and by 2026-07-26
they had drifted in four places at once:

  * _tabledata.json carried a whole second copy of Table I, still holding the
    latencies from before the three learned restorers were timed here, with
    "not timed here" in cells that now have measurements.
  * The same file gave the classifier 1.277 ms where the other gave 1.21, which
    is the number the article prints three times.
  * clahe_worst was 7 after the pool grew to twelve and the count became 6.
  * Table II's intervals came from a computation nothing in the project
    reproduced; they disagreed with ZA_deep_vs_injection.json, which is the
    script that computes intervals, in eleven of twelve rows.

None of that was visible in the rendered document. It was visible only by
computing the tables again and comparing, which is what this script now does
every time it runs.

WHY THERE IS ONE ACCURACY TABLE AND NOT TWO
The earlier version built two, because Zero-DCE, FFA-Net and PromptIR had been
scored in a separate run that covered five challenges, and putting a
five-challenge average beside a twelve-challenge one would have compared
different things. O_eval_three_locally.py scored those three here, on the same
crops as everything else, over all twelve challenges. The reason for the second
table is gone, and so is the table.

WHAT EACH NUMBER COMES FROM
  Table I    parameter counts from the weight files, latency from
             N_timing_stable, which times each front end in two blocks and
             reports the mean of the two.
  Table II   degraded average and its 95 per cent interval from
             ZA_deep_vs_injection.json, which resamples images inside their own
             (challenge, severity) cell with B = 5000 at seed 42 and shares the
             draws across front ends. Clean-image change and latency computed
             here.
  Table III  per-challenge degraded accuracy, averaged over severities 1 to 5,
             computed here from the per-image files.

The dark channel prior is not in merged_per_image.csv; Q_dcp_branch.py writes it
to dcp_cure.csv, and this script joins the two on (filename, occurrence). That
split is an accident of when the two were written. The article treats the prior
as one of four training-free operators and so does this script.

READS   outputs_revision/merged_per_image.csv          (K_merge_results.py)
        outputs_revision/dcp_cure.csv                  (Q_dcp_branch.py)
        outputs_revision/ZA_deep_vs_injection.json     (intervals)
        outputs_revision/N_timing_stable.results.json  (latency)
        models/*                                       (parameter counts)
WRITES  _table1.json      Table I, read by build_sec3.js
        _tabledata.json   Tables II and III, read by build_sec5.js
RUN     python TABLES_main.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
DOC_DIR = PROJECT_ROOT / "doc"

# The twelve front ends, in the order Table I prints them: no enhancement, the
# four training-free operators, the two routers, then the five learned
# restorers.
T1_ORDER = ["passthrough", "gamma", "clahe", "stretch", "dcp",
            "va_rule", "vb", "cidnet", "adair", "zero_dce", "ffa_net",
            "promptir"]
PRETTY = {"passthrough": "no enhancement", "gamma": "Gamma", "clahe": "CLAHE",
          "stretch": "Stretch", "dcp": "DCP", "va_rule": "base rule",
          "vb": "selector V-B", "cidnet": "CIDNet", "adair": "AdaIR",
          "zero_dce": "Zero-DCE", "ffa_net": "FFA-Net",
          "promptir": "PromptIR"}
FAMILY = {"passthrough": "none",
          "gamma": "training-free", "clahe": "training-free",
          "stretch": "training-free", "dcp": "training-free",
          "va_rule": "training-free", "vb": "training-free",
          "cidnet": "learned", "adair": "learned", "zero_dce": "learned",
          "ffa_net": "learned", "promptir": "learned"}
# The two routers are training-free but are not operators. Table I says so in
# the column beside them rather than leaving them among the four.
ROUTER = {"va_rule", "vb"}
TIMING_KEY = {"va_rule": "rule"}

CH_NAME = {1: "Decolorization", 2: "LensBlur", 3: "CodecError", 4: "Darkening",
           5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur", 8: "Noise",
           9: "Rain", 10: "Shadow", 11: "Snow", 12: "Haze"}
T3_HEAD = {"passthrough": "none", "va_rule": "rule", "vb": "V-B"}

MINUS = "\u2212"


def load_predictions(merged_path, dcp_path):
    """Every front end's per-image verdict, on one set of rows.

    The prior and the selector are built here rather than read: the prior lives
    in its own file, and the selector is the rule with the prior on the CLAHE
    branch, which is a composition and not a separate run.
    """
    rows = list(csv.DictReader(open(merged_path, newline="", encoding="utf-8")))
    dcp = {}
    for r in csv.DictReader(open(dcp_path, newline="", encoding="utf-8")):
        dcp[(r["filename"], int(r["occ"]))] = int(r["pred_dcp"])
    if len(dcp) != len(rows):
        raise SystemExit(f"ABORT: {len(rows):,} merged rows against "
                         f"{len(dcp):,} prior rows. They must be one set.")
    true = np.array([int(r["true"]) for r in rows])
    branch = np.array([r["rule_branch"] for r in rows])
    hit = {}
    for k in T1_ORDER:
        if k == "vb":
            continue
        if k == "dcp":
            p = np.array([dcp[(r["filename"], int(r["occ"]))] for r in rows])
        else:
            col = f"pred_{k}"
            if col not in rows[0]:
                raise SystemExit(f"ABORT: {col} missing from {merged_path}.")
            p = np.array([int(r[col]) for r in rows])
        hit[k] = p == true
    hit["vb"] = np.where(branch == "clahe", hit["dcp"], hit["va_rule"])
    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    return hit, ch, sev


def cell_average(ok, ch, sev, challenges=tuple(range(1, 13))):
    """The degraded average: a mean over cells, not over images.

    Cells hold different numbers of crops, and a mean over images would let the
    larger cells set the answer.
    """
    cells = [np.where((ch == c) & (sev == s))[0]
             for c in challenges for s in range(1, 6)]
    return 100.0 * float(np.mean([ok[i].mean() for i in cells]))


def count_params(models_dir):
    """Parameter counts, read from the weight files that were actually run."""
    import torch
    out = {}
    for key, pat in [("cidnet", "*cidnet*"), ("adair", "adair*"),
                     ("zero_dce", "*zero*dce*"), ("ffa_net", "*ffa*"),
                     ("promptir", "*promptir*")]:
        hits = sorted(Path(models_dir).glob(pat))
        if not hits:
            out[key] = None
            continue
        ck = torch.load(str(hits[0]), map_location="cpu")
        for k in ("state_dict", "params", "model", "net"):
            if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
                ck = ck[k]
                break
        out[key] = int(sum(v.numel() for v in ck.values()
                           if hasattr(v, "numel")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--za", default=str(OUT_DIR / "ZA_deep_vs_injection.json"))
    ap.add_argument("--timing",
                    default=str(OUT_DIR / "N_timing_stable.results.json"))
    ap.add_argument("--models", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--outdir", default=str(DOC_DIR))
    ap.add_argument("--params-json", default="",
                    help="parameter counts, if torch is not available here")
    args = ap.parse_args()

    for p in (args.merged, args.dcp, args.za, args.timing):
        if not Path(p).is_file():
            raise SystemExit(f"ABORT: {p} not found.")

    hit, ch, sev = load_predictions(args.merged, args.dcp)
    za = json.load(open(args.za))["dcp_table_ci"]
    tm = json.load(open(args.timing))
    clf_raw = float(tm["classifier_ms"])
    clf_ms = round(clf_raw, 2)
    ms, rel_ms = {}, {}
    for k in T1_ORDER:
        if k == "passthrough":
            continue
        key = TIMING_KEY.get(k, k)
        # The measurements sit under "final"; the top level holds the protocol.
        if key not in tm["final"]:
            raise SystemExit(f"ABORT: {key} missing from {args.timing}.")
        blk = tm["final"][key]
        # Two numbers, two jobs. quote_ms is the figure N_timing_stable decided
        # is the one to print, and it is what the article quotes in prose, so
        # the table has to carry the same one or the two disagree in the same
        # sentence. The ratio column divides the mean of the two blocks by the
        # classifier's own unrounded time, because a ratio of two rounded
        # numbers loses a digit that the ratio is made of.
        ms[k] = float(blk["quote_ms"])
        rel_ms[k] = (blk["block1_ms"] + blk["block2_ms"]) / 2.0
    params = (json.load(open(args.params_json)) if args.params_json
              else count_params(args.models))
    clean_idx = np.where(sev == 0)[0]
    base_clean = 100.0 * float(hit["passthrough"][clean_idx].mean())

    print(f"  {len(hit['passthrough']):,} rows, {len(T1_ORDER)} front ends, "
          f"classifier {clf_ms} ms")

    print("\n=== TABLE I ===")
    t1 = []
    for k in T1_ORDER:
        if k == "passthrough":
            t1.append({"name": PRETTY[k], "fam": "none", "params": "0",
                       "ms": "0", "rel": ""})
            print(f"  {PRETTY[k]:16s} {'none':14s} {'0':>12s}")
            continue
        n = params.get(k)
        rel = f"{rel_ms[k] / clf_raw:.2f}\u00d7"
        fam = FAMILY[k] + (" router" if k in ROUTER else "")
        t1.append({"name": PRETTY[k], "fam": fam,
                   "params": f"{n:,}" if n else "0",
                   # str() of the parsed float gives back what quote_ms was written
                   # as; ":g" would turn 118.0 into 118 and the table
                   # would stop matching the prose.
                   "ms": str(ms[k]), "rel": rel})
        print(f"  {PRETTY[k]:16s} {fam:21s} "
              f"{(f'{n:,}' if n else '0'):>12s} {ms[k]:9g} {rel}")

    print("\n=== TABLE II ===")
    t2 = []
    for k in sorted(za, key=lambda x: -za[x]["deg_acc"]):
        v = za[k]
        got = cell_average(hit[k], ch, sev)
        # The intervals come from ZA and the point estimates are recomputed
        # here. If the two disagree, one of the files is from a different run
        # and the table would carry a number no single computation produced.
        if abs(got - v["deg_acc"]) > 0.005:
            raise SystemExit(
                f"ABORT: {k} reads {got:.4f} here and {v['deg_acc']:.2f} in "
                f"{args.za}. The interval file and the per-image files "
                f"disagree; one of them is stale.")
        d = 100.0 * float(hit[k][clean_idx].mean()) - base_clean
        t2.append({"name": PRETTY[k], "acc": f"{v['deg_acc']:.2f}",
                   "ci": f"[{v['lo']:.2f}, {v['hi']:.2f}]",
                   "clean": ("0.00" if abs(d) < 0.005 else
                             f"{MINUS if d < 0 else ''}{abs(d):.2f}"),
                   "ms": "0" if k == "passthrough" else str(ms[k])})
        print(f"  {PRETTY[k]:16s} {v['deg_acc']:6.2f} "
              f"[{v['lo']:5.2f}, {v['hi']:5.2f}]  {t2[-1]['clean']:>7s}")
    if len(t2) != len(T1_ORDER):
        raise SystemExit(f"ABORT: {args.za} carries {len(t2)} front ends and "
                         f"Table I has {len(T1_ORDER)}.")

    print("\n=== TABLE III ===")
    t4 = []
    for c in range(1, 13):
        vals = [cell_average(hit[k], ch, sev, [c]) for k in T1_ORDER]
        mx = max(vals)
        t4.append({"name": CH_NAME[c], "vals": [f"{v:.2f}" for v in vals],
                   "bests": [i for i, v in enumerate(vals)
                             if abs(v - mx) < 1e-9]})
        win = ", ".join(T3_HEAD.get(T1_ORDER[i], PRETTY[T1_ORDER[i]])
                        for i in t4[-1]["bests"])
        print(f"  {CH_NAME[c]:16s} best {win}")

    # The counts the article quotes from this table, computed rather than typed.
    winners = {i for r in t4 for i in r["bests"]}
    clahe_i = T1_ORDER.index("clahe")
    clahe_worst = sum(1 for r in t4
                      if min(range(len(r["vals"])),
                             key=lambda j: float(r["vals"][j])) == clahe_i)
    print(f"\n  front ends taking first place somewhere: {len(winners)} of "
          f"{len(T1_ORDER)}")
    print(f"  rows on which CLAHE is worst: {clahe_worst} of {len(t4)}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": t1, "clf_ms": clf_ms, "clf_par": 145291},
              open(out / "_table1.json", "w"), ensure_ascii=False)
    # One copy of Table I, in _table1.json. The second copy this file used to
    # carry is not written: it drifted, and a table that exists twice is a
    # table that will disagree with itself.
    json.dump({"base_clean": base_clean, "classifier_ms": clf_ms,
               "params_clf": 145291, "t2": t2,
               "t4_head": [T3_HEAD.get(k, PRETTY[k]) for k in T1_ORDER],
               "t4": t4, "winners": len(winners),
               "clahe_worst": clahe_worst},
              open(out / "_tabledata.json", "w"), ensure_ascii=False)
    print(f"\n  wrote _table1.json and _tabledata.json to {out}")


if __name__ == "__main__":
    main()
