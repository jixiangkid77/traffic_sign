# -*- coding: utf-8 -*-
"""
EXPORT_fig06_samples.py
Choose the six crops of Fig. 6 by rule, and export the eighteen cells the figure
shows, each verified against the prediction already on record.

WHY THE CHOICE IS MADE HERE AND NOT WRITTEN DOWN
The previous export carried six filenames chosen on 2026-07-16, when only two
learned restorers were aligned image by image. Five are aligned now, so the pool
a case draws from is a different pool, and two of the six cases named a learned
restorer that is no longer the best on its challenge. Filenames in a file cannot
notice that. The rule is executed here instead, against whatever is in the merged
file, so the six crops are always the six the rule produces from the data as it
stands.

THE RULE
  Each case fixes a challenge and a severity and states an outcome. A candidate
  is any crop in that cell, with no occlusion, that shows the outcome. Among the
  candidates, one crop is taken per case so that the six cover as many sign
  classes as the data allows; ties are broken by taking the lexicographically
  smallest set of filenames. The article uses five sign classes and there are six
  cases, so exactly one class appears twice. The old rule asked for six distinct
  classes, which five classes cannot supply: the six crops it produced already
  had one class twice, so it never held as written.

WHAT AN OUTCOME MAY BE
  A case's outcome must be readable from the three cells the figure prints, which
  are no enhancement, the best training-free operator on that challenge and the
  best learned restorer on that challenge, the same two choices Fig. 1 makes. A
  case whose point lies between two operators in the same family, such as gamma
  against the dark channel prior, cannot be read from those three cells and is
  not used; the branch accounting of Fig. 5 is where that comparison belongs.
  The six outcomes are therefore the four a triple of cells can show: none of the
  twelve front ends recovers the sign, only the learned family does, only the
  training-free family does, and both do. Each is stated over the whole family
  rather than over the one member the figure prints, so that no operator left
  off the page can contradict the row.

READS   outputs_revision/merged_per_image.csv   (K_merge_results.py)
        outputs_revision/dcp_cure.csv           (Q_dcp_branch.py)
        datasets/CURE-TSR/**                    (the crops themselves)
WRITES  outputs_revision/fig06_samples/*.png    (18 cells, 32 x 32)
        outputs_revision/fig06_samples/manifest.json

RUN
  python EXPORT_fig06_samples.py
"""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import F_master_sweep_cache as F
import Q_dcp_branch as Q
import J_local_deep_eval as J
import O_eval_three_locally as O

# The root is taken from F rather than from this file's own location. Every
# other script in the project hardcodes the same path, and deriving it here from
# __file__ put it one level down, in src, where outputs_revision is not. One
# definition means the caches this script checks against and the caches it reads
# can never be two different directories.
PROJECT_ROOT = F.PROJECT_ROOT
OUT_DIR = PROJECT_ROOT / "outputs_revision"

TRAINING_FREE = ["gamma", "clahe", "stretch", "dcp"]
LEARNED = ["adair", "cidnet", "zero_dce", "ffa_net", "promptir"]
# Every front end the article compares, which is what "no front end recovers the
# sign" has to mean. The two routers are included: a claim that nothing recovers
# a crop would be false if the rule happened to recover it.
ALL_FRONT_ENDS = (["passthrough"] + TRAINING_FREE + LEARNED
                  + ["va_rule", "vb"])

# tag, challenge, severity, outcome. The outcomes are the four a triple of cells
# can show; two of them are used twice, on different challenges.
CASES = [
    ("F1", 7, 5, "none"),      # severe Gaussian blur
    ("F2", 8, 3, "learned"),   # noise
    ("F3", 11, 3, "learned"),  # snow
    ("S1", 12, 4, "free"),     # haze
    ("S2", 9, 3, "free"),      # rain
    ("S3", 4, 4, "both"),      # darkening
]

# The words as a reader writes them, not the keys as the tree stores them.
CHALLENGE_NAME = {4: "darkening", 7: "Gaussian blur", 8: "noise", 9: "rain",
                  11: "snow", 12: "haze"}

STORY = {
    "none": "no front end recovers the sign",
    "learned": "only a learned restorer recovers it",
    "free": "only a training-free operator recovers it",
    "both": "both recover it",
}


def occurrence_index(cure_root):
    """Map (filename, occurrence) to a path, the way the caches were built."""
    idx, seen = {}, defaultdict(int)
    for p in sorted(Path(cure_root).rglob("*.bmp")):
        n = p.name
        idx[(n, seen[n])] = p
        seen[n] += 1
    return idx


def load_tables(merged_path, dcp_path):
    rows = list(csv.DictReader(open(merged_path, newline="", encoding="utf-8")))
    dcp = {}
    for r in csv.DictReader(open(dcp_path, newline="", encoding="utf-8")):
        dcp[(r["filename"], int(r["occ"]))] = int(r["pred_dcp"])
    if len(dcp) != len(rows):
        raise SystemExit(f"ABORT: {len(rows)} merged rows against {len(dcp)} "
                         f"dark channel prior rows. They must be the same set.")
    return rows, dcp


def hit_table(rows, dcp):
    """One boolean array per front end: did it return the true class."""
    true = np.array([int(r["true"]) for r in rows])
    branch = np.array([r["rule_branch"] for r in rows])
    hit = {}
    for k in ["passthrough"] + TRAINING_FREE[:-1] + LEARNED + ["va_rule"]:
        col = f"pred_{k}"
        if col not in rows[0]:
            raise SystemExit(f"ABORT: {col} is missing from the merged file. "
                             f"Run K_merge_results.py on a set that has it.")
        hit[k] = np.array([int(r[col]) for r in rows]) == true
    hit["dcp"] = np.array(
        [dcp[(r["filename"], int(r["occ"]))] for r in rows]) == true
    # Selector V-B is the rule with the dark channel prior on the CLAHE branch.
    hit["vb"] = np.where(branch == "clahe", hit["dcp"], hit["va_rule"])
    return hit, true


def best_on_challenge(hit, ch, sev, pool, c):
    """The best member of a pool on a challenge, averaged over its five cells.

    This is Fig. 1's choice, cell by cell rather than crop by crop, so that a
    challenge with an uneven number of crops per severity cannot be swung by the
    severity that happens to have most of them.
    """
    cells = [np.where((ch == c) & (sev == s))[0] for s in range(1, 6)]
    return max(pool, key=lambda k: float(np.mean([hit[k][i].mean()
                                                  for i in cells])))


def candidates(hit, rows, ch, sev, occ, c, s, outcome, free, learn):
    """Every crop in the cell, unoccluded, that shows the outcome."""
    out = []
    for i in np.where((ch == c) & (sev == s))[0]:
        if occ[i] != 0:
            continue
        # The condition is on the whole family, not on the one member the
        # figure prints. A crop where the best learned restorer recovers the
        # sign and the best training-free operator does not is not a crop where
        # only the learned family recovers it: another training-free operator
        # may do so off-screen, and then the row says one thing while the data
        # says another. The family-wide form is the one a reader would check.
        if outcome == "none":
            ok = not any(hit[k][i] for k in ALL_FRONT_ENDS)
        elif outcome == "learned":
            ok = (hit[learn][i] and not hit["passthrough"][i]
                  and not any(hit[k][i] for k in TRAINING_FREE))
        elif outcome == "free":
            ok = (hit[free][i] and not hit["passthrough"][i]
                  and not any(hit[k][i] for k in LEARNED))
        else:
            ok = (hit[free][i] and hit[learn][i]
                  and not hit["passthrough"][i])
        if ok:
            out.append(i)
    return out


def choose(rows, pools):
    """One crop per case, covering as many sign classes as the data allows.

    Exhaustive over the classes each case can offer, which is at most five to the
    sixth. The objective is the number of distinct classes; the tie-break is the
    lexicographically smallest tuple of filenames, so the answer does not depend
    on the order a dictionary happens to iterate in.
    """
    per_class = {}
    for tag, idxs in pools.items():
        d = defaultdict(list)
        for i in idxs:
            d[int(rows[i]["true"])].append(i)
        per_class[tag] = {k: min(v, key=lambda j: rows[j]["filename"])
                          for k, v in d.items()}
        if not d:
            raise SystemExit(f"ABORT: case {tag} has no candidate. The outcome "
                             f"it states does not occur in its cell.")
    tags = [t for t, *_ in CASES]
    best = None
    for combo in itertools.product(*[sorted(per_class[t]) for t in tags]):
        key = (-len(set(combo)),
               tuple(rows[per_class[t][cl]]["filename"]
                     for t, cl in zip(tags, combo)))
        if best is None or key < best[0]:
            best = (key, combo)
    return {t: per_class[t][cl] for t, cl in zip(tags, best[1])}, best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root",
                    default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model", default=str(F.MODEL_PATH_DEFAULT))
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--outdir", default=str(OUT_DIR / "fig06_samples"))
    ap.add_argument("--adair-weight",
                    default=str(PROJECT_ROOT / "models" / "adair5d.ckpt"))
    ap.add_argument("--cidnet-weight", default="")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    rows, dcp = load_tables(args.merged, args.dcp)
    hit, true = hit_table(rows, dcp)
    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    occ = np.array([int(r["occ"]) for r in rows])
    print(f"  {len(rows):,} rows, {len(ALL_FRONT_ENDS)} front ends")

    # ---- the rule ----
    pools, meta = {}, {}
    for tag, c, s, outcome in CASES:
        free = best_on_challenge(hit, ch, sev, TRAINING_FREE, c)
        learn = best_on_challenge(hit, ch, sev, LEARNED, c)
        meta[tag] = (c, s, outcome, free, learn)
        pools[tag] = candidates(hit, rows, ch, sev, occ, c, s, outcome,
                                free, learn)
        klass = sorted({int(rows[i]["true"]) for i in pools[tag]})
        print(f"  {tag}  {CHALLENGE_NAME[c]:13s} severity {s}  "
              f"{free:8s} / {learn:9s}  {len(pools[tag]):4d} candidates "
              f"over classes {klass}")
    picked, combo = choose(rows, pools)
    print(f"\n  the six cover {len(set(combo))} of "
          f"{len({int(r['true']) for r in rows})} sign classes")

    # The outcome each case states, checked against the predictions on record
    # before a single model is loaded. A crop that fails here would produce a
    # picture that contradicts its own row, and finding that out after the
    # export has run is finding it out too late.
    for tag, *_ in CASES:
        i = picked[tag]
        c, s, outcome, free, learn = meta[tag]
        L, M, R = hit["passthrough"][i], hit[free][i], hit[learn][i]
        want = {"none": (False, False, False), "learned": (False, False, True),
                "free": (False, True, False), "both": (False, True, True)}[outcome]
        if (bool(L), bool(M), bool(R)) != want:
            raise SystemExit(
                f"ABORT: {tag} does not show what it claims. On record the "
                f"three cells are {bool(L)}/{bool(M)}/{bool(R)} and the "
                f"outcome '{outcome}' needs {want[0]}/{want[1]}/{want[2]}.")
        if outcome == "none" and any(hit[k][i] for k in ALL_FRONT_ENDS):
            got = [k for k in ALL_FRONT_ENDS if hit[k][i]]
            raise SystemExit(f"ABORT: {tag} claims no front end recovers the "
                             f"sign, but {got} do.")
        if outcome == "learned" and any(hit[k][i] for k in TRAINING_FREE):
            got = [k for k in TRAINING_FREE if hit[k][i]]
            raise SystemExit(f"ABORT: {tag} claims only a learned restorer "
                             f"recovers the sign, but {got} do too.")
        if outcome == "free" and any(hit[k][i] for k in LEARNED):
            got = [k for k in LEARNED if hit[k][i]]
            raise SystemExit(f"ABORT: {tag} claims only a training-free "
                             f"operator recovers the sign, but {got} do too.")
    print("  every case shows what it claims, on the predictions on record")

    # ---- the models ----
    model = F.load_model(Path(args.model), device)
    tfm = F.build_transform()
    needed = {meta[t][4] for t in picked}
    nets = {}
    for name in sorted(needed):
        if name == "adair":
            nets[name], _ = J.load_adair(device, args.adair_weight)
        elif name == "cidnet":
            nets[name], _ = J.load_cidnet(device, args.cidnet_weight)
        else:
            # O wraps each of its three so that the network returns the enhanced
            # image and nothing else, which is what enhance_batch expects. The
            # wrapper is where FFA-Net's normalisation lives, so it must be used
            # rather than the bare network.
            built, note = O.build(name, device)
            if built is None:
                raise SystemExit(f"ABORT: {name} could not be built: {note}")
            nets[name] = built[0]
        print(f"  loaded {name}")

    def classify(img_bgr):
        r = cv2.resize(img_bgr, (F.INPUT_SIZE, F.INPUT_SIZE))
        x = tfm(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            return int(model(x).argmax(1).item())

    index = occurrence_index(args.cure_root)
    print(f"  indexed {len(index):,} crops under {args.cure_root}")

    manifest, bad = [], 0
    print()
    for tag, *_ in CASES:
        i = picked[tag]
        c, s, outcome, free, learn = meta[tag]
        fn, occ_i, t = rows[i]["filename"], int(rows[i]["occ"]), int(true[i])
        path = index.get((fn, occ_i))
        if path is None:
            raise SystemExit(f"ABORT: {fn} occurrence {occ_i} is not under "
                             f"{args.cure_root}.")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"ABORT: cannot read {path}.")

        cells = {"degraded": img}
        cells[free] = (Q.dcp_enhance(img)[0] if free == "dcp"
                       else F.apply_branch(img, free))
        cells[learn] = J.enhance_batch(nets[learn], [img], device)[0]

        files, classify_of = [], {}
        for what, im in cells.items():
            out = outdir / f"{tag}_{what}.png"
            cv2.imwrite(str(out), im)
            files.append(out.name)
            got = classify(im)
            classify_of[what] = got
            key = "passthrough" if what == "degraded" else what
            want = (dcp[(fn, occ_i)] if key == "dcp"
                    else int(rows[i][f"pred_{key}"]))
            # The export is only worth anything if the picture it writes is the
            # picture that produced the number the article quotes.
            if got != want:
                bad += 1
            print(f"  {tag:3s} {what:10s} -> class {got:3d}   on record "
                  f"{want:3d}   {'ok' if got == want else 'MISMATCH'}")
        # Everything the figure or its caption might need, so that neither has
        # to go back to the merged file and risk reading a different row.
        # size is the crop's own size, which the figure prints on the row;
        # all_pred is every front end on this crop, which is what a claim such
        # as "no front end recovers the sign" has to rest on.
        h, w = img.shape[:2]
        all_pred = {"passthrough": int(rows[i]["pred_passthrough"]),
                    "dcp": dcp[(fn, occ_i)]}
        for k in ["gamma", "clahe", "stretch", "va_rule"] + LEARNED:
            all_pred[k] = int(rows[i][f"pred_{k}"])
        all_pred["vb"] = (all_pred["dcp"] if rows[i]["rule_branch"] == "clahe"
                          else all_pred["va_rule"])
        manifest.append(dict(
            tag=tag, fn=fn, occ=occ_i, ch=c, sev=s, true=t,
            size=[w, h],
            outcome=outcome, story=f"{CHALLENGE_NAME[c]}: "
                                   f"{STORY[outcome]}",
            best_free=free, best_learned=learn,
            cells=[{"what": what, "pred": classify_of[what]}
                   for what in cells],
            recovered_by=sorted(k for k, v in all_pred.items() if v == t),
            all_pred=all_pred,
            files=files))

    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=1)
    print()
    if bad:
        raise SystemExit(f"  {bad} cell(s) did not reproduce the recorded "
                         f"prediction. Do NOT use these files.")
    print(f"  every cell reproduces its recorded prediction")
    print(f"  wrote 18 images and manifest.json to {outdir}")
    verify(outdir, args.merged, args.dcp, classify)


def verify(outdir, merged_path, dcp_path, classify):
    """Read the files back off disk and check every claim they make.

    Nothing here trusts a variable the export left in memory. The manifest is
    read from the file that was just written, the images are read from the files
    that were just written, and the accuracies are recomputed from the merged
    file. A check that shares state with the thing it checks is not a check.
    """
    print("\n=== INDEPENDENT CHECK ===")
    man = json.load(open(Path(outdir) / "manifest.json"))
    rows, dcp = load_tables(merged_path, dcp_path)
    hit, true = hit_table(rows, dcp)
    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    where = {(r["filename"], int(r["occ"])): i for i, r in enumerate(rows)}
    bad = []

    if [m["tag"] for m in man] != [t for t, *_ in CASES]:
        bad.append("the manifest does not carry the six cases in figure order")
    if len({(m["fn"], m["occ"]) for m in man}) != len(man):
        bad.append("two cases point at the same crop")

    for m in man:
        tag = m["tag"]
        i = where.get((m["fn"], m["occ"]))
        if i is None:
            bad.append(f"{tag}: {m['fn']} is not in the merged file")
            continue
        # the crop is the crop the case says it is
        if (int(rows[i]["ch"]), int(rows[i]["sev"]), int(true[i])) != \
                (m["ch"], m["sev"], m["true"]):
            bad.append(f"{tag}: challenge, severity or class disagrees with "
                       f"the merged file")
        # the two named front ends are the ones Fig. 1 would name
        free = best_on_challenge(hit, ch, sev, TRAINING_FREE, m["ch"])
        learn = best_on_challenge(hit, ch, sev, LEARNED, m["ch"])
        if (free, learn) != (m["best_free"], m["best_learned"]):
            bad.append(f"{tag}: names {m['best_free']}/{m['best_learned']}, "
                       f"the challenge gives {free}/{learn}")
        # the outcome holds over the whole family, not the printed member
        tf = [k for k in TRAINING_FREE if hit[k][i]]
        ln = [k for k in LEARNED if hit[k][i]]
        anyone = [k for k in ALL_FRONT_ENDS if hit[k][i]]
        want = {"none": not anyone,
                "learned": bool(ln) and not tf,
                "free": bool(tf) and not ln,
                "both": bool(tf) and bool(ln)}[m["outcome"]]
        if not want:
            bad.append(f"{tag}: outcome '{m['outcome']}' is not what the data "
                       f"shows; training-free {tf or 'none'}, learned "
                       f"{ln or 'none'}")
        if m["outcome"] != "none" and hit["passthrough"][i]:
            bad.append(f"{tag}: no enhancement already returns the true class, "
                       f"so nothing was recovered")
        if sorted(m["recovered_by"]) != sorted(anyone):
            bad.append(f"{tag}: recovered_by disagrees with the merged file")
        # the three cells are the three the caption promises, in that order
        if [c["what"] for c in m["cells"]] != ["degraded", free, learn]:
            bad.append(f"{tag}: the cells are not no enhancement, then "
                       f"{free}, then {learn}")
        # the pictures on disk are the pictures that produced those numbers
        for cell, fname in zip(m["cells"], m["files"]):
            im = cv2.imread(str(Path(outdir) / fname), cv2.IMREAD_COLOR)
            if im is None:
                bad.append(f"{tag}: {fname} cannot be read back")
                continue
            got = classify(im)
            if got != cell["pred"]:
                bad.append(f"{tag}: {fname} now classifies as {got}, the "
                           f"manifest says {cell['pred']}")
            if cell["what"] == "degraded":
                h, w = im.shape[:2]
                if [w, h] != m["size"]:
                    bad.append(f"{tag}: the crop is {w}x{h}, the manifest "
                               f"says {m['size'][0]}x{m['size'][1]}")

    # the six cover as many sign classes as the data allows
    classes = [m["true"] for m in man]
    n_classes = len({int(r["true"]) for r in rows})
    if len(set(classes)) != min(len(man), n_classes):
        bad.append(f"the six cover {len(set(classes))} sign classes; "
                   f"{min(len(man), n_classes)} was available")

    for b in bad:
        print(f"  FAIL  {b}")
    if bad:
        raise SystemExit(f"\n  {len(bad)} check(s) failed. Do NOT use these "
                         f"files.")
    print(f"  {len(man)} cases, {sum(len(m['files']) for m in man)} images: "
          f"every claim in the manifest is what the data says, and every "
          f"picture is the picture that produced its number.")


if __name__ == "__main__":
    main()
