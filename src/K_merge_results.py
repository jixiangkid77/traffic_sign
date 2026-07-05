r"""
K_merge_results.py -- final merge of all CURE-TSR per-image results
                      (extended tables + significance + oracle + efficiency)

WHAT IT DOES (all table lookups, no image processing, no GPU)
  1. RE-KEY + VERIFY: CURE-TSR contains 9,256 filename collisions between
     Real_Train and Real_Test. This script rescans the dataset directory in
     BOTH scan orders actually used by the producing scripts
        F order: sorted(full path)          (F_master_sweep_cache.py)
        J order: sorted((basename, path))   (J_local_deep_eval.py)
     and verifies that "(filename, k-th occurrence)" maps to the SAME
     physical file under both orders. Only then are the three per-image
     files joined on (filename, occurrence). Any mismatch aborts loudly.
  2. EXTENDED TABLE 2: degraded-average accuracy and macro-F1 with paired
     bootstrap 95% CIs for the aligned methods (baseline, 3 fixed ops,
     VA rule, CIDNet, AdaIR) + recomputed rows for the legacy methods
     (Zero-DCE, FFA-Net, PromptIR) from the authoritative June per-image
     file. Paired difference CIs: AdaIR-VA, CIDNet-VA, VA-baseline,
     AdaIR-baseline.
  3. EXTENDED TABLE 3: per-challenge accuracies for all 9 methods + oracles.
  4. ORACLES: oracle-of-4 (classical pool, sanity vs 65.64), oracle-of-6
     (classical + AdaIR + CIDNet, the new headline ceiling), and the legacy
     oracle-of-7 recomputed from the June file (sanity vs 70.15).
  5. COMPLEMENTARITY: VA-vs-AdaIR win/lose split and "unique fixes"
     (images only AdaIR repairs / only the classical pool repairs).
  6. EFFICIENCY: parameter counts read from the local weight files
     (AdaIR / CIDNet / FFA-Net / Zero-DCE; PromptIR marked n/a since its
     weights live only on Colab), CompactCNN constant, measured latencies
     with their scopes stated.

HONEST LIMITATION (stated in output too)
  The June legacy file indexes images by dataset order on the Colab
  filesystem, which is not reproducible; legacy methods therefore cannot be
  per-image aligned with the new files. Their own CIs are valid (computed
  within the file); cross-family paired differences are only reported
  against methods inside the same file family.

USAGE
    python K_merge_results.py                 (full: rescan + merge + boot)
    python K_merge_results.py --boot 2000     (faster CI pass)
OUTPUT
    outputs_revision\merged_per_image.csv
    outputs_revision\cells_per_method.csv
    outputs_revision\extended_results.json

CHANGE LOG
    2026-07-04  count_params: unwrap nested checkpoint keys (state_dict /
                model / params / params_ema / model_state / net) before
                counting; a zero-tensor count now raises and prints as n/a
                instead of a silent 0. Fixes FFA-Net=0: ffa_net_ots.pk is a
                {'model': state_dict, ...} wrapper, and the old code only
                unwrapped 'state_dict'. Accuracy, CI, oracle, and
                complementarity code paths are untouched.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"

CACHE_DEFAULT = OUT_DIR / "cure_master_cache.csv"
ADAIR_DEFAULT = OUT_DIR / "deep_adair_cure.csv"
CIDNET_DEFAULT = OUT_DIR / "deep_cidnet_cure.csv"
LEGACY_DEFAULT = PROJECT_ROOT / "outputs_cure_tsr" / \
    "cure_tsr_per_image_predictions.csv"
CURE_DEFAULT = PROJECT_ROOT / "datasets" / "CURE-TSR"

FILENAME_PATTERN = re.compile(
    r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.(bmp|png|jpg|jpeg)$", re.IGNORECASE)
IMAGE_EXTS = ("*.bmp", "*.png", "*.jpg", "*.jpeg")
CURE_TO_GTSRB = {3: 9, 6: 14, 11: 12, 12: 17, 13: 13}
EVAL_CHALLENGES = [4, 8, 9, 11, 12]
CH_NAME = {4: "Darkening", 8: "Noise", 9: "Rain", 11: "Snow", 12: "Haze"}

BRANCHES = ["passthrough", "gamma", "clahe", "stretch"]
ALIGNED = ["passthrough", "gamma", "clahe", "stretch",
           "va_rule", "cidnet", "adair"]
LEGACY_METHODS = ["zero_dce", "ffa_net", "promptir"]

ANCHORS = {"passthrough": 52.56, "va_rule": 57.32, "oracle4": 65.64,
           "cf_baseline": 80.77, "legacy_zero_dce": 47.72,
           "legacy_ffa_net": 54.72, "legacy_promptir": 54.61,
           "legacy_oracle7": 70.15}


# ------------------------------------------------------------------
# 1) disk rescan + occurrence-key verification
# ------------------------------------------------------------------
def usable(fp):
    m = FILENAME_PATTERN.match(fp.name)
    if not m:
        return False
    seq, sign, ch, sev = (int(m.group(1)), int(m.group(2)),
                          int(m.group(3)), int(m.group(4)))
    if seq != 1 or sign not in CURE_TO_GTSRB:
        return False
    if ch not in set([0] + EVAL_CHALLENGES):
        return False
    if ch != 0 and not (1 <= sev <= 5):
        return False
    return True


def _all_images(root):
    out = []
    for ext in IMAGE_EXTS:
        out.extend(root.rglob(ext))
    return [p for p in out if usable(p)]


def occ_map(paths_ordered, root):
    seen = defaultdict(int)
    m = {}
    for p in paths_ordered:
        k = seen[p.name]
        seen[p.name] += 1
        m[(p.name, k)] = str(p.relative_to(root)).replace("\\", "/")
    return m


def rescan_verify(cure_root):
    root = Path(cure_root)
    paths = _all_images(root)
    if not paths:
        sys.exit(f"[FATAL] no usable images under {root}")
    order_f = sorted(paths)                                # F order
    order_j = sorted(paths, key=lambda p: (p.name, str(p)))  # J order
    mf, mj = occ_map(order_f, root), occ_map(order_j, root)
    if mf.keys() != mj.keys():
        sys.exit("[FATAL] occurrence key sets differ between scan orders")
    bad = [(k, mf[k], mj[k]) for k in mf if mf[k] != mj[k]]
    if bad:
        print("[FATAL] occurrence->file mapping differs between the two "
              "scan orders; per-image join would be corrupted. Examples:")
        for k, a, b in bad[:5]:
            print("   ", k, "F->", a, " J->", b)
        sys.exit(1)
    n_dup = sum(1 for (nm, k) in mf if k > 0)
    print(f"[rescan] {len(mf)} usable images; duplicated-name copies: "
          f"{n_dup}; F-order and J-order agree on every occurrence key. OK")
    return mf


# ------------------------------------------------------------------
# 2) keyed loading + join
# ------------------------------------------------------------------
def load_keyed(path, kind):
    """kind: 'cache' | 'deep'. Returns dict key->row(dict of needed fields).
    Occurrence index assigned by row order within the file."""
    seen = defaultdict(int)
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            nm = r["filename"]
            k = seen[nm]
            seen[nm] += 1
            row = {"ch": int(r["ch_id"]), "sev": int(r["sev"]),
                   "sign": int(r["cure_sign"]), "true": int(r["gtsrb_true"])}
            if kind == "cache":
                row["b"] = float(r["b"])
                row["c"] = float(r["c"])
                row["e"] = float(r["e"])
                row["rule_branch"] = r["rule_branch"]
                for br in BRANCHES:
                    row[f"pred_{br}"] = int(r[f"pred_{br}"])
            else:
                row["pred"] = int(r["pred"])
                row["prob"] = float(r["prob"])
            out[(nm, k)] = row
    return out


def join_all(cache, adair, cidnet):
    keys = set(cache) & set(adair) & set(cidnet)
    if not (len(keys) == len(cache) == len(adair) == len(cidnet)):
        sys.exit(f"[FATAL] key sets differ: cache={len(cache)} "
                 f"adair={len(adair)} cidnet={len(cidnet)} "
                 f"common={len(keys)}")
    merged = []
    for key in sorted(keys):
        a, d, c = cache[key], adair[key], cidnet[key]
        for fld in ("ch", "sev", "sign", "true"):
            if not (a[fld] == d[fld] == c[fld]):
                sys.exit(f"[FATAL] metadata mismatch at {key}: field {fld} "
                         f"cache={a[fld]} adair={d[fld]} cidnet={c[fld]}")
        va_pred = a[f"pred_{a['rule_branch']}"]
        row = {"key": key, "ch": a["ch"], "sev": a["sev"], "true": a["true"],
               "b": a["b"], "c": a["c"], "e": a["e"],
               "rule_branch": a["rule_branch"],
               "pred_passthrough": a["pred_passthrough"],
               "pred_gamma": a["pred_gamma"],
               "pred_clahe": a["pred_clahe"],
               "pred_stretch": a["pred_stretch"],
               "pred_va_rule": va_pred,
               "pred_cidnet": c["pred"], "pred_adair": d["pred"],
               "prob_cidnet": c["prob"], "prob_adair": d["prob"]}
        merged.append(row)
    print(f"[join] merged {len(merged)} images x {len(ALIGNED)} methods; "
          "metadata consistent on every key")
    return merged


# ------------------------------------------------------------------
# metrics helpers
# ------------------------------------------------------------------
def cell_arrays(rows, methods, pred_prefix="pred_"):
    """-> cells: dict (ch,sev)->{'true':arr,'preds':{m:arr}} (degraded only),
       cf: same structure single entry for ch==0."""
    cells = defaultdict(lambda: {"true": [], "preds": defaultdict(list)})
    cf = {"true": [], "preds": defaultdict(list)}
    for r in rows:
        tgt = cf if r["ch"] == 0 else cells[(r["ch"], r["sev"])]
        tgt["true"].append(r["true"])
        for m in methods:
            tgt["preds"][m].append(r[pred_prefix + m])
    def _np(d):
        d["true"] = np.asarray(d["true"], np.int64)
        d["preds"] = {m: np.asarray(v, np.int64)
                      for m, v in d["preds"].items()}
        return d
    return {k: _np(v) for k, v in cells.items()}, _np(cf)


def macro_f1(t, p, classes):
    f1s = []
    for cls in classes:
        tp = int(np.sum((t == cls) & (p == cls)))
        fp = int(np.sum((t != cls) & (p == cls)))
        fn = int(np.sum((t == cls) & (p != cls)))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return 100.0 * float(np.mean(f1s))


def point_metrics(cells, cf, methods, classes):
    out = {}
    for m in methods:
        accs = [100.0 * float(np.mean(c["preds"][m] == c["true"]))
                for c in cells.values()]
        f1s = [macro_f1(c["true"], c["preds"][m], classes)
               for c in cells.values()]
        cfa = 100.0 * float(np.mean(cf["preds"][m] == cf["true"])) \
            if len(cf["true"]) else float("nan")
        out[m] = {"deg_acc": float(np.mean(accs)),
                  "deg_f1": float(np.mean(f1s)), "cf_acc": cfa}
    return out


def bootstrap(cells, methods, classes, B, seed, f1_for=()):
    """Shared per-cell draws across methods (paired). Returns
    acc_samples[m] (B,), f1_samples[m] (B,) for m in f1_for."""
    rng = np.random.default_rng(seed)
    acc = {m: np.zeros(B) for m in methods}
    f1 = {m: np.zeros(B) for m in f1_for}
    ncell = len(cells)
    for ci, (_, c) in enumerate(sorted(cells.items())):
        n = len(c["true"])
        idx = rng.integers(0, n, size=(B, n))
        tt = c["true"][idx]                      # (B, n)
        for m in methods:
            pp = c["preds"][m][idx]
            acc[m] += 100.0 * (pp == tt).mean(axis=1)
            if m in f1:
                fsum = np.zeros(B)
                for cls in classes:
                    tmask = tt == cls
                    pmask = pp == cls
                    tp = (tmask & pmask).sum(axis=1).astype(np.float64)
                    fp = (~tmask & pmask).sum(axis=1)
                    fn = (tmask & ~pmask).sum(axis=1)
                    pr = np.divide(tp, tp + fp,
                                   out=np.zeros(B), where=(tp + fp) > 0)
                    rc = np.divide(tp, tp + fn,
                                   out=np.zeros(B), where=(tp + fn) > 0)
                    fsum += np.divide(2 * pr * rc, pr + rc,
                                      out=np.zeros(B), where=(pr + rc) > 0)
                f1[m] += 100.0 * fsum / len(classes)
        print(f"[boot] cell {ci+1}/{ncell} done", end="\r")
    print()
    for m in methods:
        acc[m] /= ncell
    for m in f1:
        f1[m] /= ncell
    return acc, f1


def ci(v):
    return (round(float(np.percentile(v, 2.5)), 2),
            round(float(np.percentile(v, 97.5)), 2))


# ------------------------------------------------------------------
# legacy file (June, Colab): self-contained recompute
# ------------------------------------------------------------------
def legacy_metrics(path, B, seed, classes):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    per = defaultdict(lambda: defaultdict(lambda: {"t": [], "p": []}))
    cf = defaultdict(lambda: {"t": [], "p": []})
    for r in rows:
        m = r["method"]
        t, p = int(r["true_label"]), int(r["pred_label"])
        if r["split"] == "challengefree":
            cf[m]["t"].append(t)
            cf[m]["p"].append(p)
        else:
            per[m][(r["challenge"], int(r["severity"]))]["t"].append(t)
            per[m][(r["challenge"], int(r["severity"]))]["p"].append(p)
    out = {}
    rng = np.random.default_rng(seed)
    for m in LEGACY_METHODS:
        if m not in per:
            continue
        cells = per[m]
        accs, f1s = [], []
        boot = np.zeros(B)
        for (_, cell) in sorted(cells.items()):
            t = np.asarray(cell["t"])
            p = np.asarray(cell["p"])
            accs.append(100.0 * float(np.mean(t == p)))
            f1s.append(macro_f1(t, p, classes))
            idx = rng.integers(0, len(t), size=(B, len(t)))
            boot += 100.0 * (p[idx] == t[idx]).mean(axis=1)
        boot /= len(cells)
        cft = np.asarray(cf[m]["t"])
        cfp = np.asarray(cf[m]["p"])
        out[m] = {"deg_acc": float(np.mean(accs)),
                  "deg_f1": float(np.mean(f1s)),
                  "cf_acc": 100.0 * float(np.mean(cft == cfp)),
                  "acc_ci": ci(boot),
                  "per_challenge": {ch: float(np.mean(
                      [100.0 * np.mean(np.asarray(cell["p"]) ==
                                       np.asarray(cell["t"]))
                       for (c2, s2), cell in cells.items() if c2 == ch]))
                      for ch in set(c for c, _ in cells)}}
    # legacy oracle-of-7 (aligned within the file by idx)
    key_rows = defaultdict(dict)
    for r in rows:
        if r["split"] == "challengefree":
            continue
        key_rows[(r["challenge"], r["severity"], r["idx"])][r["method"]] = \
            int(r["correct"])
    cells7 = defaultdict(lambda: [0, 0])
    pool = ["baseline", "fixed_gamma", "fixed_clahe", "fixed_stretch"] + \
        LEGACY_METHODS
    for (chn, sev, _), d in key_rows.items():
        cells7[(chn, sev)][1] += 1
        if any(d.get(m, 0) == 1 for m in pool):
            cells7[(chn, sev)][0] += 1
    o7 = float(np.mean([100.0 * c / n for c, n in cells7.values()]))
    return out, o7


# ------------------------------------------------------------------
# efficiency: parameter counts from local weight files (guarded)
# ------------------------------------------------------------------
def count_params():
    res = {"CompactCNN": 145291}
    try:
        import torch
    except Exception:
        print("[params] torch unavailable here; skipping counts")
        return res
    def _unwrap(ck):
        if not isinstance(ck, dict):
            return ck
        for key in ("state_dict", "model", "params",
                    "params_ema", "model_state", "net"):
            v = ck.get(key)
            if isinstance(v, dict) and any(
                    hasattr(t, "numel") for t in v.values()):
                return v
        return ck

    def _count_sd(sd):
        n = int(sum(v.numel() for v in sd.values()
                    if hasattr(v, "numel")))
        if n == 0:
            raise ValueError("no tensors found; checkpoint layout?")
        return n
    try:
        ck = torch.load(str(PROJECT_ROOT / "models" / "adair5d.ckpt"),
                        map_location="cpu")
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
        res["AdaIR"] = _count_sd(sd)
    except Exception as ex:
        res["AdaIR"] = f"n/a ({type(ex).__name__})"
    try:
        import safetensors.torch as sf
        from huggingface_hub import hf_hub_download
        wf = hf_hub_download(repo_id="Fediory/HVI-CIDNet-Generalization",
                             filename="model.safetensors")
        res["CIDNet"] = _count_sd(sf.load_file(wf))
    except Exception as ex:
        res["CIDNet"] = f"n/a ({type(ex).__name__})"
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from evaluate_learned_baselines import (ZERO_DCE_WEIGHTS,
                                                FFA_NET_WEIGHTS)
        for name, p in [("Zero-DCE", ZERO_DCE_WEIGHTS),
                        ("FFA-Net", FFA_NET_WEIGHTS)]:
            try:
                ck = torch.load(str(p), map_location="cpu")
                sd = _unwrap(ck)
                if not isinstance(sd, dict):
                    raise ValueError("unrecognized checkpoint layout")
                res[name] = _count_sd(sd)
            except Exception as ex:
                res[name] = f"n/a ({type(ex).__name__})"
    except Exception as ex:
        res["Zero-DCE"] = res["FFA-Net"] = f"n/a ({type(ex).__name__})"
    res["PromptIR"] = "n/a (weights only on Colab)"
    return res


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(CURE_DEFAULT))
    ap.add_argument("--cache", default=str(CACHE_DEFAULT))
    ap.add_argument("--adair", default=str(ADAIR_DEFAULT))
    ap.add_argument("--cidnet", default=str(CIDNET_DEFAULT))
    ap.add_argument("--legacy", default=str(LEGACY_DEFAULT))
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-rescan", action="store_true",
                    help="DANGEROUS: trust occurrence alignment blindly")
    ap.add_argument("--skip-params", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_rescan:
        rescan_verify(args.cure_root)
    else:
        print("[rescan] SKIPPED by flag; alignment not verified this run")

    cache = load_keyed(args.cache, "cache")
    adair = load_keyed(args.adair, "deep")
    cidnet = load_keyed(args.cidnet, "deep")
    merged = join_all(cache, adair, cidnet)

    classes = sorted(set(CURE_TO_GTSRB.values()))
    cells, cf = cell_arrays(merged, ALIGNED)
    pts = point_metrics(cells, cf, ALIGNED, classes)

    # anchors
    print("\n=== anchor check (must match published/cache values) ===")
    for name, exp, got in [
            ("passthrough", ANCHORS["passthrough"],
             pts["passthrough"]["deg_acc"]),
            ("va_rule", ANCHORS["va_rule"], pts["va_rule"]["deg_acc"]),
            ("CF baseline", ANCHORS["cf_baseline"],
             pts["passthrough"]["cf_acc"])]:
        d = got - exp
        flag = "" if abs(d) <= 0.05 else "  << WARNING"
        print(f"  {name:12s} {got:6.2f} vs {exp:6.2f}  {d:+.2f}{flag}")

    print(f"\n[boot] paired bootstrap B={args.boot} seed={args.seed} "
          f"(acc CIs for all aligned; F1 CIs for key methods)")
    f1_for = ("passthrough", "va_rule", "cidnet", "adair")
    acc_s, f1_s = bootstrap(cells, ALIGNED, classes,
                            args.boot, args.seed, f1_for=f1_for)

    legacy, o7 = ({}, float("nan"))
    if Path(args.legacy).exists():
        legacy, o7 = legacy_metrics(args.legacy, args.boot,
                                    args.seed + 1, classes)
        for m, exp in [("zero_dce", ANCHORS["legacy_zero_dce"]),
                       ("ffa_net", ANCHORS["legacy_ffa_net"]),
                       ("promptir", ANCHORS["legacy_promptir"])]:
            if m in legacy:
                d = legacy[m]["deg_acc"] - exp
                flag = "" if abs(d) <= 0.05 else "  << WARNING"
                print(f"  legacy {m:9s} {legacy[m]['deg_acc']:6.2f} "
                      f"vs {exp:6.2f}  {d:+.2f}{flag}")
        d7 = o7 - ANCHORS["legacy_oracle7"]
        flag = "" if abs(d7) <= 0.05 else "  << WARNING"
        print(f"  legacy oracle7 {o7:6.2f} vs "
              f"{ANCHORS['legacy_oracle7']:6.2f}  {d7:+.2f}{flag}")
    else:
        print(f"[legacy] file not found, legacy rows skipped: {args.legacy}")

    # oracles on aligned pool
    def oracle_avg(pool):
        vals = []
        per_ch = defaultdict(list)
        for (chsev, c) in sorted(cells.items()):
            ok = np.zeros(len(c["true"]), bool)
            for m in pool:
                ok |= (c["preds"][m] == c["true"])
            v = 100.0 * float(ok.mean())
            vals.append(v)
            per_ch[chsev[0]].append(v)
        return float(np.mean(vals)), {CH_NAME[k]: float(np.mean(v))
                                      for k, v in per_ch.items()}
    o4, o4ch = oracle_avg(BRANCHES)
    o6, o6ch = oracle_avg(BRANCHES + ["adair", "cidnet"])
    d4 = o4 - ANCHORS["oracle4"]
    flag = "" if abs(d4) <= 0.05 else "  << WARNING"
    print(f"  oracle4 {o4:6.2f} vs {ANCHORS['oracle4']:6.2f}  {d4:+.2f}{flag}")

    ceiling = pts["passthrough"]["cf_acc"]
    base = pts["passthrough"]["deg_acc"]
    gap = ceiling - base

    # ---------------- extended Table 2 ----------------
    print("\n=== EXTENDED TABLE 2: real CURE-TSR (cell-averaged) ===")
    print(f"{'method':14s} {'deg-acc [95% CI]':>22s} "
          f"{'deg-F1':>7s} {'CF':>7s}")
    order = ["passthrough", "clahe", "gamma", "stretch"]
    for m in order:
        a = pts[m]
        lo, hi = ci(acc_s[m])
        print(f"{m:14s} {a['deg_acc']:8.2f} [{lo:5.2f},{hi:6.2f}] "
              f"{a['deg_f1']:7.2f} {a['cf_acc']:7.2f}")
    for m in LEGACY_METHODS:
        if m in legacy:
            a = legacy[m]
            lo, hi = a["acc_ci"]
            print(f"{m+'*':14s} {a['deg_acc']:8.2f} [{lo:5.2f},{hi:6.2f}] "
                  f"{a['deg_f1']:7.2f} {a['cf_acc']:7.2f}")
    for m in ["cidnet", "adair", "va_rule"]:
        a = pts[m]
        lo, hi = ci(acc_s[m])
        print(f"{m:14s} {a['deg_acc']:8.2f} [{lo:5.2f},{hi:6.2f}] "
              f"{a['deg_f1']:7.2f} {a['cf_acc']:7.2f}")
    print(f"{'oracle4':14s} {o4:8.2f}   (gap captured "
          f"{100*(o4-base)/gap:4.1f}%)")
    print(f"{'oracle6':14s} {o6:8.2f}   (gap captured "
          f"{100*(o6-base)/gap:4.1f}%)")
    if not np.isnan(o7):
        print(f"{'oracle7*':14s} {o7:8.2f}   (legacy pool, June file)")
    print("(* legacy rows recomputed from the June Colab per-image file; "
          "not per-image alignable with the new runs)")

    # F1 CIs for key methods
    print("\nmacro-F1 95% CIs (key methods): " + "  ".join(
        f"{m}={ci(f1_s[m])}" for m in f1_for))

    # paired differences
    print("\n=== paired differences (95% CI; excludes 0 => significant) ===")
    diffs = {}
    for a_m, b_m in [("adair", "va_rule"), ("cidnet", "va_rule"),
                     ("va_rule", "passthrough"), ("adair", "passthrough"),
                     ("adair", "cidnet")]:
        d = acc_s[a_m] - acc_s[b_m]
        lo, hi = ci(d)
        point = pts[a_m]["deg_acc"] - pts[b_m]["deg_acc"]
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        diffs[f"{a_m}-{b_m}"] = {"point": round(point, 2),
                                 "ci": [lo, hi], "verdict": sig}
        print(f"  {a_m:10s} - {b_m:11s} = {point:+6.2f} "
              f"[{lo:+6.2f},{hi:+6.2f}]  {sig}")
    dF = f1_s["adair"] - f1_s["va_rule"]
    lo, hi = ci(dF)
    print(f"  adair - va_rule (macro-F1)   = "
          f"{pts['adair']['deg_f1']-pts['va_rule']['deg_f1']:+6.2f} "
          f"[{lo:+6.2f},{hi:+6.2f}]")

    # ---------------- extended Table 3 ----------------
    print("\n=== EXTENDED TABLE 3: per-challenge deg-avg ===")
    meths3 = ["passthrough", "gamma", "clahe", "stretch",
              "va_rule", "cidnet", "adair"]
    hdr = f"{'challenge':11s}" + "".join(f"{m[:7]:>8s}" for m in meths3)
    if legacy:
        hdr += "".join(f"{m[:7]+'*':>9s}" for m in LEGACY_METHODS)
    hdr += f"{'orcl4':>7s}{'orcl6':>7s}"
    print(hdr)
    for chid in EVAL_CHALLENGES:
        chn = CH_NAME[chid]
        line = f"{chn:11s}"
        for m in meths3:
            vals = [100.0 * float(np.mean(c["preds"][m] == c["true"]))
                    for (cs, c) in cells.items() if cs[0] == chid]
            line += f"{np.mean(vals):8.1f}"
        if legacy:
            for m in LEGACY_METHODS:
                v = legacy[m]["per_challenge"].get(chn, float("nan"))
                line += f"{v:9.1f}"
        line += f"{o4ch[chn]:7.1f}{o6ch[chn]:7.1f}"
        print(line)

    # ---------------- complementarity ----------------
    print("\n=== VA vs AdaIR complementarity (degraded images) ===")
    va_ok = np.concatenate([c["preds"]["va_rule"] == c["true"]
                            for _, c in sorted(cells.items())])
    ad_ok = np.concatenate([c["preds"]["adair"] == c["true"]
                            for _, c in sorted(cells.items())])
    cl_ok = None
    for m in BRANCHES:
        v = np.concatenate([c["preds"][m] == c["true"]
                            for _, c in sorted(cells.items())])
        cl_ok = v if cl_ok is None else (cl_ok | v)
    n = len(va_ok)
    comp = {
        "both_correct": float(np.mean(va_ok & ad_ok)),
        "va_only": float(np.mean(va_ok & ~ad_ok)),
        "adair_only": float(np.mean(~va_ok & ad_ok)),
        "both_wrong": float(np.mean(~va_ok & ~ad_ok)),
        "adair_unique_vs_classical_pool": float(np.mean(ad_ok & ~cl_ok)),
        "classical_pool_unique_vs_adair": float(np.mean(cl_ok & ~ad_ok)),
    }
    for k, v in comp.items():
        print(f"  {k:32s} {100*v:6.2f}%")

    # ---------------- efficiency ----------------
    eff = {}
    if not args.skip_params:
        eff = count_params()
        print("\n=== parameters (counted from local weight files) ===")
        for k, v in eff.items():
            print(f"  {k:12s} {v if isinstance(v,str) else f'{v:,}'}")
        print("  latency scopes: VA <0.2 ms/img (routing+operator only, "
              "Table 4); AdaIR ~333 ms/img and CIDNet ~89 ms/img "
              "(end-to-end J runs incl. IO+classifier, this CPU, "
              "2026-07-03/04)")

    # ---------------- outputs ----------------
    merged_csv = OUT_DIR / "merged_per_image.csv"
    with open(merged_csv, "w", encoding="utf-8", newline="") as f:
        flds = ["filename", "occ", "ch", "sev", "true", "b", "c", "e",
                "rule_branch"] + [f"pred_{m}" for m in ALIGNED] + \
               ["prob_cidnet", "prob_adair"]
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in merged:
            row = {"filename": r["key"][0], "occ": r["key"][1],
                   "ch": r["ch"], "sev": r["sev"], "true": r["true"],
                   "b": r["b"], "c": r["c"], "e": r["e"],
                   "rule_branch": r["rule_branch"],
                   "prob_cidnet": r["prob_cidnet"],
                   "prob_adair": r["prob_adair"]}
            for m in ALIGNED:
                row[f"pred_{m}"] = r[f"pred_{m}"]
            w.writerow(row)

    cells_csv = OUT_DIR / "cells_per_method.csv"
    with open(cells_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ch_id", "ch_name", "sev"] + ALIGNED +
                   ["oracle4", "oracle6"])
        for (chid, sev), c in sorted(cells.items()):
            row = [chid, CH_NAME[chid], sev]
            for m in ALIGNED:
                row.append(round(100.0 * float(
                    np.mean(c["preds"][m] == c["true"])), 2))
            ok4 = np.zeros(len(c["true"]), bool)
            for m in BRANCHES:
                ok4 |= (c["preds"][m] == c["true"])
            ok6 = ok4 | (c["preds"]["adair"] == c["true"]) | \
                (c["preds"]["cidnet"] == c["true"])
            row += [round(100.0 * float(ok4.mean()), 2),
                    round(100.0 * float(ok6.mean()), 2)]
            w.writerow(row)

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "boot": args.boot, "seed": args.seed,
        "point_metrics": {m: {k: round(v, 2) for k, v in d.items()}
                          for m, d in pts.items()},
        "acc_ci": {m: ci(acc_s[m]) for m in ALIGNED},
        "f1_ci": {m: ci(f1_s[m]) for m in f1_for},
        "legacy": {m: {k: (v if isinstance(v, (list, tuple, dict))
                           else round(v, 2)) for k, v in d.items()}
                   for m, d in legacy.items()},
        "legacy_oracle7": round(o7, 2) if not np.isnan(o7) else None,
        "oracle4": round(o4, 2), "oracle6": round(o6, 2),
        "oracle4_per_challenge": {k: round(v, 2) for k, v in o4ch.items()},
        "oracle6_per_challenge": {k: round(v, 2) for k, v in o6ch.items()},
        "ceiling_cf": round(ceiling, 2),
        "gap_capture": {"va_rule": round(100 * (pts['va_rule']['deg_acc']
                                                - base) / gap, 1),
                        "adair": round(100 * (pts['adair']['deg_acc']
                                              - base) / gap, 1),
                        "oracle4": round(100 * (o4 - base) / gap, 1),
                        "oracle6": round(100 * (o6 - base) / gap, 1)},
        "paired_diffs": diffs,
        "complementarity_pct": {k: round(100 * v, 2)
                                for k, v in comp.items()},
        "params": eff,
    }
    out_json = OUT_DIR / "extended_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] wrote {merged_csv}")
    print(f"[out] wrote {cells_csv}")
    print(f"[out] wrote {out_json}")
    print("\nSend me the full console output plus extended_results.json.")


if __name__ == "__main__":
    main()
