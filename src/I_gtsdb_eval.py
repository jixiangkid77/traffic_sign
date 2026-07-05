"""
I_gtsdb_eval.py -- cross-dataset evaluation on GTSDB (FullIJCNN2013)
                   + threshold-transferability experiment

WHAT IT ANSWERS
  1) Does the frozen GTSRB classifier + VA-Adaptive routing generalize to
     signs cropped from a different capture campaign (GTSDB: same 43-class
     labeling scheme, different camera / scenes / years)?
  2) Is what transfers the four THRESHOLD VALUES, or the CALIBRATION RECIPE
     (percentiles of clean statistics)?  We evaluate VA twice on the SAME
     test crops:
       (A) GTSRB-calibrated thresholds (the paper values), applied as-is;
       (B) thresholds re-derived on GTSDB train-split crops with the same
           recipe: T1=P15(b), T2=P15(c), T3=P15(e), T4=P70(b).
     If (A) is close to (B), the portable object is the recipe, which is the
     honest version of a "generality" claim.

HONEST FRAMING (goes into the manuscript as well)
  GTSDB was collected by the same research group and country as GTSRB, so
  this is a CROSS-DATASET generalization check, not an adverse-conditions
  benchmark.  Crops carry the OFFICIAL 43-class labels from gt.txt (same
  class ids as GTSRB), so no label mapping is invented by us.

DATA LAYOUT EXPECTED (from FullIJCNN2013.zip)
    D:\\Project\\traffic_sign\\datasets\\GTSDB\\FullIJCNN2013\\00000.ppm ...
    D:\\Project\\traffic_sign\\datasets\\GTSDB\\FullIJCNN2013\\gt.txt
  gt.txt lines: "00000.ppm;774;411;815;446;11"
                 filename;leftCol;topRow;rightCol;bottomRow;classId

SPLIT CONVENTION
  Image ids 00000-00599 -> calibration split ("train" of the IJCNN 2013
  competition); 00600-00899 -> evaluation split.  Only the evaluation split
  enters the accuracy tables; the calibration split is used solely to
  re-derive thresholds (condition B).

CROP MARGIN
  GTSRB classification images include a border of about 10 percent around
  the sign (at least 5 px) by dataset construction.  To avoid an artificial
  domain shift we crop GTSDB boxes with the same convention by default
  (--margin 0.10, min 5 px, clamped to image bounds).  Set --margin 0 to
  ablate.

PIPELINE (authoritative deployment order, same as F/G)
  crop (original size) -> stats@original (LIVE enhance.compute_image_stats)
  -> branch operator@original -> cv2.resize(32) -> BGR2RGB -> ToTensor
  -> Normalize(GTSRB mean/std) -> frozen CompactCNN.
  All four branch outputs are cached per crop, so VA-A, VA-B, oracle and any
  future router are pure table lookups (same design as the CURE cache).

USAGE
    python I_gtsdb_eval.py            # sweep + analysis
    python I_gtsdb_eval.py --verify-only     # re-run analysis on the cache
OUTPUT
    outputs_revision\\gtsdb_master_cache.csv (+ .run_config.json)
    outputs_revision\\gtsdb_results.json
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from enhance import (                       # noqa: E402  (LIVE modules)
    apply_clahe, apply_gamma, compute_image_stats,
)
from revision_utils import (                # noqa: E402
    load_gtsrb_compactcnn, GTSRB_MEAN, GTSRB_STD,
)

GTSDB_DIR_DEFAULT = PROJECT_ROOT / "datasets" / "GTSDB" / "FullIJCNN2013"
OUT_CSV_DEFAULT = PROJECT_ROOT / "outputs_revision" / "gtsdb_master_cache.csv"
OUT_JSON_DEFAULT = PROJECT_ROOT / "outputs_revision" / "gtsdb_results.json"

INPUT_SIZE = 32
GTSRB_THRESHOLDS = {"T1": 0.1206, "T2": 0.1061, "T3": 0.0726, "T4": 0.4085}
BRANCHES = ["passthrough", "gamma", "clahe", "stretch"]
SPLIT_BOUNDARY = 600           # ids < 600 calibration, >= 600 evaluation

CSV_FIELDS = ["crop_id", "img_id", "split", "true",
              "left", "top", "right", "bottom", "cw", "ch",
              "b", "c", "e"]
for _br in BRANCHES:
    CSV_FIELDS += [f"pred_{_br}", f"prob_{_br}"]


# ------------------------------------------------------------------
def apply_stretch(img_bgr):
    f = img_bgr.astype(np.float32) / 255.0
    out = np.clip((f - 0.5) * 1.5 + 0.5, 0, 1)
    return (out * 255).astype(np.uint8)


def apply_branch(img_bgr, branch):
    if branch == "passthrough":
        return img_bgr
    if branch == "gamma":
        return apply_gamma(img_bgr, gamma=0.5)
    if branch == "clahe":
        return apply_clahe(img_bgr)
    if branch == "stretch":
        return apply_stretch(img_bgr)
    raise ValueError(branch)


def route_decision(b, c, e, T):
    if b < T["T1"]:
        return "gamma"
    elif c < T["T2"]:
        return "clahe"
    elif e < T["T3"] and b > T["T4"]:
        return "stretch"
    else:
        return "passthrough"


def parse_gt(gt_path):
    """Return list of (img_name, left, top, right, bottom, class_id)."""
    rows = []
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(";")
            if len(parts) < 6:
                continue
            name = parts[0]
            l, t, r, btm, cls = (int(parts[1]), int(parts[2]),
                                 int(parts[3]), int(parts[4]), int(parts[5]))
            rows.append((name, l, t, r, btm, cls))
    return rows


def margined_box(l, t, r, btm, W, H, margin, min_px):
    """Expand the box by `margin` of its size per side (>= min_px),
    clamped to image bounds."""
    bw, bh = r - l, btm - t
    mx = max(int(round(bw * margin)), min_px) if margin > 0 else 0
    my = max(int(round(bh * margin)), min_px) if margin > 0 else 0
    L = max(0, l - mx)
    T_ = max(0, t - my)
    R = min(W, r + mx)
    B = min(H, btm + my)
    return L, T_, R, B


def macro_f1(pairs, classes):
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return 100.0 * sum(f1s) / len(f1s) if f1s else float("nan")


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])


# ------------------------------------------------------------------
def run_sweep(args):
    gtsdb = Path(args.gtsdb_dir)
    gt_path = gtsdb / "gt.txt"
    if not gt_path.exists():
        print(f"[FATAL] gt.txt not found: {gt_path}")
        print("        Expected FullIJCNN2013 layout; check --gtsdb-dir.")
        sys.exit(1)
    boxes = parse_gt(gt_path)
    print(f"[data] gt.txt boxes: {len(boxes)}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    mode = "w"
    if args.resume and out_csv.exists():
        with open(out_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["crop_id"])
        mode = "a"
        print(f"[resume] {len(done)} crops already cached")

    model = load_gtsrb_compactcnn("cpu")
    tfm = build_transform()

    fout = open(out_csv, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()

    # group boxes by image to read each full frame once
    by_img = defaultdict(list)
    for k, (name, l, t, r, btm, cls) in enumerate(boxes):
        by_img[name].append((k, l, t, r, btm, cls))

    buf_rows, buf_tensors = [], []

    def flush():
        if not buf_rows:
            return
        x = torch.stack(buf_tensors, dim=0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            top_p, top_i = probs.max(dim=1)
        top_p = top_p.numpy()
        top_i = top_i.numpy()
        for j, row in enumerate(buf_rows):
            base = j * 4
            for kk, br in enumerate(BRANCHES):
                row[f"pred_{br}"] = int(top_i[base + kk])
                row[f"prob_{br}"] = round(float(top_p[base + kk]), 6)
            writer.writerow(row)
        fout.flush()
        buf_rows.clear()
        buf_tensors.clear()

    t0 = time.time()
    n_done, n_skip_small = 0, 0
    for name in sorted(by_img):
        fpath = gtsdb / name
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"[warn] unreadable frame skipped: {fpath}")
            continue
        H, W = img.shape[:2]
        img_id = int(Path(name).stem)
        split = "calib" if img_id < SPLIT_BOUNDARY else "eval"
        for (k, l, t, r, btm, cls) in by_img[name]:
            crop_id = f"{Path(name).stem}_{k}"
            if crop_id in done:
                continue
            L, T_, R, B = margined_box(l, t, r, btm, W, H,
                                       args.margin, args.min_margin_px)
            crop = img[T_:B, L:R]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                n_skip_small += 1
                continue
            b, c, e = compute_image_stats(crop)      # ORIGINAL crop size
            row = {
                "crop_id": crop_id, "img_id": img_id, "split": split,
                "true": int(cls), "left": l, "top": t, "right": r,
                "bottom": btm, "cw": crop.shape[1], "ch": crop.shape[0],
                "b": round(float(b), 6), "c": round(float(c), 6),
                "e": round(float(e), 6),
            }
            for br in BRANCHES:
                out = apply_branch(crop, br)         # op at ORIGINAL size
                out = cv2.resize(out, (INPUT_SIZE, INPUT_SIZE))
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                buf_tensors.append(tfm(rgb))
            buf_rows.append(row)
            if len(buf_rows) >= args.batch:
                flush()
            n_done += 1
    flush()
    fout.close()
    print(f"[sweep] cached {n_done} crops in {(time.time()-t0)/60:.1f} min "
          f"(skipped tiny: {n_skip_small})")

    cfg = {
        "script": "I_gtsdb_eval.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gtsdb_dir": str(gtsdb),
        "margin": args.margin, "min_margin_px": args.min_margin_px,
        "split_boundary": SPLIT_BOUNDARY,
        "pipeline": "crop@orig->stats@orig->op@orig->resize32->BGR2RGB"
                    "->ToTensor->Normalize",
        "stats_source": "enhance.compute_image_stats (live import)",
        "gtsrb_thresholds": GTSRB_THRESHOLDS,
        "torch": torch.__version__, "opencv": cv2.__version__,
    }
    with open(out_csv.with_suffix(".run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ------------------------------------------------------------------
def analyze(args):
    out_csv = Path(args.out_csv)
    rows = []
    with open(out_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["true"] = int(r["true"])
            r["b"], r["c"], r["e"] = float(r["b"]), float(r["c"]), float(r["e"])
            for br in BRANCHES:
                r[f"pred_{br}"] = int(r[f"pred_{br}"])
            rows.append(r)
    calib = [r for r in rows if r["split"] == "calib"]
    ev = [r for r in rows if r["split"] == "eval"]
    print(f"[analyze] crops: calib={len(calib)}  eval={len(ev)}")
    if not calib or not ev:
        print("[FATAL] one split is empty; check gt.txt and split boundary.")
        sys.exit(1)

    sizes = sorted(min(int(r["cw"]), int(r["ch"])) for r in ev)
    print(f"[analyze] eval crop min-side: min={sizes[0]} "
          f"median={sizes[len(sizes)//2]} max={sizes[-1]}")

    # threshold set B: recalibrate on calib split with the paper recipe
    bs = np.array([r["b"] for r in calib])
    cs = np.array([r["c"] for r in calib])
    es = np.array([r["e"] for r in calib])
    T_B = {"T1": float(np.percentile(bs, 15)),
           "T2": float(np.percentile(cs, 15)),
           "T3": float(np.percentile(es, 15)),
           "T4": float(np.percentile(bs, 70))}
    print("\n=== thresholds ===")
    print(f"{'':6s} {'T1=P15(b)':>10s} {'T2=P15(c)':>10s} "
          f"{'T3=P15(e)':>10s} {'T4=P70(b)':>10s}")
    print("A GTSRB" + "".join(f"{GTSRB_THRESHOLDS[k]:11.4f}"
                              for k in ["T1", "T2", "T3", "T4"]))
    print("B GTSDB" + "".join(f"{T_B[k]:11.4f}"
                              for k in ["T1", "T2", "T3", "T4"]))

    classes = sorted({r["true"] for r in ev})
    print(f"[analyze] eval split covers {len(classes)} of 43 classes")

    def method_pred(r, name):
        if name == "va_A":
            br = route_decision(r["b"], r["c"], r["e"], GTSRB_THRESHOLDS)
            return r[f"pred_{br}"], br
        if name == "va_B":
            br = route_decision(r["b"], r["c"], r["e"], T_B)
            return r[f"pred_{br}"], br
        if name == "oracle4":
            for br in BRANCHES:
                if r[f"pred_{br}"] == r["true"]:
                    return r["true"], br
            return r["pred_passthrough"], "passthrough"
        return r[f"pred_{name}"], name

    methods = ["passthrough", "gamma", "clahe", "stretch",
               "va_A", "va_B", "oracle4"]
    results = {}
    print("\n=== GTSDB eval split (crops with official 43-class labels) ===")
    print(f"{'method':12s} {'acc %':>8s} {'macroF1 %':>10s}")
    for m in methods:
        pairs = []
        dist = defaultdict(int)
        for r in ev:
            p, br = method_pred(r, m)
            pairs.append((r["true"], p))
            dist[br] += 1
        acc = 100.0 * sum(1 for t, p in pairs if t == p) / len(pairs)
        f1 = macro_f1(pairs, classes)
        results[m] = {"acc": round(acc, 2), "macro_f1": round(f1, 2)}
        if m in ("va_A", "va_B"):
            results[m]["routing"] = {k: round(100.0 * v / len(ev), 1)
                                     for k, v in sorted(dist.items())}
        print(f"{m:12s} {acc:8.2f} {f1:10.2f}")

    for m in ("va_A", "va_B"):
        d = results[m]["routing"]
        print(f"routing {m}: " + "  ".join(f"{k[:5]}={v:4.1f}%"
                                           for k, v in d.items()))

    d_acc = results["va_A"]["acc"] - results["va_B"]["acc"]
    d_base = results["va_A"]["acc"] - results["passthrough"]["acc"]
    print(f"\ntransferability: acc(VA, GTSRB thresholds) - "
          f"acc(VA, GTSDB-recalibrated) = {d_acc:+.2f} pp")
    print(f"VA_A vs baseline on GTSDB eval: {d_base:+.2f} pp")

    # worst classes under va_A (informational)
    per_cls = defaultdict(lambda: [0, 0])
    for r in ev:
        p, _ = method_pred(r, "va_A")
        per_cls[r["true"]][1] += 1
        if p == r["true"]:
            per_cls[r["true"]][0] += 1
    worst = sorted(((100.0 * c / n, cls, n) for cls, (c, n)
                    in per_cls.items() if n >= 3))[:5]
    print("worst classes for VA_A (acc%, class, n): "
          + "; ".join(f"{a:.0f}%, c{cls}, n={n}" for a, cls, n in worst))

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_calib": len(calib), "n_eval": len(ev),
        "classes_in_eval": classes,
        "margin": args.margin,
        "thresholds_A_gtsrb": GTSRB_THRESHOLDS,
        "thresholds_B_gtsdb": {k: round(v, 4) for k, v in T_B.items()},
        "eval_results": results,
        "transferability_delta_pp": round(d_acc, 2),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] wrote {args.out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtsdb-dir", default=str(GTSDB_DIR_DEFAULT))
    ap.add_argument("--out-csv", default=str(OUT_CSV_DEFAULT))
    ap.add_argument("--out-json", default=str(OUT_JSON_DEFAULT))
    ap.add_argument("--margin", type=float, default=0.10,
                    help="crop margin per side as fraction of box size "
                         "(GTSRB-style); 0 disables")
    ap.add_argument("--min-margin-px", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        run_sweep(args)
    analyze(args)


if __name__ == "__main__":
    main()
