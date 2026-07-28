"""
F_master_sweep_cache.py -- L1 master sweep over CURE-TSR (real, mapped subset)

PURPOSE
  One pass over the CURE-TSR evaluation images that caches, per image:
    - filename, CURE sign id, mapped GTSRB label, challenge id/name, severity
    - the three routing statistics b, c, e   (computed on the 32x32 image,
      exactly as in evaluate_cure_tsr_external.py)
    - the rule-based branch chosen by VA-Adaptive
    - predictions (argmax class) AND softmax top-1 prob for ALL FOUR branches:
      passthrough / gamma / clahe / stretch
  After this cache exists, the following become pure table lookups
  (no further image processing needed):
    oracle upper bound, recoverable-gap analysis, learned-router evaluation,
    misrouting decomposition, worst-class / worst-cell metrics,
    threshold re-analysis, statistic-space scatter figure.

PIPELINE (bit-identical to the LIVE evaluate_cure_tsr_external.py, verified
2026-07-02: the uploaded per-image predictions CSV reproduces every Table 2
number to 0.00, and that harness enhances BEFORE resizing):
    cv2.imread (BGR, ORIGINAL size) -> enhance AT ORIGINAL SIZE
    -> cv2.resize to 32x32 -> cv2 BGR2RGB -> transforms.ToTensor
    -> Normalize(GTSRB mean/std) -> CompactCNN (frozen)
  Routing statistics b, c, e are computed AT ORIGINAL SIZE with the verbatim
  math of enhance.py::compute_image_stats (b=mean/255, c=std/128,
  e=Canny(50,150).mean()/255), because that is exactly what adaptive_enhance
  does inside the published runs.  Thresholds are the paper values.

SELF-CONTAINED
  This script does NOT import enhance.py / evaluate_cure_tsr_external.py,
  to avoid signature drift (fn_va_adaptive in the zip calls adaptive_enhance
  without thresholds).  All operators and the routing rule are re-implemented
  here verbatim, with the same parameters.  Only model.py (CompactCNN) is
  imported from src.

ANCHOR VERIFICATION (runs automatically at the end, or standalone with
  --verify-only):
  Recomputes, from the cache alone, the degraded-average accuracy of the five
  classical methods and compares against the published Table 2 values:
      baseline 52.56 | fixed_gamma 51.11 | fixed_clahe 42.18 |
      fixed_stretch 45.70 | va_adaptive 57.32   (ChallengeFree baseline 80.77)
  A mismatch > 0.05 pp prints a WARNING (it does not abort).  If you see
  warnings, send me the console output and (if available) the exact script
  used for the final Colab run (dump_cure_predictions.py) so I can reconcile.

USAGE (Windows, conda env pcm_sim, from D:\\Project\\traffic_sign\\src):
    python F_master_sweep_cache.py                 # full run
    python F_master_sweep_cache.py --smoke         # quick sanity run
    python F_master_sweep_cache.py --resume        # continue interrupted run
    python F_master_sweep_cache.py --verify-only   # re-check anchors on cache

OUTPUT
    outputs_revision\\cure_master_cache.csv
    outputs_revision\\cure_master_cache.run_config.json
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

# ------------------------------------------------------------------
# Paths / constants (mirror evaluate_cure_tsr_external.py)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import CompactCNN  # noqa: E402

# SCOPE (verified against session records on 2026-07-02):
# The published Table 2/3 came from the harness in evaluate_cure_tsr_external.py,
# which rglobs the PARENT folder datasets/CURE-TSR. On the user's machine that
# folder holds Real_Train AND Real_Test (both real, seq=01), giving 1,352 mapped
# images per challenge x severity cell and 33,800 degraded images in total.
# The classifier never trains on CURE-TSR, so pooling both real splits is a
# legitimate external-evaluation protocol, and it is the published one.
# Real_Test alone yields only 356/cell and does NOT reproduce the anchors.
CURE_TSR_DIR_DEFAULT = PROJECT_ROOT / "datasets" / "CURE-TSR"
MODEL_PATH_DEFAULT = PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"
OUT_DIR_DEFAULT = PROJECT_ROOT / "outputs_revision"

INPUT_SIZE = 32
GTSRB_MEAN = [0.3401, 0.3120, 0.3212]
GTSRB_STD = [0.2725, 0.2609, 0.2669]

THRESHOLDS = {"T1": 0.1206, "T2": 0.1061, "T3": 0.0726, "T4": 0.4085}

# Tolerant: fields may be zero-padded to different widths; extension varies.
FILENAME_PATTERN = re.compile(
    r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.(bmp|png|jpg|jpeg)$", re.IGNORECASE
)
IMAGE_EXTS = ("*.bmp", "*.png", "*.jpg", "*.jpeg")

CHALLENGE_TYPES = {
    0: "ChallengeFree", 1: "Decolorization", 2: "LensBlur", 3: "CodecError",
    4: "Darkening", 5: "DirtyLens", 6: "Exposure", 7: "GaussianBlur",
    8: "Noise", 9: "Rain", 10: "Shadow", 11: "Snow", 12: "Haze",
}

CURE_TO_GTSRB = {3: 9, 6: 14, 11: 12, 12: 17, 13: 13}

# All 12 official CURE-TSR challenge types (was [4,8,9,11,12] for the 5-class
# run). With --resume, the 5 already-cached challenges are skipped by filename
# and only the 7 new ones (Decolorization, LensBlur, CodecError, DirtyLens,
# Exposure, GaussianBlur, Shadow) are computed and appended.
EVAL_CHALLENGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# The original 5 challenges the published Table 2 anchors were computed over.
# Anchor verification restricts to these so the reproducibility check stays
# valid after the cache is extended to all 12 challenges.
ANCHOR_CHALLENGES = [4, 8, 9, 11, 12]
EVAL_SEVERITIES = [1, 2, 3, 4, 5]

BRANCHES = ["passthrough", "gamma", "clahe", "stretch"]

# Published Table 2 anchors (degraded-average accuracy, %)
ANCHORS = {
    "passthrough": 52.56,   # baseline
    "gamma": 51.11,         # fixed_gamma
    "clahe": 42.18,         # fixed_clahe
    "stretch": 45.70,       # fixed_stretch
    "va_rule": 57.32,       # va_adaptive
}
ANCHOR_CHALLENGEFREE_BASELINE = 80.77

CSV_FIELDS = [
    "filename", "cure_sign", "gtsrb_true", "ch_id", "ch_name", "sev",
    "b", "c", "e", "rule_branch",
    "pred_passthrough", "prob_passthrough",
    "pred_gamma", "prob_gamma",
    "pred_clahe", "prob_clahe",
    "pred_stretch", "prob_stretch",
]


# ------------------------------------------------------------------
# Operators (verbatim parameters from evaluate_cure_tsr_external.py)
# ------------------------------------------------------------------
def apply_gamma(img_bgr, gamma=0.5):
    img_norm = img_bgr.astype(np.float32) / 255.0
    out = np.power(img_norm, gamma)
    return (out * 255).astype(np.uint8)


def apply_clahe(img_bgr, clip_limit=3.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge((l_ch, a_ch, b_ch))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


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
        return apply_clahe(img_bgr, clip_limit=3.0, tile_grid=(8, 8))
    if branch == "stretch":
        return apply_stretch(img_bgr)
    raise ValueError(branch)


# ------------------------------------------------------------------
# Statistics + routing rule (verbatim from evaluate_cure_tsr_external.py)
# ------------------------------------------------------------------
def compute_stats(img_bgr):
    """Verbatim math of enhance.py::compute_image_stats, applied to the
    ORIGINAL-size image (this is what adaptive_enhance actually does):
        b = gray.mean()/255 ; c = gray.std()/128 ; e = Canny(50,150).mean()/255
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b = float(gray.mean()) / 255.0
    c = float(gray.std()) / 128.0
    edges = cv2.Canny(gray, 50, 150)
    e = float(edges.mean()) / 255.0
    return b, c, e


def route_decision(b, c, e, T):
    if b < T["T1"]:
        return "gamma"
    elif c < T["T2"]:
        return "clahe"
    elif e < T["T3"] and b > T["T4"]:
        return "stretch"
    else:
        return "passthrough"


# ------------------------------------------------------------------
# Model / transform
# ------------------------------------------------------------------
def _load_checkpoint(model_path, device):
    """torch.load without the FutureWarning: try weights_only=True first."""
    try:
        return torch.load(str(model_path), map_location=device,
                          weights_only=True)
    except TypeError:
        # torch too old to know the kwarg
        return torch.load(str(model_path), map_location=device)
    except Exception:
        # checkpoint holds non-tensor python objects; load explicitly
        return torch.load(str(model_path), map_location=device,
                          weights_only=False)


def load_model(model_path, device):
    model = CompactCNN(num_classes=43)
    ckpt = _load_checkpoint(model_path, device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] loaded {model_path}  params={n_params:,}")
    return model


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])


# ------------------------------------------------------------------
# Sample collection
# ------------------------------------------------------------------
def _iter_image_files(root):
    for ext in IMAGE_EXTS:
        yield from root.rglob(ext)


def collect_samples(cure_root, smoke=False):
    """Return list of dicts for seq=01 (real), mapped signs,
    challenge in {0} + EVAL_CHALLENGES (sev filter for 0 is ignored).
    Scans multiple image extensions; if nothing usable is found, prints a
    diagnostic that reveals exactly what is on disk and why it was rejected."""
    wanted_ch = set([0] + EVAL_CHALLENGES)
    samples = []
    n_files = 0
    n_matched_name = 0
    ext_hist = defaultdict(int)
    unmatched_examples = []

    for fpath in sorted(_iter_image_files(cure_root)):
        n_files += 1
        ext_hist[fpath.suffix.lower()] += 1
        m = FILENAME_PATTERN.match(fpath.name)
        if not m:
            if len(unmatched_examples) < 12:
                unmatched_examples.append(fpath.name)
            continue
        n_matched_name += 1
        seq = int(m.group(1))
        sign = int(m.group(2))
        ch = int(m.group(3))
        sev = int(m.group(4))
        if seq != 1:
            continue
        if sign not in CURE_TO_GTSRB:
            continue
        if ch not in wanted_ch:
            continue
        if ch != 0 and sev not in EVAL_SEVERITIES:
            continue
        samples.append({
            "path": fpath, "filename": fpath.name, "sign": sign,
            "ch": ch, "sev": sev,
        })

    if len(samples) == 0:
        print("\n[DIAGNOSE] no usable samples were collected.")
        print(f"  root scanned: {cure_root}")
        print(f"  image files found: {n_files}")
        print(f"  by extension: {dict(ext_hist)}")
        print(f"  filenames matching seq_sign_ch_lvl_num.<ext>: {n_matched_name}")
        if n_files == 0:
            print("  -> no image files at all under this path.")
            print("     Point --cure-root at the folder holding the real")
            print("     images, e.g.  ...\\CURE-TSR\\Real_Test")
        elif n_matched_name == 0:
            print("  -> files exist but none parse. Example names:")
            for nm in unmatched_examples:
                print(f"       {nm}")
            print("  -> send me these names; I will adjust the parser.")
        else:
            print("  -> names parse but all were filtered out")
            print(f"     (kept: seq==1, sign in {sorted(CURE_TO_GTSRB)},")
            print(f"      challenge in {[0] + EVAL_CHALLENGES}).")
        exs = list(_iter_image_files(cure_root))[:5]
        if exs:
            print("  example full paths found:")
            for p in exs:
                print(f"       {p}")
        return samples

    print(f"[data] scanned {n_files} image files, "
          f"{n_matched_name} parsed, {len(samples)} usable")
    if smoke:
        by_cell = defaultdict(list)
        for s in samples:
            by_cell[(s["ch"], s["sev"])].append(s)
        samples = []
        for k in sorted(by_cell):
            samples.extend(by_cell[k][:40])
    return samples


# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------
def run_sweep(args):
    device = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cure_root = Path(args.cure_root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(cure_root, smoke=args.smoke)
    print(f"[data] collected {len(samples)} images "
          f"(mapped signs={sorted(CURE_TO_GTSRB)}, "
          f"challenges={[0]+EVAL_CHALLENGES})")
    if len(samples) == 0:
        print("[FATAL] no samples found. Check --cure-root.")
        sys.exit(1)

    done = set()
    mode = "w"
    if args.resume and out_csv.exists():
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["filename"])
        mode = "a"
        print(f"[resume] {len(done)} rows already in cache, appending")

    todo = [s for s in samples if s["filename"] not in done]
    print(f"[sweep] {len(todo)} images to process")

    model = load_model(args.model, device)
    tfm = build_transform()

    fout = open(out_csv, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()

    t0 = time.time()
    buf_rows, buf_tensors = [], []

    def flush_batch():
        if not buf_rows:
            return
        x = torch.stack(buf_tensors, dim=0)          # [4*B, 3, 32, 32]
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            top_p, top_i = probs.max(dim=1)
        top_p = top_p.cpu().numpy()
        top_i = top_i.cpu().numpy()
        for j, row in enumerate(buf_rows):
            base = j * 4
            for k, br in enumerate(BRANCHES):
                row[f"pred_{br}"] = int(top_i[base + k])
                row[f"prob_{br}"] = round(float(top_p[base + k]), 6)
            writer.writerow(row)
        fout.flush()
        buf_rows.clear()
        buf_tensors.clear()

    n_done = 0
    for s in todo:
        img = cv2.imread(str(s["path"]))
        if img is None:
            print(f"[warn] unreadable image skipped: {s['path']}")
            continue
        # AUTHORITATIVE ORDER: stats and operators run at ORIGINAL size;
        # the resize to 32x32 happens AFTER enhancement (mirrors the live
        # harness __getitem__ that produced the published per-image CSV).
        b, c, e = compute_stats(img)
        rule_branch = route_decision(b, c, e, THRESHOLDS)

        row = {
            "filename": s["filename"], "cure_sign": s["sign"],
            "gtsrb_true": CURE_TO_GTSRB[s["sign"]],
            "ch_id": s["ch"], "ch_name": CHALLENGE_TYPES[s["ch"]],
            "sev": s["sev"],
            "b": round(b, 6), "c": round(c, 6), "e": round(e, 6),
            "rule_branch": rule_branch,
        }
        for br in BRANCHES:
            out = apply_branch(img, br)
            out = cv2.resize(out, (INPUT_SIZE, INPUT_SIZE))
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            buf_tensors.append(tfm(rgb))
        buf_rows.append(row)

        if len(buf_rows) >= args.batch:
            flush_batch()
        n_done += 1
        if n_done % 2000 == 0:
            dt = time.time() - t0
            rate = n_done / dt
            eta = (len(todo) - n_done) / max(rate, 1e-9)
            print(f"[sweep] {n_done}/{len(todo)}  "
                  f"{rate:.0f} img/s  ETA {eta/60:.1f} min")

    flush_batch()
    fout.close()
    dt = time.time() - t0
    print(f"[sweep] done: {n_done} images in {dt/60:.1f} min")

    cfg = {
        "script": "F_master_sweep_cache.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cure_root": str(cure_root),
        "model": str(args.model),
        "pipeline": "imread->enhance@ORIGINAL->resize32->BGR2RGB->ToTensor->Normalize",
        "thresholds": THRESHOLDS,
        "mapping": CURE_TO_GTSRB,
        "challenges": EVAL_CHALLENGES,
        "severities": EVAL_SEVERITIES,
        "operators": {
            "gamma": 0.5, "clahe_clip": 3.0, "clahe_tile": [8, 8],
            "stretch": "(f-0.5)*1.5+0.5",
        },
        "n_rows_this_run": n_done,
        "smoke": bool(args.smoke),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
    }
    cfg_path = out_csv.with_suffix(".run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] wrote {cfg_path}")


# ------------------------------------------------------------------
# Anchor verification + bonus numbers (oracle, ceiling)
# ------------------------------------------------------------------
def macro_f1(y_true, y_pred, classes):
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return 100.0 * sum(f1s) / len(f1s)


def verify(args):
    out_csv = Path(args.out)
    if not out_csv.exists():
        print(f"[FATAL] cache not found: {out_csv}")
        sys.exit(1)
    rows = []
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["ch_id"] = int(r["ch_id"])
            r["sev"] = int(r["sev"])
            r["gtsrb_true"] = int(r["gtsrb_true"])
            for br in BRANCHES:
                r[f"pred_{br}"] = int(r[f"pred_{br}"])
            rows.append(r)
    deg = [r for r in rows if r["ch_id"] != 0]
    cf = [r for r in rows if r["ch_id"] == 0]
    classes = sorted(set(CURE_TO_GTSRB.values()))
    print(f"[verify] cache rows: total={len(rows)}  degraded={len(deg)}  "
          f"challengefree={len(cf)}")
    if args.smoke:
        print("[verify] NOTE: smoke cache; anchors will NOT match. "
              "Use only to confirm the script runs end to end.")

    def method_pred(r, name):
        if name == "va_rule":
            return r[f"pred_{r['rule_branch']}"]
        return r[f"pred_{name}"]

    def cellwise_deg_avg(name, challenges=None):
        cells = defaultdict(lambda: [0, 0])
        for r in deg:
            if challenges is not None and r["ch_id"] not in challenges:
                continue
            key = (r["ch_id"], r["sev"])
            cells[key][1] += 1
            if method_pred(r, name) == r["gtsrb_true"]:
                cells[key][0] += 1
        accs = [100.0 * c / n for c, n in cells.values() if n > 0]
        return sum(accs) / len(accs) if accs else float("nan")

    print("\n=== ANCHOR CHECK vs published Table 2 (degraded-average acc, %) ===")
    print(f"    (restricted to the original 5 challenges {ANCHOR_CHALLENGES}; "
          f"the cache may now hold all 12)")
    print(f"{'method':14s} {'cache':>8s} {'paper':>8s} {'diff':>7s}")
    for name, expected in ANCHORS.items():
        got = cellwise_deg_avg(name, challenges=ANCHOR_CHALLENGES)
        diff = got - expected
        flag = "" if abs(diff) <= 0.05 else "  << WARNING"
        print(f"{name:14s} {got:8.2f} {expected:8.2f} {diff:+7.2f}{flag}")

    if cf:
        cf_acc = 100.0 * sum(
            1 for r in cf if r["pred_passthrough"] == r["gtsrb_true"]
        ) / len(cf)
        d = cf_acc - ANCHOR_CHALLENGEFREE_BASELINE
        flag = "" if abs(d) <= 0.05 else "  << WARNING"
        print(f"{'CF baseline':14s} {cf_acc:8.2f} "
              f"{ANCHOR_CHALLENGEFREE_BASELINE:8.2f} {d:+7.2f}{flag}")

    # ---- bonus: oracle upper bound + ceiling + per-challenge table ----
    print("\n=== BONUS: oracle / ceiling (new numbers for the paper) ===")

    def oracle_correct(r):
        return any(r[f"pred_{br}"] == r["gtsrb_true"] for br in BRANCHES)

    cells = defaultdict(lambda: [0, 0])
    for r in deg:
        key = (r["ch_id"], r["sev"])
        cells[key][1] += 1
        if oracle_correct(r):
            cells[key][0] += 1
    oracle_avg = sum(100.0 * c / n for c, n in cells.values()) / len(cells)
    va_avg = cellwise_deg_avg("va_rule")
    base_avg = cellwise_deg_avg("passthrough")
    print(f"oracle (best-of-4 per image) deg-avg : {oracle_avg:6.2f}")
    print(f"va_rule deg-avg                      : {va_avg:6.2f}")
    print(f"baseline deg-avg                     : {base_avg:6.2f}")
    if cf:
        print(f"ceiling (ChallengeFree baseline acc) : {cf_acc:6.2f}")
        gap_total = cf_acc - base_avg
        if gap_total > 0:
            print(f"recoverable gap captured by VA       : "
                  f"{100.0*(va_avg-base_avg)/gap_total:6.1f} % of gap")
            print(f"recoverable gap captured by oracle   : "
                  f"{100.0*(oracle_avg-base_avg)/gap_total:6.1f} % of gap")

    print("\nper-challenge deg-avg (mean over 5 severities):")
    hdr = f"{'challenge':12s}" + "".join(
        f"{m:>9s}" for m in ["base", "gamma", "clahe", "stretch", "va", "oracle"])
    print(hdr)
    for ch in EVAL_CHALLENGES:
        line = f"{CHALLENGE_TYPES[ch]:12s}"
        for m in ["passthrough", "gamma", "clahe", "stretch", "va_rule"]:
            sub = [r for r in deg if r["ch_id"] == ch]
            per_sev = defaultdict(lambda: [0, 0])
            for r in sub:
                per_sev[r["sev"]][1] += 1
                if method_pred(r, m) == r["gtsrb_true"]:
                    per_sev[r["sev"]][0] += 1
            vals = [100.0 * c / n for c, n in per_sev.values()]
            line += f"{sum(vals)/len(vals):9.1f}"
        sub = [r for r in deg if r["ch_id"] == ch]
        per_sev = defaultdict(lambda: [0, 0])
        for r in sub:
            per_sev[r["sev"]][1] += 1
            if oracle_correct(r):
                per_sev[r["sev"]][0] += 1
        vals = [100.0 * c / n for c, n in per_sev.values()]
        line += f"{sum(vals)/len(vals):9.1f}"
        print(line)

    print("\nrouting distribution per challenge (rule router):")
    for ch in [0] + EVAL_CHALLENGES:
        sub = [r for r in rows if r["ch_id"] == ch]
        if not sub:
            continue
        cnt = defaultdict(int)
        for r in sub:
            cnt[r["rule_branch"]] += 1
        tot = len(sub)
        parts = "  ".join(f"{k[:5]}={100.0*v/tot:4.1f}%"
                          for k, v in sorted(cnt.items(), key=lambda x: -x[1]))
        print(f"  {CHALLENGE_TYPES[ch]:14s} n={tot:5d}  {parts}")

    print("\n[verify] done. Send me this console output together with "
          "cure_master_cache.csv (zipped).")



def verify_determinism(args):
    """Re-run the FULL classical pipeline for one already-cached cell and compare
    byte-for-byte against the cache. Any difference means the local environment
    (code / weights / libraries) drifted since the cached run, in which case
    appending new challenges would mix two conditions -- STOP and regenerate all
    12 challenges fresh instead. Run this BEFORE the 12-class --resume sweep."""
    out_csv = Path(args.out)
    if not out_csv.exists():
        print(f"[FATAL] cache not found: {out_csv}"); sys.exit(1)
    try:
        vch, vsev = (int(x) for x in str(args.verify_cell).split(","))
    except Exception:
        print("[FATAL] --verify-cell must be 'ch,sev', e.g. 12,5"); sys.exit(1)

    cached = {}
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["ch_id"]) == vch and int(r["sev"]) == vsev:
                cached[r["filename"]] = r
    if not cached:
        print(f"[FATAL] no cached rows for cell ch={vch} sev={vsev}; pick a cell "
              f"already in the cache with --verify-cell.")
        sys.exit(1)
    print(f"[determinism] cell ch={vch}({CHALLENGE_TYPES.get(vch,'?')}) sev={vsev}: "
          f"{len(cached)} cached rows to re-verify")

    cure_root = Path(args.cure_root)
    all_samples = collect_samples(cure_root, smoke=False)
    samples = [s for s in all_samples
               if s["ch"] == vch and s["sev"] == vsev and s["filename"] in cached]
    if not samples:
        print("[FATAL] could not find the cell's images on disk under "
              f"{cure_root}. Point --cure-root at the same folder used for the "
              "cached run."); sys.exit(1)

    device = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    model = load_model(args.model, device)
    tfm = build_transform()

    def canon_from_cached(r):
        t = [f"{float(r['b']):.6f}", f"{float(r['c']):.6f}", f"{float(r['e']):.6f}",
             r["rule_branch"]]
        for br in BRANCHES:
            t.append(str(int(r[f"pred_{br}"])))
        for br in BRANCHES:
            t.append(f"{float(r[f'prob_{br}']):.6f}")
        return tuple(t)

    # regroup cached rows for this cell by filename, keeping BOTH occurrences of
    # any duplicate name (Real_Train/Real_Test collisions); compare as a set.
    from collections import defaultdict as _dd
    cached_by_fn = _dd(list)
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["ch_id"]) == vch and int(r["sev"]) == vsev:
                cached_by_fn[r["filename"]].append(canon_from_cached(r))

    fresh_by_fn = _dd(list)
    for s2 in samples:
        img = cv2.imread(str(s2["path"]))
        if img is None:
            continue
        b, c, e = compute_stats(img)
        rb = route_decision(b, c, e, THRESHOLDS)
        tensors = []
        for br in BRANCHES:
            out = apply_branch(img, br)
            out = cv2.resize(out, (INPUT_SIZE, INPUT_SIZE))
            tensors.append(tfm(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
        x = torch.stack(tensors, 0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            tp, ti = probs.max(dim=1)
        t = [f"{round(b,6):.6f}", f"{round(c,6):.6f}", f"{round(e,6):.6f}", rb]
        for kk in range(len(BRANCHES)):
            t.append(str(int(ti[kk])))
        for kk in range(len(BRANCHES)):
            t.append(f"{round(float(tp[kk]),6):.6f}")
        fresh_by_fn[s2["filename"]].append(tuple(t))

    n_fn = 0; n_fn_ok = 0; n_tuples = 0; n_tuples_ok = 0; bad = []
    all_fn = set(cached_by_fn) | set(fresh_by_fn)
    for fn in all_fn:
        cl = sorted(cached_by_fn.get(fn, []))
        fl = sorted(fresh_by_fn.get(fn, []))
        n_fn += 1
        n_tuples += max(len(cl), len(fl))
        if cl == fl:
            n_fn_ok += 1
            n_tuples_ok += len(cl)
        else:
            if len(bad) < 8:
                bad.append((fn, len(cl), len(fl)))

    print("\n=== DETERMINISM CHECK (F, classical pipeline; per-(filename,occurrence)) ===")
    print(f"  distinct filenames in cell : {n_fn}")
    print(f"  filenames fully matching   : {n_fn_ok}")
    print(f"  filenames differing        : {n_fn - n_fn_ok}")
    print(f"  image-rows compared        : {n_tuples}   matched: {n_tuples_ok}")
    if bad:
        print("  first differing filenames (name | #cached | #fresh):")
        for fn, a, b_ in bad:
            print(f"    {fn}  {a}  {b_}")
    if n_fn > 0 and n_fn_ok == n_fn:
        print("\n  [OK] environment is STABLE (every filename's occurrences match the\n"
              "       cache as a set). Safe to run the 12-class sweep with --resume:\n"
              "       the cached 5 challenges are skipped and only the 7 new ones\n"
              "       are appended.")
    else:
        print("\n  [STOP] genuine mismatch remains after accounting for duplicate\n"
              "         filenames. This would indicate real drift. Send me this\n"
              "         output before running the sweep.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(CURE_TSR_DIR_DEFAULT))
    ap.add_argument("--model", default=str(MODEL_PATH_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT / "cure_master_cache.csv"))
    ap.add_argument("--batch", type=int, default=128,
                    help="images per forward batch (x4 tensors)")
    ap.add_argument("--threads", type=int, default=0,
                    help=">0 to set torch CPU threads")
    ap.add_argument("--smoke", action="store_true",
                    help="max 40 images per cell, for a quick end-to-end test")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="re-run one already-cached cell and diff it "
                         "byte-for-byte against the cache to detect "
                         "environment drift before appending 7 new "
                         "challenges")
    ap.add_argument("--verify-cell", default="12,5",
                    help="challenge,severity cell for --verify-determinism "
                         "(must already be in the cache; default Haze sev5)")
    args = ap.parse_args()

    if args.verify_determinism:
        verify_determinism(args)
        return
    if args.verify_only:
        verify(args)
        return
    run_sweep(args)
    verify(args)


if __name__ == "__main__":
    main()
