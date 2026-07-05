"""
G_synth_router_data.py -- training data for the LEARNED-ROUTER experiment

WHAT IT DOES
  Takes a stratified, seeded sample of GTSRB TRAINING images (never test
  images, so no leakage into Table 1 or CURE-TSR evaluations), applies the
  paper's five synthetic degradations plus a clean pass, and for every
  (image, condition) records:
    - the three routing statistics b, c, e  (ORIGINAL size, verbatim
      enhance.py::compute_image_stats via direct import of the LIVE module)
    - the rule router's branch decision (paper thresholds)
    - predictions of the frozen CompactCNN for ALL FOUR branches
      (pred, top-1 prob, and prob of the TRUE class).
  The prob-of-true-class per branch is what defines the "best branch" label
  that the learned router will be trained on (H_learned_router.py).

BIT-CONSISTENCY DESIGN (lesson from the CLAHE incident)
  Nothing algorithmic is re-implemented here except the stretch operator
  (a fixed one-line formula) and the routing if/else (four comparisons):
    - degradations   -> imported from LIVE src/degrade.py
    - stats          -> imported from LIVE src/enhance.py (compute_image_stats)
    - gamma / clahe  -> imported from LIVE src/enhance.py
    - model + ckpt   -> revision_utils.load_gtsrb_compactcnn (LIVE)
    - sampling       -> revision_utils.collect_gtsrb_train_samples_stratified
  Recognition pipeline mirrors the CURE DEPLOYMENT path on purpose:
      enhance at ORIGINAL size -> cv2.resize(32) -> BGR2RGB
      -> ToTensor -> Normalize(GTSRB mean/std)
  Rationale: the learned router will be judged on CURE, so its training
  labels must reflect the deployment pipeline, not the Table-1 PIL pipeline.
  (The two differ only in the resize backend; documented choice.)

DETERMINISM
  - The image sample list is fixed by (--per-class, --seed) via the seeded
    stratified collector, so its order is stable across runs and resumes.
  - degrade_noisy uses numpy's GLOBAL RNG: we call np.random.seed(f(seed,i))
    right before it, where i is the image's stable index in the sample list.
  - degrade_mixed receives its own np.random.default_rng(g(seed,i)).
  Therefore every row is reproducible bit-for-bit regardless of --resume.

USAGE (Windows, conda env pcm_sim, from D:\\Project\\traffic_sign\\src):
    python G_synth_router_data.py --smoke     # ~4 images/class, sanity only
    python G_synth_router_data.py             # full (default 120/class)
    python G_synth_router_data.py --resume    # continue an interrupted run

OUTPUT
    outputs_revision\\gtsrb_router_train.csv
    outputs_revision\\gtsrb_router_train.run_config.json
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

# ---- LIVE modules (single source of truth; do not re-implement) ----
from degrade import DEGRADATIONS                       # noqa: E402
from enhance import (                                  # noqa: E402
    apply_clahe, apply_gamma, compute_image_stats,
)
from revision_utils import (                           # noqa: E402
    collect_gtsrb_train_samples_stratified,
    load_gtsrb_compactcnn, GTSRB_MEAN, GTSRB_STD,
)

INPUT_SIZE = 32
THRESHOLDS = {"T1": 0.1206, "T2": 0.1061, "T3": 0.0726, "T4": 0.4085}
BRANCHES = ["passthrough", "gamma", "clahe", "stretch"]
CONDITIONS = ["clean", "lowlight", "foggy", "lowcontrast", "noisy", "mixed"]

OUT_DEFAULT = PROJECT_ROOT / "outputs_revision" / "gtsrb_router_train.csv"

CSV_FIELDS = ["filename", "true", "condition", "b", "c", "e", "rule_branch"]
for _br in BRANCHES:
    CSV_FIELDS += [f"pred_{_br}", f"prob_{_br}", f"ptrue_{_br}"]


def apply_stretch(img_bgr):
    """Verbatim fixed linear stretch of the harness / enhance.py fog branch."""
    f = img_bgr.astype(np.float32) / 255.0
    out = np.clip((f - 0.5) * 1.5 + 0.5, 0, 1)
    return (out * 255).astype(np.uint8)


def apply_branch(img_bgr, branch):
    if branch == "passthrough":
        return img_bgr
    if branch == "gamma":
        return apply_gamma(img_bgr, gamma=0.5)
    if branch == "clahe":
        return apply_clahe(img_bgr)          # live defaults: clip 3.0, tile 8x8
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


def degrade_image(img_bgr, condition, seed, sample_index):
    """Apply one named degradation with per-(image, condition) determinism."""
    if condition == "clean":
        return img_bgr
    if condition == "noisy":
        # degrade_noisy draws from numpy's GLOBAL RNG
        np.random.seed((seed * 1_000_003 + sample_index * 7 + 3) % (2**31 - 1))
        return DEGRADATIONS["noisy"](img_bgr)
    if condition == "mixed":
        rng = np.random.default_rng(seed * 1_000_003 + sample_index * 7 + 5)
        return DEGRADATIONS["mixed"](img_bgr, rng=rng)
    return DEGRADATIONS[condition](img_bgr)


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120,
                    help="stratified sample size per GTSRB class (43 classes)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--batch", type=int, default=128,
                    help="images per forward batch (x4 tensors each)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="4 images/class, quick end-to-end sanity")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cpu"

    per_class = 4 if args.smoke else args.per_class
    samples = collect_gtsrb_train_samples_stratified(
        samples_per_class=per_class, seed=args.seed)
    print(f"[data] stratified GTSRB TRAIN sample: {len(samples)} images "
          f"({per_class}/class, seed={args.seed})")
    print(f"[data] conditions per image: {CONDITIONS}  "
          f"-> {len(samples) * len(CONDITIONS)} rows total")

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    mode = "w"
    if args.resume and out_csv.exists():
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["filename"] + "|" + row["condition"])
        mode = "a"
        print(f"[resume] {len(done)} rows already present, appending")

    model = load_gtsrb_compactcnn(device)
    tfm = build_transform()

    fout = open(out_csv, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()

    buf_rows, buf_tensors, buf_true = [], [], []

    def flush():
        if not buf_rows:
            return
        x = torch.stack(buf_tensors, dim=0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            top_p, top_i = probs.max(dim=1)
        probs = probs.numpy()
        top_p = top_p.numpy()
        top_i = top_i.numpy()
        for j, row in enumerate(buf_rows):
            base = j * 4
            t = buf_true[j]
            for k, br in enumerate(BRANCHES):
                row[f"pred_{br}"] = int(top_i[base + k])
                row[f"prob_{br}"] = round(float(top_p[base + k]), 6)
                row[f"ptrue_{br}"] = round(float(probs[base + k, t]), 6)
            writer.writerow(row)
        fout.flush()
        buf_rows.clear()
        buf_tensors.clear()
        buf_true.clear()

    t0 = time.time()
    n_done = 0
    n_total = len(samples) * len(CONDITIONS) - len(done)
    for i, (fpath, cls) in enumerate(samples):
        img0 = cv2.imread(fpath)
        if img0 is None:
            print(f"[warn] unreadable, skipped: {fpath}")
            continue
        rel = str(Path(fpath).relative_to(PROJECT_ROOT)) \
            if str(fpath).startswith(str(PROJECT_ROOT)) else Path(fpath).name
        for cond in CONDITIONS:
            key = rel + "|" + cond
            if key in done:
                continue
            img = degrade_image(img0, cond, args.seed, i)
            b, c, e = compute_image_stats(img)          # ORIGINAL size
            rule_branch = route_decision(b, c, e, THRESHOLDS)
            row = {
                "filename": rel, "true": int(cls), "condition": cond,
                "b": round(float(b), 6), "c": round(float(c), 6),
                "e": round(float(e), 6), "rule_branch": rule_branch,
            }
            for br in BRANCHES:
                out = apply_branch(img, br)              # ORIGINAL size op
                out = cv2.resize(out, (INPUT_SIZE, INPUT_SIZE))
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                buf_tensors.append(tfm(rgb))
            buf_rows.append(row)
            buf_true.append(int(cls))
            if len(buf_rows) >= args.batch:
                flush()
            n_done += 1
            if n_done % 2000 == 0:
                dt = time.time() - t0
                rate = n_done / dt
                eta = (n_total - n_done) / max(rate, 1e-9)
                print(f"[sweep] {n_done}/{n_total} rows  "
                      f"{rate:.0f} rows/s  ETA {eta/60:.1f} min")
    flush()
    fout.close()
    print(f"[sweep] done: {n_done} new rows in {(time.time()-t0)/60:.1f} min")

    cfg = {
        "script": "G_synth_router_data.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_class": per_class, "seed": args.seed,
        "conditions": CONDITIONS,
        "pipeline": "degrade@orig->stats@orig->branch_op@orig->resize32"
                    "->BGR2RGB->ToTensor->Normalize",
        "stats_source": "enhance.compute_image_stats (live import)",
        "degradations_source": "degrade.DEGRADATIONS (live import)",
        "thresholds": THRESHOLDS,
        "n_rows_this_run": n_done,
        "smoke": bool(args.smoke),
        "torch": torch.__version__, "opencv": cv2.__version__,
    }
    cfg_path = out_csv.with_suffix(".run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] wrote {cfg_path}")

    # ---------- sanity summary (NOT an anchor: train split, new sample) ----
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8", newline="")))
    print("\n=== sanity summary (train split; for eyeballing only) ===")
    print(f"{'condition':12s} {'n':>6s} {'rule_acc':>9s} {'oracle4':>8s}  "
          f"best-branch label distribution")
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        n = len(sub)
        rule_ok = sum(1 for r in sub
                      if int(r[f"pred_{r['rule_branch']}"]) == int(r["true"]))
        orc_ok = sum(1 for r in sub
                     if any(int(r[f"pred_{br}"]) == int(r["true"])
                            for br in BRANCHES))
        lab = defaultdict(int)
        for r in sub:
            vals = [float(r[f"ptrue_{br}"]) for br in BRANCHES]
            lab[BRANCHES[int(np.argmax(vals))]] += 1
        dist = "  ".join(f"{k[:5]}={100.0*v/n:4.1f}%"
                         for k, v in sorted(lab.items(), key=lambda x: -x[1]))
        print(f"{cond:12s} {n:6d} {100.0*rule_ok/n:8.2f}% {100.0*orc_ok/n:7.2f}%  {dist}")
    print("\n[done] next step: python H_learned_router.py")


if __name__ == "__main__":
    main()
