# -*- coding: utf-8 -*-
"""
Q_dcp_branch.py
Pre-registered fifth-operator experiment (Evaluation Protocol, Part 8;
registered 2026-07-09). Per-image predictions for the Dark Channel Prior
operator (He et al. [5]) over ALL CURE-TSR cells (12 challenges x 5 severities
+ ChallengeFree), at original crop resolution before the 32x32 resize, with the
publication's parameters FROZEN:

    dark-channel patch 15x15 | omega = 0.95 | t0 = 0.1
    atmospheric light A: per-channel mean over brightest 0.1% dark-channel px
    transmission refinement: guided filter, radius 60, eps 1e-3

No constant above may be changed (Protocol Part 8). Degeneracy statistics
(t-clip fraction, output std) are logged per image for the pre-registered
failure criterion; the criterion itself is judged on development folds only.

THE GUIDED FILTER
  He and Sun, Guided Image Filtering, TPAMI 2013, single-channel guide, built
  here from cv2.boxFilter. It is written out rather than imported so that the
  refinement can never depend on whether an optional module happens to be
  installed in a given runtime. Border handling is OpenCV's boxFilter default,
  BORDER_REFLECT_101.

  NOTE TO SELF, NOT TO THE PAPER: this was checked against
  cv2.ximgproc.guidedFilter at OpenCV 4.13.0 on crops from 11x11 to 285x211.
  The transmission maps agree to 1e-4..2e-3 and the recovered images differ by
  at most ONE grey level out of 255. The two are the same filter; the residue is
  float rounding. So there is no need to install opencv-contrib anywhere, and
  merged_per_image.csv, which was built under opencv-python, stays valid.

Writes:  outputs_revision/dcp_cure.csv
         (filename, occ, ch, sev, true, pred_dcp, prob_dcp, t_clip_frac, out_std)
         outputs_revision/Q_dcp.run_config.json

Run:     python Q_dcp_branch.py --limit 500      (speed probe, appends)
         python Q_dcp_branch.py --fresh          (full run, overwrites)
         python Q_dcp_branch.py --resume         (continue after interrupt)
Safety:  refuses to overwrite an existing dcp_cure.csv unless --fresh.
"""
import argparse, csv, json, re, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import (  # noqa: E402
    load_model, build_transform, INPUT_SIZE, CURE_TO_GTSRB, CHALLENGE_TYPES,
)

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
FILENAME_PATTERN = re.compile(
    r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.(bmp|png|jpg|jpeg)$", re.IGNORECASE)

# ---- frozen DCP constants (Protocol Part 8; DO NOT EDIT) ----
DCP_PATCH = 15
DCP_OMEGA = 0.95
DCP_T0 = 0.1
GF_RADIUS = 60
GF_EPS = 1e-3



def guided_filter_gray(guide, src, radius, eps):
    """He and Sun, TPAMI 2013, single-channel guide. Assembled from cv2.boxFilter
    so the refinement never rests on an optional module."""
    k = (2 * radius + 1, 2 * radius + 1)
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mean_I = cv2.boxFilter(I, -1, k)
    mean_p = cv2.boxFilter(p, -1, k)
    var_I = cv2.boxFilter(I * I, -1, k) - mean_I * mean_I
    cov_Ip = cv2.boxFilter(I * p, -1, k) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def dcp_enhance(img_bgr):
    """He et al. [5] dark-channel-prior dehazing with frozen constants.
    Returns (enhanced uint8 BGR, t_clip_frac, out_std_norm)."""
    I = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DCP_PATCH, DCP_PATCH))

    # 1) dark channel of I
    dark = cv2.erode(I.min(axis=2), kernel)

    # 2) atmospheric light: per-channel mean over brightest 0.1% dark pixels
    n = dark.size
    k = max(1, int(round(0.001 * n)))
    idx = np.argpartition(dark.reshape(-1), n - k)[n - k:]
    ys, xs = np.unravel_index(idx, dark.shape)
    A = I[ys, xs].mean(axis=0)                      # (3,)
    A = np.maximum(A, 1e-3)

    # 3) transmission from dark channel of I/A
    dark_norm = cv2.erode((I / A[None, None, :]).min(axis=2), kernel)
    t = 1.0 - DCP_OMEGA * dark_norm

    # 4) refinement (frozen: radius 60, eps 1e-3)
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = guided_filter_gray(guide, t.astype(np.float32), GF_RADIUS, GF_EPS)

    t_clip_frac = float(np.mean(t <= DCP_T0 + 1e-6))
    t = np.maximum(t, DCP_T0)

    # 5) recover J = (I - A)/t + A
    J = (I - A[None, None, :]) / t[..., None] + A[None, None, :]
    J = np.clip(J, 0.0, 1.0)
    out_std = float(J.std())
    return (J * 255.0).astype(np.uint8), t_clip_frac, out_std


def scan_images(cure_root):
    """Replicates F's filter and F-order (sorted by full path); assigns the
    occurrence index occ within (ch, sev, filename) so rows join safely with
    merged_per_image.csv despite the 9,256 duplicate filenames."""
    files = []
    for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
        files.extend(cure_root.rglob(ext))
    files = sorted(files, key=lambda p: str(p))
    samples, counter = [], defaultdict(int)
    for fp in files:
        m = FILENAME_PATTERN.match(fp.name)
        if not m:
            continue
        seq, sign, ch, sev = (int(m.group(1)), int(m.group(2)),
                              int(m.group(3)), int(m.group(4)))
        if seq != 1 or sign not in CURE_TO_GTSRB:
            continue
        if ch not in CHALLENGE_TYPES or (ch != 0 and not 1 <= sev <= 5):
            continue
        key = (ch, sev, fp.name)
        occ = counter[key]
        counter[key] += 1
        samples.append({"path": fp, "filename": fp.name, "ch": ch,
                        "sev": sev, "occ": occ,
                        "true": CURE_TO_GTSRB[sign]})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="process at most N pending images this run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="allow overwriting an existing dcp_cure.csv")
    args = ap.parse_args()

    out_csv = OUT_DIR / "dcp_cure.csv"
    if out_csv.exists() and not args.resume and not args.fresh:
        sys.exit(f"[SAFETY] {out_csv} exists. Use --resume to continue or "
                 f"--fresh to overwrite deliberately.")

    device = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    model = load_model(args.model, device)
    tfm = build_transform()

    samples = scan_images(Path(args.cure_root))
    print(f"[data] usable images: {len(samples)} "
          f"(expect 82472 for the full 12-challenge set)")
    print(f"[dcp ] guided filter: radius {GF_RADIUS}, eps {GF_EPS} "
          f"(patch {DCP_PATCH}, omega {DCP_OMEGA}, t0 {DCP_T0})")

    done = set()
    if args.resume and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"])))
        print(f"[resume] {len(done)} rows already in cache")
    pending = [s for s in samples
               if (s["filename"], s["occ"], s["ch"], s["sev"]) not in done]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f"[plan] processing {len(pending)} images this run")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and out_csv.exists()) else "w"
    fout = open(out_csv, mode, newline="", encoding="utf-8")
    w = csv.writer(fout)
    if mode == "w":
        w.writerow(["filename", "occ", "ch", "sev", "true",
                    "pred_dcp", "prob_dcp", "t_clip_frac", "out_std"])

    t0 = time.time()
    buf_meta, buf_tensor = [], []
    n_done = 0

    def flush():
        nonlocal n_done
        if not buf_meta:
            return
        x = torch.stack(buf_tensor, 0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            tp, ti = probs.max(dim=1)
        for i, meta in enumerate(buf_meta):
            w.writerow([meta["filename"], meta["occ"], meta["ch"], meta["sev"],
                        meta["true"], int(ti[i]), round(float(tp[i]), 6),
                        round(meta["clip"], 4), round(meta["std"], 4)])
        n_done += len(buf_meta)
        buf_meta.clear(); buf_tensor.clear()
        if n_done % 2048 < args.batch:
            r = n_done / max(time.time() - t0, 1e-9)
            eta = (len(pending) - n_done) / max(r, 1e-9) / 60
            print(f"[dcp] {n_done} done  {r:.1f} img/s  ETA {eta:.1f} min "
                  f"for remaining {len(pending) - n_done}")

    for s in pending:
        img = cv2.imread(str(s["path"]))
        if img is None:
            continue
        enh, clip_frac, out_std = dcp_enhance(img)
        enh = cv2.resize(enh, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
        buf_meta.append({**{k: s[k] for k in
                            ("filename", "occ", "ch", "sev", "true")},
                         "clip": clip_frac, "std": out_std})
        buf_tensor.append(tfm(rgb))
        if len(buf_meta) >= args.batch:
            flush()
    flush()
    fout.close()
    dt = (time.time() - t0) / 60
    print(f"[done] processed {n_done} images in {dt:.1f} min")

    # run config for reproducibility
    cfg = {"script": "Q_dcp_branch.py", "protocol": "Evaluation Protocol Part 8",
           "constants": {"patch": DCP_PATCH, "omega": DCP_OMEGA, "t0": DCP_T0,
                          "gf_radius": GF_RADIUS, "gf_eps": GF_EPS},
           "guided_filter": "He and Sun TPAMI 2013 via cv2.boxFilter, "
                            "border reflect101",
           "opencv": cv2.__version__, "torch": torch.__version__,
           "images_this_run": n_done, "minutes": round(dt, 1)}
    with open(OUT_DIR / "Q_dcp.run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- summary from the (possibly now-complete) cache ----
    rows = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    ch = np.array([int(r["ch"]) for r in rows])
    sev = np.array([int(r["sev"]) for r in rows])
    tru = np.array([int(r["true"]) for r in rows])
    pr = np.array([int(r["pred_dcp"]) for r in rows])
    clipf = np.array([float(r["t_clip_frac"]) for r in rows])
    ostd = np.array([float(r["out_std"]) for r in rows])
    deg_ch = sorted(set(ch[ch > 0].tolist()))
    print(f"\n=== dcp on CURE (cache rows: {len(rows)}) ===")
    print(f"  degeneracy stats (all rows): "
          f"share with t clipped on >=80% px: "
          f"{100*np.mean(clipf >= 0.8):.1f}%  | share out_std < 5/255: "
          f"{100*np.mean(ostd < 5/255):.1f}%")
    print(f"  (pre-registered failure criterion evaluates these on the "
          f"development folds only)")
    if deg_ch:
        accs = []
        print("  per-challenge (mean over severities):")
        for cc in deg_ch:
            cell = [100*np.mean(pr[(ch == cc) & (sev == ss)] ==
                                tru[(ch == cc) & (sev == ss)])
                    for ss in range(1, 6)
                    if np.sum((ch == cc) & (sev == ss)) > 0]
            accs.extend(cell)
            print(f"    {CHALLENGE_TYPES.get(cc, cc):14s} {np.mean(cell):5.2f}")
        print(f"  degraded-average (cell-averaged): {np.mean(accs):.2f}")
    cf = ch == 0
    if cf.sum():
        print(f"  ChallengeFree accuracy: {100*np.mean(pr[cf]==tru[cf]):.2f}")
    print("\nNext: send dcp_cure.csv (zip) + this console output; V-A "
          "(oracle-of-5) and V-B (low-contrast branch -> DCP) are computed "
          "from the merged join per Protocol Part 8.")


if __name__ == "__main__":
    main()
