# -*- coding: utf-8 -*-
"""
W_blur_anomaly_diagnostic.py
Diagnostic, run BEFORE any conclusion is drawn from the blur-plus-noise data.

THE ANOMALY
On the blur challenges, adding sensor noise to the image and applying NO
operator raises accuracy: GaussianBlur (severities 3 to 5) goes from 12.53%
at sigma = 0 to 25.33% at sigma = 8. Every operator ranking measured in that
setting is therefore uninterpretable until the anomaly is explained.

TWO HYPOTHESES, VERY DIFFERENT CONSEQUENCES
  (A) Rendering artefact. In a real camera, blur is optical and happens BEFORE
      the sensor, so a genuinely blurred capture still carries sensor noise.
      CURE's blur is applied in post-production to an already captured image,
      which smooths the sensor noise away. If so, the sigma = 0 blur images are
      unrealistically noise free, adding noise makes them MORE realistic, and
      the matched operator for blur must be re-decided in the noisy setting.
  (B) Classifier artefact. The noise acts as a dither that lets the classifier
      extract information from a degenerate input. If so, the effect has nothing
      to do with realism, and no operator conclusion may be drawn from it.

TWO MEASUREMENTS THAT SEPARATE THEM
  1. The Immerkaer noise estimate of the ORIGINAL images, per challenge. Under
     (A) the blur challenges must sit clearly BELOW the clean split's noise
     floor, because the rendering removed the sensor noise. Under (B) there is
     no reason for them to.
  2. Noise added to the CLEAN (ChallengeFree) images, classified with no
     operator. Under (A) accuracy must NOT improve, since clean images already
     carry their sensor noise. Under (B) it may improve there too.

DECISION RULE (fixed before running)
  If the blur challenges are below the clean floor AND noise does not help on
  clean images, hypothesis (A) is supported: the blur-plus-noise setting is the
  realistic one, and the blur experiment may proceed with AdaIR.
  If noise helps on clean images as well, hypothesis (B) is supported: the blur
  experiment is confounded and no operator conclusion is drawn from it.

Writes: outputs_revision/W_blur_diagnostic.json
Run:    python W_blur_anomaly_diagnostic.py       (a few minutes, no deep model)
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict
from math import pi, sqrt
from pathlib import Path

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import (            # noqa: E402
    load_model, build_transform, CHALLENGE_TYPES,
)
from J_local_deep_eval import classify_batch  # noqa: E402
from Q_dcp_branch import scan_images          # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
K = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
SIGMAS = (0, 2, 4, 8, 16)
N_CLEAN = 1352            # the whole ChallengeFree split


def immerkaer(gray):
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    c = cv2.filter2D(gray.astype(np.float64), -1, K,
                     borderType=cv2.BORDER_REPLICATE)[1:-1, 1:-1]
    return float(np.sum(np.abs(c)) * sqrt(pi / 2.0) / (6.0 * (w - 2) * (h - 2)))


def noisy(img, sigma, key):
    if sigma == 0:
        return img
    h = hashlib.sha256(f"{key}|{sigma}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little") % (2 ** 32))
    return np.clip(img.astype(np.float32) +
                   rng.normal(0, float(sigma), img.shape), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--per-cell", type=int, default=200,
                    help="images per (challenge, severity) for the noise scan")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    samples = scan_images(Path(args.cure_root))
    out = {"noise_floor_per_challenge": {}, "clean_noise_scan": {}}

    # ---------- measurement 1: Immerkaer per challenge ----------
    print("=== 1. Immerkaer noise estimate of the ORIGINAL images, per "
          "challenge ===")
    rng = np.random.default_rng(42)
    buckets = defaultdict(list)
    for s in samples:
        buckets[(s["ch"], s["sev"])].append(s)
    per_ch = defaultdict(list)
    for (c, v), lst in buckets.items():
        lst = sorted(lst, key=lambda x: (x["filename"], x["occ"]))
        idx = rng.permutation(len(lst))[:args.per_cell]
        for i in idx:
            im = cv2.imread(str(lst[i]["path"]))
            if im is None:
                continue
            per_ch[c].append(immerkaer(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))
    floor = float(np.median(per_ch[0])) if per_ch.get(0) else float("nan")
    print(f"  clean split (ChallengeFree) noise floor: {floor:.3f}/255\n")
    print(f"  {'challenge':16s}{'median sigma_hat':>18s}{'vs clean floor':>16s}")
    for c in sorted(per_ch):
        if c == 0:
            continue
        m = float(np.median(per_ch[c]))
        rel = m / floor if floor else float("nan")
        flag = "  <== BELOW the floor" if rel < 0.75 else ""
        out["noise_floor_per_challenge"][CHALLENGE_TYPES.get(c, str(c))] = {
            "median_sigma_hat": round(m, 3), "ratio_to_clean_floor": round(rel, 3)}
        print(f"  {CHALLENGE_TYPES.get(c, str(c)):16s}{m:18.3f}{rel:15.2f}x{flag}")
    out["clean_floor"] = round(floor, 3)

    # ---------- measurement 2: does noise help on CLEAN images? ----------
    print("\n=== 2. Noise added to CLEAN images, no operator "
          "(the decisive control) ===")
    device = "cpu"
    model = load_model(args.model, device)
    tfm = build_transform()
    clean = [s for s in samples if s["ch"] == 0][:N_CLEAN]
    imgs = []
    for s in clean:
        im = cv2.imread(str(s["path"]))
        if im is not None:
            imgs.append((im, s["true"], f"{s['filename']}|{s['occ']}|0|0"))
    print(f"  {len(imgs)} clean images")
    print(f"  {'sigma':>6s}{'passthrough acc':>18s}{'delta vs sigma=0':>19s}")
    a0 = None
    for sg in SIGMAS:
        preds, trues = [], []
        for i in range(0, len(imgs), 64):
            chunk = imgs[i:i + 64]
            batch = [noisy(im, sg, k) for im, _, k in chunk]
            p, _ = classify_batch(model, batch, tfm, device)
            preds.extend(p.tolist()); trues.extend([t for _, t, _ in chunk])
        a = 100 * float(np.mean(np.array(preds) == np.array(trues)))
        if a0 is None:
            a0 = a
        out["clean_noise_scan"][str(sg)] = round(a, 2)
        print(f"  {sg:6d}{a:18.2f}{a - a0:+19.2f}")

    # ---------- verdict ----------
    print("\n=== VERDICT (decision rule fixed before running) ===")
    blur_names = ["LensBlur", "GaussianBlur"]
    below = all(out["noise_floor_per_challenge"].get(n, {}).get(
        "ratio_to_clean_floor", 9) < 0.75 for n in blur_names)
    clean_gain = max(out["clean_noise_scan"][str(s)] for s in SIGMAS
                     if s > 0) - out["clean_noise_scan"]["0"]
    helps_clean = clean_gain > 1.0
    print(f"  blur challenges below the clean noise floor : {below}")
    print(f"  best gain from noise on CLEAN images        : {clean_gain:+.2f} "
          f"points -> noise {'HELPS' if helps_clean else 'does NOT help'} "
          f"on clean")
    if below and not helps_clean:
        v = ("(A) RENDERING ARTEFACT. CURE's blur rendering removed the sensor "
             "noise.\n      The noisy setting is the realistic one; the blur "
             "experiment may proceed\n      with AdaIR, and the matched "
             "operator for blur must be re-decided there.")
    elif helps_clean:
        v = ("(B) CLASSIFIER ARTEFACT. Noise improves accuracy even on clean "
             "images.\n      The blur-plus-noise setting is confounded; NO "
             "operator conclusion may be\n      drawn from it, and the "
             "anomaly itself is reported as a finding about the\n      "
             "classifier, not about restoration.")
    else:
        v = ("INCONCLUSIVE: the blur challenges are not clearly below the "
             "floor, yet noise\n      does not help on clean images. Send me "
             "this output; the blur experiment stays\n      on hold.")
    print(f"  -> {v}")
    out["verdict"] = {"blur_below_floor": bool(below),
                      "clean_gain": round(clean_gain, 2),
                      "noise_helps_clean": bool(helps_clean)}

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "W_blur_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {Path(args.outdir) / 'W_blur_diagnostic.json'}")


if __name__ == "__main__":
    main()
