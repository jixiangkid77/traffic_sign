# -*- coding: utf-8 -*-
"""
X_noise_domain_shift.py
Measures the high-frequency / noise domain shift between the classifier's
training distribution (GTSRB) and the evaluation data (CURE-TSR), and repairs
two methodological weaknesses of the earlier diagnostic (W).

WHY
W established that CURE's rendered challenges sit far below the clean split's
Immerkaer level (GaussianBlur 0.21x, LensBlur 0.23x, Darkening 0.27x, Haze and
Rain 0.41x) and that adding noise to CLEAN images appeared to raise accuracy by
1.63 points. Two problems with that diagnostic:

  (1) NO STATISTICS. The 1.63-point gain was compared to an arbitrary threshold
      with no test. The verdict that decided the fate of the blur experiment
      therefore rested on an untested point estimate. Repaired here with an
      exact McNemar test and a paired bootstrap interval.

  (2) THE ESTIMATOR MEASURES HIGH-FREQUENCY ENERGY, NOT NOISE. Immerkaer's
      kernel also responds to edges, so a blurred image scores low both because
      its noise was smoothed and because its edges were smoothed. The claim
      "the rendering removed the sensor noise" was therefore beyond the
      evidence. Repaired here with a second, edge-insensitive estimator: the
      10th percentile of local 5x5 standard deviations, which is dominated by
      flat regions and hence by noise.

THE MISSING NUMBER
The classifier was trained on GTSRB. If GTSRB images carry substantially more
high-frequency energy than the rendered CURE challenges, then every operator
that AMPLIFIES high-frequency content (CLAHE, and DCP through its division by
the transmission) enjoys a systematic advantage on CURE that would not exist on
data matched to the training distribution. That would be a confound affecting
the MAIN results, not only the noise experiment. This script measures it.

MEASUREMENTS
  1. Both estimators on GTSRB training images and on CURE (clean and each
     challenge), computed at the 32x32 input resolution the classifier actually
     sees, and additionally at original resolution for reference.
  2. A fine sigma scan on the clean CURE split with exact McNemar and paired
     bootstrap against sigma = 0, to establish whether noise genuinely helps.
  3. The same scan restricted to the challenge with the largest deficit
     (GaussianBlur) and to Haze, to see whether the gain tracks the deficit.

Writes: outputs_revision/X_noise_domain_shift.json
Run:    python X_noise_domain_shift.py --gtsrb-root <path to GTSRB training>
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict
from math import comb, pi, sqrt
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from F_master_sweep_cache import (            # noqa: E402
    load_model, build_transform, INPUT_SIZE, CHALLENGE_TYPES,
)
from J_local_deep_eval import classify_batch  # noqa: E402
from Q_dcp_branch import scan_images          # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"
IMM_K = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
SIGMAS = (0, 1, 2, 3, 4, 6, 8, 12, 16)
B, SEED = 5000, 42


def immerkaer(gray):
    """High-frequency residual energy (responds to noise AND edges)."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    c = cv2.filter2D(gray.astype(np.float64), -1, IMM_K,
                     borderType=cv2.BORDER_REPLICATE)[1:-1, 1:-1]
    return float(np.sum(np.abs(c)) * sqrt(pi / 2.0) / (6.0 * (w - 2) * (h - 2)))


def flat_sigma(gray, win=5, pct=10):
    """Edge-insensitive noise estimate: a low percentile of the local standard
    deviation, which is dominated by flat regions and therefore by noise."""
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (win, win))
    mu2 = cv2.blur(g * g, (win, win))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    b = win // 2
    if sd.shape[0] <= 2 * b or sd.shape[1] <= 2 * b:
        return float(np.percentile(sd, pct))
    return float(np.percentile(sd[b:-b, b:-b], pct))


def noisy(img, sigma, key):
    """Identical seed formula to T/U/V."""
    if sigma == 0:
        return img
    h = hashlib.sha256(f"{key}|{sigma}".encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2 ** 32)
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) +
                   rng.normal(0.0, float(sigma), img.shape),
                   0, 255).astype(np.uint8)


def both_estimators(img_bgr):
    """Returns (immerkaer, flat) at 32x32 (what the classifier sees) and at
    original resolution."""
    g_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    g_32 = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    return (immerkaer(g_32), flat_sigma(g_32),
            immerkaer(g_orig), flat_sigma(g_orig))


def mcnemar(pa, pb, y):
    b = int(np.sum((pa == y) & (pb != y)))
    c = int(np.sum((pa != y) & (pb == y)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n <= 1000:
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    else:
        from math import erfc
        p = erfc(abs(b - c) / sqrt(n) / sqrt(2))
    return b, c, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--gtsrb-root", default=str(PROJECT_ROOT / "data" / "gtsrb"),
                    help="GTSRB root (searched recursively for ppm/png/jpg); "
                         "the degraded siblings gtsrb_foggy/_lowlight/_noisy/"
                         "... are deliberately NOT included")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--n-gtsrb", type=int, default=2000)
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    out = {"note": "immerkaer = high-frequency energy (noise AND edges); "
                   "flat = edge-insensitive noise estimate"}

    # ---------- 1. GTSRB (the training distribution) ----------
    print("=== 1. The training distribution (GTSRB) ===")
    gr = Path(args.gtsrb_root)
    files = []
    if gr.exists():
        for ext in ("*.ppm", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(gr.rglob(ext))
    if not files:
        print(f"  [!] no images found under {gr}")
        print("      pass the correct folder with --gtsrb-root "
              "<path to GTSRB training images>")
        print("      (this is the decisive measurement; the rest still runs)")
        gt = None
    else:
        rng = np.random.default_rng(SEED)
        pick = [files[i] for i in
                rng.permutation(len(files))[:args.n_gtsrb]]
        vals = []
        for p in pick:
            im = cv2.imread(str(p))
            if im is not None:
                vals.append(both_estimators(im))
        a = np.array(vals)
        gt = {"n": len(vals), "n_found": len(files),
              "immerkaer_32": float(np.median(a[:, 0])),
              "flat_32": float(np.median(a[:, 1])),
              "immerkaer_orig": float(np.median(a[:, 2])),
              "flat_orig": float(np.median(a[:, 3]))}
        print(f"  {len(vals)} GTSRB images sampled from {len(files)} found "
              f"under {gr}")
        # breakdown by top-level subfolder, so the train/test structure is
        # visible and any difference between the splits is caught
        sub = defaultdict(list)
        for p, v in zip(pick, vals):
            try:
                rel = p.relative_to(gr).parts
                top = rel[0] if len(rel) > 1 else "(root)"
            except Exception:
                top = "(root)"
            sub[top].append(v)
        if len(sub) > 1 or True:
            print(f"  {'subfolder':28s}{'n':>7s}{'immerkaer_32':>14s}"
                  f"{'flat_32':>10s}")
            gt["by_subfolder"] = {}
            for k2 in sorted(sub):
                aa = np.array(sub[k2])
                gt["by_subfolder"][k2] = {
                    "n": len(aa),
                    "immerkaer_32": round(float(np.median(aa[:, 0])), 3),
                    "flat_32": round(float(np.median(aa[:, 1])), 3)}
                print(f"  {k2:28s}{len(aa):7d}"
                      f"{np.median(aa[:, 0]):14.3f}{np.median(aa[:, 1]):10.3f}")
        print(f"  at 32x32 (what the classifier sees):  "
              f"immerkaer {gt['immerkaer_32']:.3f}   flat-noise "
              f"{gt['flat_32']:.3f}")
        print(f"  at original resolution:               "
              f"immerkaer {gt['immerkaer_orig']:.3f}   flat-noise "
              f"{gt['flat_orig']:.3f}")
    out["gtsrb"] = gt

    # ---------- 2. CURE, both estimators, at 32x32 ----------
    print("\n=== 2. CURE-TSR, both estimators, at 32x32 ===")
    samples = scan_images(Path(args.cure_root))
    rng = np.random.default_rng(SEED)
    buckets = defaultdict(list)
    for s in samples:
        buckets[(s["ch"], s["sev"])].append(s)
    per_ch = defaultdict(list)
    for (c, v), lst in buckets.items():
        lst = sorted(lst, key=lambda x: (x["filename"], x["occ"]))
        for i in rng.permutation(len(lst))[:args.per_cell]:
            im = cv2.imread(str(lst[i]["path"]))
            if im is not None:
                per_ch[c].append(both_estimators(im))
    cure = {}
    for c in sorted(per_ch):
        a = np.array(per_ch[c])
        cure[CHALLENGE_TYPES.get(c, str(c))] = {
            "immerkaer_32": float(np.median(a[:, 0])),
            "flat_32": float(np.median(a[:, 1]))}
    out["cure"] = cure
    cf = cure.get("ChallengeFree", cure.get("0"))
    print(f"  {'challenge':16s}{'immerkaer':>11s}{'flat-noise':>12s}"
          f"{'flat vs clean':>15s}{'flat vs GTSRB':>15s}")
    for k, v in cure.items():
        r1 = v["flat_32"] / cf["flat_32"] if cf else float("nan")
        r2 = (v["flat_32"] / gt["flat_32"]) if gt else float("nan")
        print(f"  {k:16s}{v['immerkaer_32']:11.3f}{v['flat_32']:12.3f}"
              f"{r1:14.2f}x{r2:14.2f}x")

    # ---------- 3. fine sigma scan WITH STATISTICS ----------
    print("\n=== 3. Does noise genuinely help? (exact McNemar + paired "
          "bootstrap) ===")
    device = "cpu"
    model = load_model(args.model, device)
    tfm = build_transform()

    def scan(sel, tag):
        imgs = []
        for s in sel:
            im = cv2.imread(str(s["path"]))
            if im is not None:
                imgs.append((im, s["true"],
                             f"{s['filename']}|{s['occ']}|{s['ch']}|{s['sev']}"))
        y = np.array([t for _, t, _ in imgs])
        preds = {}
        for sg in SIGMAS:
            pl = []
            for i in range(0, len(imgs), 64):
                ch_ = imgs[i:i + 64]
                p, _ = classify_batch(model,
                                      [noisy(im, sg, k) for im, _, k in ch_],
                                      tfm, device)
                pl.extend(p.tolist())
            preds[sg] = np.array(pl)
        base = preds[0]
        rows = {}
        print(f"\n  {tag}  (n={len(imgs)})")
        print(f"  {'sigma':>6s}{'acc':>8s}{'delta':>9s}"
              f"{'95% CI':>18s}{'McNemar p':>12s}")
        rng2 = np.random.default_rng(SEED)
        ok0 = (base == y).astype(float)
        for sg in SIGMAS:
            ok = (preds[sg] == y).astype(float)
            acc = 100 * ok.mean()
            d = 100 * (ok.mean() - ok0.mean())
            if sg == 0:
                lo = hi = 0.0
                p = 1.0
            else:
                idx = rng2.integers(0, len(ok), (B, len(ok)))
                ds = 100 * (ok[idx].mean(axis=1) - ok0[idx].mean(axis=1))
                lo, hi = np.percentile(ds, [2.5, 97.5])
                _, _, p = mcnemar(preds[sg], base, y)
            rows[str(sg)] = {"acc": round(acc, 2), "delta": round(d, 2),
                             "lo": round(float(lo), 2), "hi": round(float(hi), 2),
                             "p": p}
            star = " SIGNIFICANT" if (lo > 0 or hi < 0) else ""
            print(f"  {sg:6d}{acc:8.2f}{d:+9.2f}   [{lo:+6.2f},{hi:+6.2f}]"
                  f"{p:12.2e}{star}")
        return rows

    out["scan_clean"] = scan([s for s in samples if s["ch"] == 0],
                             "CLEAN (ChallengeFree)")
    sel_gb = [s for s in samples if s["ch"] == 7 and s["sev"] in (3, 4, 5)]
    rng3 = np.random.default_rng(SEED)
    sel_gb = [sel_gb[i] for i in rng3.permutation(len(sel_gb))[:1500]]
    out["scan_gaussblur"] = scan(sel_gb, "GaussianBlur sev 3-5 (largest deficit)")
    sel_hz = [s for s in samples if s["ch"] == 12 and s["sev"] in (3, 4, 5)]
    rng4 = np.random.default_rng(SEED)
    sel_hz = [sel_hz[i] for i in rng4.permutation(len(sel_hz))[:1500]]
    out["scan_haze"] = scan(sel_hz, "Haze sev 3-5")

    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "X_noise_domain_shift.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {Path(args.outdir) / 'X_noise_domain_shift.json'}")
    print("\nREADING GUIDE\n  If the clean-image gain is NOT significant, the "
          "earlier verdict (B) was based on\n  noise and must be revised. If "
          "GTSRB's flat-noise level is far above CURE's, then\n  operators that "
          "amplify high-frequency content are systematically favoured on\n  "
          "CURE, and that confound must be disclosed for the MAIN results, not "
          "only for the\n  noise experiment.")


if __name__ == "__main__":
    main()
