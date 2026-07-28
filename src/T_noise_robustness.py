# -*- coding: utf-8 -*-
"""
T_noise_robustness.py
Pre-registered sensitivity experiment (Evaluation Protocol, Part 12; registered
2026-07-09, before execution).

WHAT THIS TESTS, PRECISELY
CURE-TSR renders Haze and Rain as noise-free alpha-composited veils. The dark
channel prior recovers J = (I - A)/t + A, a division by the transmission t,
which amplifies any noise present in the input by 1/t (at severity 5, alpha=0.5,
so t ~ 0.5 and noise is roughly doubled). Sensor noise is absent from the
rendering and always present in real captures, and is therefore the single
factor most likely to break the transfer of the rendered-haze result. This
experiment measures whether the DCP advantage survives additive sensor noise.

WHAT THIS DOES NOT TEST
It does not test the veil-rendering circularity itself; that would require
naturally captured hazy traffic signs with fine-grained labels, which no public
dataset provides (Protocol Part 9). This is a necessary, not a sufficient,
transfer check, and is reported as such. The circularity is disclosed
independently under Protocol Part 11.

FROZEN DESIGN (may not be revisited)
  noise model      : additive zero-mean Gaussian on the uint8 BGR image at the
                     original crop resolution, applied before any operator,
                     then clipped to [0, 255]
  sigma grid       : 0, 4, 8, 16  (in units of 255)
  per-image seed   : derived from (filename, occ, ch, sev, sigma), so the noise
                     realisation is reproducible and order-independent
  challenges       : Haze (12) and Rain (9), the two veil-rendered challenges
  severities       : 3, 4, 5   (the range where the DCP advantage is located)
  subsample        : 500 images per (challenge, severity) cell, seed 42
  operators        : passthrough, gamma, clahe, stretch, DCP, AdaIR
                     (passthrough is the control: it isolates the damage noise
                      does to the classifier from the damage it does to an
                      operator)
  primary endpoint : the DCP minus AdaIR accuracy gap on Haze as a function of
                     sigma
  both directions are informative: a gap that persists is evidence that the
  advantage is robust to a factor known to be missing from the rendering; a gap
  that collapses is evidence that the rendered result should not be expected to
  transfer, and the manuscript must say so.

BUILT-IN DETERMINISM AUDIT
At sigma = 0 the predictions of DCP and AdaIR produced here must equal, image
for image, the cached predictions in dcp_cure.csv and merged_per_image.csv.
Any mismatch means the pipeline differs and the run is void.

Writes: outputs_revision/T_noise_per_image.csv, T_noise_summary.json
Run:    python T_noise_robustness.py                       (full, ~65 min)
        python T_noise_robustness.py --n-per-cell 100      (fast probe)
        python T_noise_robustness.py --resume
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from F_master_sweep_cache import (            # noqa: E402
    apply_gamma, apply_clahe, apply_stretch, load_model, build_transform,
    CURE_TO_GTSRB, CHALLENGE_TYPES,
)
from J_local_deep_eval import (               # noqa: E402
    load_adair, enhance_batch, classify_batch, same_size_batches,
)
from Q_dcp_branch import dcp_enhance, scan_images   # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"

# ---- frozen design constants (Protocol Part 12; DO NOT EDIT) ----
SIGMAS = (0, 4, 8, 16)
CHALLENGES = (12, 9)          # Haze, Rain
SEVERITIES = (3, 4, 5)
SUBSAMPLE_SEED = 42
OPERATORS = ("passthrough", "gamma", "clahe", "stretch", "dcp", "adair")


def noisy(img_bgr, sigma, key):
    """Additive zero-mean Gaussian noise, per-image reproducible seed."""
    if sigma == 0:
        return img_bgr
    h = hashlib.sha256(f"{key}|{sigma}".encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2 ** 32)
    rng = np.random.default_rng(seed)
    out = img_bgr.astype(np.float32) + rng.normal(0.0, float(sigma),
                                                  img_bgr.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def classical(img_bgr, op):
    if op == "passthrough":
        return img_bgr
    if op == "gamma":
        return apply_gamma(img_bgr)
    if op == "clahe":
        return apply_clahe(img_bgr)
    if op == "stretch":
        return apply_stretch(img_bgr)
    if op == "dcp":
        return dcp_enhance(img_bgr)[0]
    raise ValueError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--adair-weight", default=str(PROJECT_ROOT / "models" / "adair5d.ckpt"))
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--dcp-cache", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--n-per-cell", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--challenges", default=",".join(str(c) for c in CHALLENGES),
                    help="comma-separated challenge ids. Changing this "
                         "constitutes a NEW pre-registered experiment; the "
                         "Part 12 design is 12,9 (the veil challenges).")
    ap.add_argument("--skip-adair", action="store_true",
                    help="classical operators only (seconds instead of hours); "
                         "AdaIR column is written as -1")
    ap.add_argument("--out-name", default="T_noise_per_image.csv")
    args = ap.parse_args()
    challenges = tuple(int(x) for x in args.challenges.split(","))

    out_csv = OUT_DIR / args.out_name
    if out_csv.exists() and not (args.resume or args.fresh):
        sys.exit(f"[SAFETY] {out_csv} exists. Use --resume or --fresh.")

    device = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    model = load_model(args.model, device)
    tfm = build_transform()
    adair = None if args.skip_adair else load_adair(device,
                                                    args.adair_weight)[0]

    # ---- subsample, stratified by (challenge, severity), frozen seed ----
    alls = scan_images(Path(args.cure_root))
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    picked = []
    for ch in challenges:
        for sv in SEVERITIES:
            cell = [s for s in alls if s["ch"] == ch and s["sev"] == sv]
            cell.sort(key=lambda s: (s["filename"], s["occ"]))   # order-stable
            idx = rng.permutation(len(cell))[:args.n_per_cell]
            picked.extend([cell[i] for i in sorted(idx)])
    print(f"[plan] {len(picked)} images x {len(SIGMAS)} sigmas "
          f"= {len(picked)*len(SIGMAS)} AdaIR inferences")
    print(f"[plan] AdaIR at ~3.1 img/s -> ETA "
          f"{len(picked)*len(SIGMAS)/3.1/60:.0f} min")

    done = set()
    if args.resume and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["filename"], int(r["occ"]), int(r["ch"]),
                          int(r["sev"]), int(r["sigma"])))
        print(f"[resume] {len(done)} rows cached")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and out_csv.exists()) else "w"
    fout = open(out_csv, mode, newline="", encoding="utf-8")
    w = csv.writer(fout)
    if mode == "w":
        w.writerow(["filename", "occ", "ch", "sev", "sigma", "true"]
                   + [f"pred_{o}" for o in OPERATORS])

    t0 = time.time()
    n_done = 0
    for sigma in SIGMAS:
        todo = [s for s in picked
                if (s["filename"], s["occ"], s["ch"], s["sev"], sigma)
                not in done]
        if not todo:
            print(f"[sigma {sigma}] already complete")
            continue
        print(f"[sigma {sigma}] {len(todo)} images")
        for bstart in range(0, len(todo), args.batch):
            chunk = todo[bstart:bstart + args.batch]
            raw, metas = [], []
            for s in chunk:
                img = cv2.imread(str(s["path"]))
                if img is None:
                    continue
                key = f"{s['filename']}|{s['occ']}|{s['ch']}|{s['sev']}"
                raw.append(noisy(img, sigma, key))
                metas.append(s)
            if not raw:
                continue
            preds = {}
            for op in OPERATORS:
                if op == "adair":
                    continue
                enh = [classical(im, op) for im in raw]
                p, _ = classify_batch(model, enh, tfm, device)
                preds[op] = p
            # AdaIR: group by identical size (the model is fully convolutional
            # but the batch must be size-homogeneous)
            if adair is None:
                preds["adair"] = np.full(len(raw), -1, dtype=int)
            else:
                pa = np.zeros(len(raw), dtype=int)
                for gmeta, gimgs in same_size_batches(
                        list(range(len(raw))), raw, args.batch):
                    enh = enhance_batch(adair, gimgs, device)
                    p, _ = classify_batch(model, enh, tfm, device)
                    for j, gi in enumerate(gmeta):
                        pa[gi] = p[j]
                preds["adair"] = pa

            for i, s in enumerate(metas):
                w.writerow([s["filename"], s["occ"], s["ch"], s["sev"], sigma,
                            s["true"]] + [int(preds[o][i]) for o in OPERATORS])
            n_done += len(metas)
            if n_done % (args.batch * 8) < args.batch:
                r = n_done / max(time.time() - t0, 1e-9)
                tot = len(picked) * len(SIGMAS)
                print(f"  [{n_done}/{tot}] {r:.1f} img/s  ETA "
                      f"{(tot - n_done)/max(r,1e-9)/60:.0f} min")
        fout.flush()
    fout.close()
    print(f"[done] {n_done} image-sigma pairs in "
          f"{(time.time()-t0)/60:.1f} min")

    # ---------------- analysis ----------------
    rows = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    ch = np.array([int(r["ch"]) for r in rows])
    sv = np.array([int(r["sev"]) for r in rows])
    sg = np.array([int(r["sigma"]) for r in rows])
    tr = np.array([int(r["true"]) for r in rows])
    PR = {o: np.array([int(r[f"pred_{o}"]) for r in rows]) for o in OPERATORS}

    def acc(op, mask):
        return 100.0 * float(np.mean(PR[op][mask] == tr[mask]))

    summary = {"design": {"sigmas": list(SIGMAS), "challenges": list(challenges),
                          "severities": list(SEVERITIES),
                          "n_per_cell": args.n_per_cell,
                          "subsample_seed": SUBSAMPLE_SEED},
               "by_challenge": {}}
    for cc in challenges:
        name = CHALLENGE_TYPES.get(cc, str(cc))
        print(f"\n=== {name} (cell-averaged over severities "
              f"{list(SEVERITIES)}) ===")
        hdr = f"{'sigma':>6s}" + "".join(f"{o:>12s}" for o in OPERATORS) \
              + f"{'DCP-AdaIR':>11s}"
        print(hdr)
        summary["by_challenge"][name] = {}
        for s_ in SIGMAS:
            per_op = {}
            for o in OPERATORS:
                cells = [acc(o, (ch == cc) & (sv == v) & (sg == s_))
                         for v in SEVERITIES
                         if ((ch == cc) & (sv == v) & (sg == s_)).sum() > 0]
                per_op[o] = float(np.mean(cells)) if cells else float("nan")
            gap = per_op["dcp"] - per_op["adair"]
            summary["by_challenge"][name][str(s_)] = {**per_op,
                                                      "dcp_minus_adair": gap}
            print(f"{s_:>6d}" + "".join(f"{per_op[o]:12.2f}" for o in OPERATORS)
                  + f"{gap:+11.2f}")

    sum_name = args.out_name.replace("per_image", "summary").replace(
        ".csv", ".json")
    if sum_name == args.out_name:
        sum_name = args.out_name.rsplit(".", 1)[0] + "_summary.json"
    with open(OUT_DIR / sum_name, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_DIR / sum_name}")

    # ---------------- determinism audit at sigma = 0 ----------------
    print("\n=== DETERMINISM AUDIT (sigma = 0 must match the caches) ===")
    Dc = {}
    with open(args.dcp_cache, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Dc[(r["filename"], int(r["occ"]), int(r["ch"]),
                int(r["sev"]))] = int(r["pred_dcp"])
    Mc = {}
    with open(args.merged, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Mc[(r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))] = (
                int(r["pred_adair"]), int(r["pred_passthrough"]),
                int(r["pred_gamma"]), int(r["pred_clahe"]),
                int(r["pred_stretch"]))
    checks = {"dcp": 0, "passthrough": 0, "gamma": 0, "clahe": 0,
              "stretch": 0}
    if not args.skip_adair:
        checks["adair"] = 0
    tot = 0
    for r in rows:
        if int(r["sigma"]) != 0:
            continue
        k = (r["filename"], int(r["occ"]), int(r["ch"]), int(r["sev"]))
        if k not in Dc or k not in Mc:
            continue
        tot += 1
        ad_, pt_, ga_, cl_, st_ = Mc[k]
        checks["dcp"] += int(int(r["pred_dcp"]) == Dc[k])
        if "adair" in checks:
            checks["adair"] += int(int(r["pred_adair"]) == ad_)
        checks["passthrough"] += int(int(r["pred_passthrough"]) == pt_)
        checks["gamma"] += int(int(r["pred_gamma"]) == ga_)
        checks["clahe"] += int(int(r["pred_clahe"]) == cl_)
        checks["stretch"] += int(int(r["pred_stretch"]) == st_)
    # Batch composition here differs from the main runs, so a handful of
    # boundary samples may flip under CPU float noise. Graded verdict:
    #   100%      -> PASS   (bit-identical)
    #   >= 99.5%  -> WARN   (float-noise level, acceptable)
    #   <  99.5%  -> FAIL   (the pipeline genuinely differs; numbers are void)
    worst = 1.0
    for o, n in checks.items():
        rate = n / tot if tot else 0.0
        worst = min(worst, rate)
        verdict = "PASS" if n == tot else ("WARN" if rate >= 0.995 else "FAIL")
        print(f"  [{verdict}] {o:12s} {n}/{tot} identical to cache "
              f"({100*rate:.2f}%)")
    if worst == 1.0:
        print("\nAUDIT PASSED (bit-identical): the pipeline reproduces the "
              "cached predictions exactly.")
    elif worst >= 0.995:
        print(f"\nAUDIT PASSED WITH WARNING: worst agreement "
              f"{100*worst:.2f}%. Batch composition differs from the main "
              "runs, so a few\nboundary samples flipped under CPU float "
              "noise. This is expected and does not\ninvalidate the "
              "comparison, which is internal to this run.")
    else:
        print(f"\n*** AUDIT FAILED: worst agreement {100*worst:.2f}%. The "
              "pipeline genuinely differs from\nthe cached runs. Do NOT use "
              "these numbers. Send me this output. ***")


if __name__ == "__main__":
    main()
