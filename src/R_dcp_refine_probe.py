# -*- coding: utf-8 -*-
"""
R_dcp_refine_probe.py
Measures what the transmission-refinement step is worth on CURE-TSR crops, so
that the operator the paper describes is described from evidence.

WHY THIS EXISTS
  Q_dcp_branch gates its refinement on cv2.ximgproc being importable:

      if HAS_GUIDED:
          t = cv2.ximgproc.guidedFilter(guide, t, GF_RADIUS, GF_EPS)
      t = np.maximum(t, DCP_T0)

  ximgproc is absent here, so the block is skipped ENTIRELY. There is no
  substitute filter: the block-wise transmission from the 15x15 erosion is used
  as it stands, floored at t0. dcp_cure.csv was produced on that path.

  This is not a footnote, because DCP is not a baseline. It sits INSIDE the
  proposed selector: V-B is the routing rule with its low-contrast branch served
  by DCP. The operator therefore has to be stated exactly, and the cost of the
  missing step has to be measured rather than assumed.

WHAT THE ARITHMETIC ALREADY SAYS, BEFORE ANY IMAGE IS READ
  Degraded crops average 40.4 x 40.1 px, from 11x11 to 285x211. Against that:
      DCP_PATCH 15   -> the erosion kernel spans 37 per cent of a median crop and
                        exceeds the whole image at 11x11
      A: 0.1% of px  -> two pixels on a median crop, one at 11x11
      GF_RADIUS 60   -> a 121x121 window, three times the median crop
  So the refinement, had it run, would have been a near-global smoothing rather
  than the edge-aware halo removal it performs on the large hazy photographs it
  was built for. That is a claim about the regime, and it is testable.

WHAT THIS PROBE DOES
  Runs both transmission paths over a stratified sample. It installs nothing: the
  guided filter is He and Sun's (TPAMI 2013), assembled from cv2.boxFilter, which
  base OpenCV has.

  Per image it produces THREE predictions:
      pred_rule    the base routing rule, F.apply_branch on the routed branch
      pred_dcp_U   DCP with the transmission unrefined   (the current cache)
      pred_dcp_R   DCP with the guided filter restored
  and from them, directly and not by any bound:
      V-B (U) = where(branch == clahe, pred_dcp_U, pred_rule)
      V-B (R) = where(branch == clahe, pred_dcp_R, pred_rule)

THREE GUARDS, BECAUSE ONE IS NOT ENOUGH
  G1  the unrefined image this script computes must be BIT-IDENTICAL to
      Q.dcp_enhance's, checked pixel for pixel on every sampled image. This is
      what proves the shared pre-fork math was not mistyped, and therefore that
      the REFINED path is built on the right foundation. A cache check alone
      would not catch a bug here.
  G2  pred_dcp_U must reproduce dcp_cure.csv on the sampled rows. This proves the
      sampling, the join keys, the resize and the classifier are right.
  G3  pred_rule and the routed branch must reproduce merged_per_image.csv on the
      sampled rows. This proves the routing and the rule's operators are right.
  A guard that fails aborts the run. Nothing is reported from a harness that
  cannot reproduce what is already on disk.

ONE CAVEAT, STATED RATHER THAN BURIED
  At radius 60 on a 40 px crop every filter window extends far past the image, so
  the refined result is dominated by the border-extension policy. This uses
  OpenCV's default for boxFilter (reflect101). cv2.ximgproc may extend borders
  differently, so path R is a faithful implementation of the published filter but
  is not guaranteed bit-identical to ximgproc's. It measures MAGNITUDE. If the
  magnitude proves to matter, the refinement belongs in Q explicitly, with a
  stated border policy, instead of resting on an optional dependency.

THE DECISION IS A RULE, NOT A PREFERENCE
  If V-B's degraded-average moves by less than 0.20 points AND the two paths
  agree on at least 99 per cent of clahe-branch predictions, the existing cache
  stands and Section III states the operator as it is. Otherwise the refinement
  is material, and Q must carry the guided filter explicitly and be rerun.

  V-B's CLEAN accuracy cannot move at all, and no measurement is needed to know
  why: on clean input the rule stays in passthrough on every image, so DCP is
  never called. The probe checks that this holds on the sample rather than
  asserting it.

READS   datasets/CURE-TSR                     (read-only)
        outputs_revision/dcp_cure.csv         (read-only, guard G2)
        outputs_revision/merged_per_image.csv (read-only, guard G3)
WRITES  outputs_revision/R_dcp_refine_probe.results.json
        outputs_revision/R_dcp_refine_probe.execution_log.txt
RUN     python R_dcp_refine_probe.py
        python R_dcp_refine_probe.py --per-cell 80 --clean-n 1600
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import F_master_sweep_cache as F           # noqa: E402
import Q_dcp_branch as Q                   # noqa: E402

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUT_DIR = PROJECT_ROOT / "outputs_revision"

A_FRACTION = 0.001      # Q hardcodes this; it is not a named constant there.
                        # Guard G1 is what proves the line was copied correctly.

LOG = []


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line)
    LOG.append(line)


def die(msg):
    log(msg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "R_dcp_refine_probe.execution_log.txt").write_text(
        "\n".join(LOG), encoding="utf-8")
    sys.exit(1)


def guided_filter_gray(guide, src, radius, eps):
    """He and Sun, Guided Image Filtering, TPAMI 2013, single-channel guide.
    Assembled from cv2.boxFilter so that no contrib module is required. This is
    the filter cv2.ximgproc.guidedFilter computes when the guide has one
    channel."""
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


def dcp_prefork(img_bgr):
    """Q's math up to, but not including, the refinement fork. Every constant is
    read from Q. Guard G1 proves this reproduces Q exactly."""
    I = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (Q.DCP_PATCH, Q.DCP_PATCH))
    dark = cv2.erode(I.min(axis=2), kernel)
    n = dark.size
    kpix = max(1, int(round(A_FRACTION * n)))
    idx = np.argpartition(dark.reshape(-1), n - kpix)[n - kpix:]
    ys, xs = np.unravel_index(idx, dark.shape)
    A = np.maximum(I[ys, xs].mean(axis=0), 1e-3)
    dark_norm = cv2.erode((I / A[None, None, :]).min(axis=2), kernel)
    t = 1.0 - Q.DCP_OMEGA * dark_norm
    return I, A, t, kpix


def recover(I, A, t):
    """Q's step 5, verbatim."""
    tt = np.maximum(t, Q.DCP_T0)
    J = (I - A[None, None, :]) / tt[..., None] + A[None, None, :]
    return (np.clip(J, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_tensor(enh_bgr, tfm):
    small = cv2.resize(enh_bgr, (F.INPUT_SIZE, F.INPUT_SIZE))
    return tfm(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root",
                    default=str(PROJECT_ROOT / "datasets" / "CURE-TSR"))
    ap.add_argument("--model",
                    default=str(PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"))
    ap.add_argument("--dcp-csv", default=str(OUT_DIR / "dcp_cure.csv"))
    ap.add_argument("--merged-csv", default=str(OUT_DIR / "merged_per_image.csv"))
    ap.add_argument("--per-cell", type=int, default=40,
                    help="degraded images per (challenge, severity) cell, 60 cells")
    ap.add_argument("--clean-n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)

    if Q.HAS_GUIDED:
        die("ABORT: Q.HAS_GUIDED is True here, so dcp_cure.csv was already made "
            "WITH refinement and there is nothing to probe. This script exists "
            "only for the unrefined-cache case.")
    log(f"Q.HAS_GUIDED = {Q.HAS_GUIDED}. The cache is on the unrefined path.")
    log(f"constants read from Q: patch {Q.DCP_PATCH}, omega {Q.DCP_OMEGA}, "
        f"t0 {Q.DCP_T0}, gf_radius {Q.GF_RADIUS}, gf_eps {Q.GF_EPS}")
    log(f"thresholds read from F: {F.THRESHOLDS}")
    log(f"torch {torch.__version__}, opencv {cv2.__version__}, "
        f"threads {args.threads}")

    samples = Q.scan_images(Path(args.cure_root))
    log(f"scanned: {len(samples)} usable images")

    cells = defaultdict(list)
    for s in samples:
        cells[(s["ch"], s["sev"])].append(s)
    rng = random.Random(args.seed)
    pick, n_deg_cells = [], 0
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda s: (s["filename"], s["occ"]))
        rng.shuffle(pool)
        if key[0] == 0:
            pick.extend(pool[:args.clean_n])
        else:
            pick.extend(pool[:args.per_cell])
            n_deg_cells += 1
    n_clean = sum(1 for s in pick if s["ch"] == 0)
    log(f"stratified sample: {len(pick)} images "
        f"({len(pick)-n_clean} degraded over {n_deg_cells} cells at "
        f"{args.per_cell} each, {n_clean} clean), seed {args.seed}")

    cache_dcp = {}
    with open(args.dcp_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cache_dcp[(r["filename"], int(r["occ"]), int(r["ch"]),
                       int(r["sev"]))] = int(r["pred_dcp"])
    cache_rule = {}
    with open(args.merged_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cache_rule[(r["filename"], int(r["occ"]), int(r["ch"]),
                        int(r["sev"]))] = (int(r["pred_va_rule"]),
                                           r["rule_branch"])
    log(f"dcp_cure.csv: {len(cache_dcp)} rows | "
        f"merged_per_image.csv: {len(cache_rule)} rows")

    model = F.load_model(args.model, "cpu")
    tfm = F.build_transform()

    rows, buf = [], []
    g1_checked = g1_bad = 0
    t_pre = t_gf = 0.0

    def flush():
        if not buf:
            return
        with torch.no_grad():
            p_rule = model(torch.stack([m["x_rule"] for m in buf], 0)).argmax(1)
            p_u = model(torch.stack([m["x_u"] for m in buf], 0)).argmax(1)
            p_r = model(torch.stack([m["x_r"] for m in buf], 0)).argmax(1)
        for i, m in enumerate(buf):
            rows.append({**{k: m[k] for k in
                            ("filename", "occ", "ch", "sev", "true", "branch",
                             "cv_u", "cv_r", "kpix", "h", "w")},
                         "pred_rule": int(p_rule[i]), "pred_u": int(p_u[i]),
                         "pred_r": int(p_r[i])})
        buf.clear()

    t_start = time.time()
    for s in pick:
        img = cv2.imread(str(s["path"]))
        if img is None:
            continue

        b, c, e = F.compute_stats(img)
        branch = F.route_decision(b, c, e, F.THRESHOLDS)
        rule_img = F.apply_branch(img, branch)

        t0 = time.perf_counter()
        I, A, t_raw, kpix = dcp_prefork(img)
        enh_u = recover(I, A, t_raw)
        t1 = time.perf_counter()
        guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_ref = guided_filter_gray(guide, t_raw.astype(np.float32),
                                   Q.GF_RADIUS, Q.GF_EPS)
        enh_r = recover(I, A, t_ref)
        t2 = time.perf_counter()
        t_pre += t1 - t0
        t_gf += t2 - t1

        # ---- G1: bit for bit against Q's own function, on every image ----
        q_enh, _, _ = Q.dcp_enhance(img)
        g1_checked += 1
        if not np.array_equal(q_enh, enh_u):
            g1_bad += 1

        buf.append({
            "filename": s["filename"], "occ": s["occ"], "ch": s["ch"],
            "sev": s["sev"], "true": s["true"], "branch": branch,
            "cv_u": float(t_raw.std()) / max(float(t_raw.mean()), 1e-9),
            "cv_r": float(t_ref.std()) / max(float(t_ref.mean()), 1e-9),
            "kpix": kpix, "h": img.shape[0], "w": img.shape[1],
            "x_rule": to_tensor(rule_img, tfm), "x_u": to_tensor(enh_u, tfm),
            "x_r": to_tensor(enh_r, tfm)})
        if len(buf) >= args.batch:
            flush()
    flush()
    log(f"processed {len(rows)} images in {(time.time()-t_start)/60:.1f} min")

    # ===================== the three guards =====================
    log("")
    log("=== GUARDS ===")
    log(f"  G1  unrefined output bit-identical to Q.dcp_enhance: "
        f"{g1_checked - g1_bad}/{g1_checked}")
    if g1_bad:
        die("  G1 FAILED. The shared pre-fork math in this script does not match "
            "Q, so the refined path is built on the wrong foundation and nothing "
            "below would mean anything. Nothing is reported.")

    hit_d = [r for r in rows
             if (r["filename"], r["occ"], r["ch"], r["sev"]) in cache_dcp]
    ok_d = [r for r in hit_d
            if cache_dcp[(r["filename"], r["occ"], r["ch"],
                          r["sev"])] == r["pred_u"]]
    log(f"  G2  pred_dcp_U reproduces dcp_cure.csv: {len(ok_d)}/{len(hit_d)}")
    if not hit_d or len(ok_d) != len(hit_d):
        die("  G2 FAILED. The sampling, the join, the resize or the classifier is "
            "wrong. Nothing is reported.")

    hit_r = [r for r in rows
             if (r["filename"], r["occ"], r["ch"], r["sev"]) in cache_rule]
    ok_r = [r for r in hit_r
            if cache_rule[(r["filename"], r["occ"], r["ch"], r["sev"])] ==
            (r["pred_rule"], r["branch"])]
    log(f"  G3  pred_rule and branch reproduce merged_per_image.csv: "
        f"{len(ok_r)}/{len(hit_r)}")
    if not hit_r or len(ok_r) != len(hit_r):
        die("  G3 FAILED. The routing or the rule's operators do not match the "
            "merged cache. Nothing is reported.")
    log("  all three guards pass: the harness reproduces what is on disk, so the "
        "comparison below is about the refinement and nothing else.")

    # ===================== the comparison =====================
    ch = np.array([r["ch"] for r in rows])
    sev = np.array([r["sev"] for r in rows])
    tru = np.array([r["true"] for r in rows])
    brn = np.array([r["branch"] for r in rows])
    p_rule = np.array([r["pred_rule"] for r in rows])
    p_u = np.array([r["pred_u"] for r in rows])
    p_r = np.array([r["pred_r"] for r in rows])
    deg, cfm = ch > 0, ch == 0
    clahe = brn == "clahe"

    vb_u = np.where(clahe, p_u, p_rule)
    vb_r = np.where(clahe, p_r, p_rule)

    CELLS = [(c, s) for c in range(1, 13) for s in range(1, 6)]
    masks = [m for m in ((ch == c) & (sev == s) for c, s in CELLS) if m.sum()]

    def degavg(p):
        return float(np.mean([100.0 * np.mean(p[m] == tru[m]) for m in masks]))

    def clean(p):
        return (100.0 * float(np.mean(p[cfm] == tru[cfm]))
                if cfm.sum() else float("nan"))

    def agree(p, q, m):
        return 100.0 * float(np.mean(p[m] == q[m])) if m.sum() else float("nan")

    log("")
    log("=== 1. where the two paths disagree ===")
    for nm, m in [("all", np.ones_like(deg)), ("degraded", deg),
                  ("clean", cfm),
                  ("CLAHE branch, the only place V-B calls DCP", clahe)]:
        log(f"    agreement, {nm:44s} {agree(p_u, p_r, m):6.2f}%   "
            f"n = {int(m.sum())}")

    log("")
    log("=== 2. always-DCP, the baseline in the figures ===")
    log(f"    degraded-average   unrefined {degavg(p_u):6.2f}   "
        f"refined {degavg(p_r):6.2f}   delta {degavg(p_r)-degavg(p_u):+.2f}")
    log(f"    clean              unrefined {clean(p_u):6.2f}   "
        f"refined {clean(p_r):6.2f}   delta {clean(p_r)-clean(p_u):+.2f}")

    log("")
    log("=== 3. V-B, the proposed selector, computed directly ===")
    n_cl_deg = int((clahe & deg).sum())
    log(f"    V-B sends {n_cl_deg} of {int(deg.sum())} degraded images "
        f"({100*n_cl_deg/max(int(deg.sum()),1):.1f}%) to DCP")
    log(f"    on clean it sends {int((clahe & cfm).sum())} of {int(cfm.sum())} "
        f"to DCP; this must be 0, since the rule never leaves passthrough there")
    log(f"    degraded-average   unrefined {degavg(vb_u):6.2f}   "
        f"refined {degavg(vb_r):6.2f}   delta {degavg(vb_r)-degavg(vb_u):+.2f}")
    log(f"    clean              unrefined {clean(vb_u):6.2f}   "
        f"refined {clean(vb_r):6.2f}   delta {clean(vb_r)-clean(vb_u):+.2f}")

    log("")
    log("=== 4. how much spatial structure the transmission map keeps ===")
    cvu = np.array([r["cv_u"] for r in rows])
    cvr = np.array([r["cv_r"] for r in rows])
    log(f"    coefficient of variation of t   unrefined median "
        f"{np.median(cvu):.4f}   refined median {np.median(cvr):.4f}")
    log("    (near zero means t is effectively constant, so the operator acts as "
        "a global affine stretch rather than a spatially varying dehaze)")

    log("")
    log("=== 5. the regime the frozen constants land in ===")
    kp = np.array([r["kpix"] for r in rows])
    hs = np.array([r["h"] for r in rows])
    ws = np.array([r["w"] for r in rows])
    ratio = Q.DCP_PATCH / np.minimum(hs, ws)
    gfw = 2 * Q.GF_RADIUS + 1
    log(f"    A is averaged over {int(np.median(kp))} pixels at the median crop "
        f"(range {int(kp.min())} to {int(kp.max())})")
    log(f"    the {Q.DCP_PATCH}x{Q.DCP_PATCH} kernel spans "
        f"{100*float(np.median(ratio)):.0f}% of the shorter side at the median "
        f"crop, and covers the whole crop on "
        f"{100*float(np.mean(ratio >= 1.0)):.1f}% of them")
    log(f"    the {gfw}x{gfw} guided-filter window is larger than the whole crop "
        f"on {100*float(np.mean(np.maximum(hs, ws) < gfw)):.1f}% of them")

    log("")
    log("=== 6. what the refinement would cost ===")
    n = max(len(rows), 1)
    log(f"    DCP without refinement {1000*t_pre/n:.3f} ms/img; the guided filter "
        f"adds {1000*t_gf/n:.3f} ms/img (per-image means)")

    # ===================== the decision =====================
    d_vb = abs(degavg(vb_r) - degavg(vb_u))
    ag_cl = agree(p_u, p_r, clahe)
    immaterial = (d_vb < 0.20) and (ag_cl >= 99.0)
    log("")
    log("=== DECISION ===")
    log(f"    V-B's degraded-average moves {d_vb:.3f} points (threshold 0.20)")
    log(f"    the paths agree on {ag_cl:.2f}% of clahe-branch predictions "
        f"(threshold 99.00)")
    if immaterial:
        log("    -> IMMATERIAL. dcp_cure.csv stands, no rerun. Section III states "
            "the operator as it is: a dark-channel-prior contrast operator with "
            "the published constants, using the block-wise transmission without "
            "edge-aware refinement.")
    else:
        log("    -> MATERIAL. The refinement changes the proposed method. "
            "Q_dcp_branch must carry the guided filter explicitly, built from "
            "boxFilter so it never rests on an optional module, and be rerun over "
            "all 82,472 images; K_merge and every downstream figure follow.")

    res = dict(
        config=dict(
            script="R_dcp_refine_probe.py",
            timestamp=datetime.now().isoformat(timespec="seconds"),
            per_cell=args.per_cell, clean_n=args.clean_n, seed=args.seed,
            n=len(rows), has_guided=Q.HAS_GUIDED,
            constants=dict(patch=Q.DCP_PATCH, omega=Q.DCP_OMEGA, t0=Q.DCP_T0,
                           gf_radius=Q.GF_RADIUS, gf_eps=Q.GF_EPS,
                           a_fraction=A_FRACTION),
            thresholds=F.THRESHOLDS,
            guided_filter="He and Sun TPAMI 2013 via cv2.boxFilter, border "
                          "reflect101; faithful to the published filter but not "
                          "guaranteed bit-identical to cv2.ximgproc",
            torch=torch.__version__, opencv=cv2.__version__),
        guards=dict(
            g1_bit_identical_to_Q=dict(passed=g1_checked - g1_bad,
                                       checked=g1_checked),
            g2_reproduces_dcp_cure=dict(passed=len(ok_d), checked=len(hit_d)),
            g3_reproduces_merged=dict(passed=len(ok_r), checked=len(hit_r))),
        agreement=dict(all=agree(p_u, p_r, np.ones_like(deg)),
                       degraded=agree(p_u, p_r, deg),
                       clean=agree(p_u, p_r, cfm),
                       clahe_branch=ag_cl, n_clahe=int(clahe.sum())),
        always_dcp=dict(deg_avg_unrefined=degavg(p_u),
                        deg_avg_refined=degavg(p_r),
                        clean_unrefined=clean(p_u), clean_refined=clean(p_r)),
        vb=dict(deg_avg_unrefined=degavg(vb_u), deg_avg_refined=degavg(vb_r),
                clean_unrefined=clean(vb_u), clean_refined=clean(vb_r),
                clahe_share_degraded=float(np.mean(clahe[deg])),
                clahe_on_clean=int((clahe & cfm).sum())),
        transmission=dict(cv_unrefined_median=float(np.median(cvu)),
                          cv_refined_median=float(np.median(cvr))),
        regime=dict(a_pixels_median=int(np.median(kp)),
                    a_pixels_min=int(kp.min()), a_pixels_max=int(kp.max()),
                    kernel_over_short_side_median=float(np.median(ratio)),
                    share_kernel_covers_crop=float(np.mean(ratio >= 1.0)),
                    share_gf_window_exceeds_crop=float(
                        np.mean(np.maximum(hs, ws) < gfw))),
        timing_ms=dict(dcp_unrefined=1000 * t_pre / n,
                       guided_filter_added=1000 * t_gf / n),
        decision=dict(immaterial=bool(immaterial),
                      vb_deg_avg_shift=d_vb, clahe_agreement=ag_cl))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "R_dcp_refine_probe.results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    (OUT_DIR / "R_dcp_refine_probe.execution_log.txt").write_text(
        "\n".join(LOG), encoding="utf-8")
    log("")
    log("wrote outputs_revision/R_dcp_refine_probe.results.json")
    log("wrote outputs_revision/R_dcp_refine_probe.execution_log.txt")


if __name__ == "__main__":
    main()
