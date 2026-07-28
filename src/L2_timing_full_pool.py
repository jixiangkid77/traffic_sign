r"""
L2_timing_full_pool.py   一次量完:把 DCP 与 V-B 补进 L 的同一把尺子

WHY
  L_timing_enhance_only.py 是对的,但它【过期了两重】:

    1. 它跑于 2026-07-04,那时算子池里【没有 DCP】,也【没有 V-B 选择器】。
       论文现在的前端是 V-B,它的代价从来没被量过。
    2. 它的取样池是 J.EVAL_CHALLENGES = [4, 8, 9, 11, 12],即 25 个退化 cell
       + ChallengeFree(它自己的注释就是这么写的)。论文现在评的是【12 挑战,
       60 个退化 cell】。分支占比也因此对不上:L 的样本是 passthrough 150/210
       = 71.4%,而真实退化集是 75.46%。

  规划里那几个效率数字(DCP 11.1 ms/img、V-B 前端 1.52 ms、always-AdaIR 333 ms、
  219 倍)全部作废:11.1 是从批处理吞吐日志 "[dcp] ... 89.9 img/s" 反推的,不是
  逐图延迟;1.52 与 333 两头都推不出。效率是 T-ITS 全文的框架,不能建立在对不上
  账的数字上。

WHAT THIS SCRIPT IS
  它【不是重写】,是把 L 扩一遍:同一个 harness、同一个 pstats、同一个
  stratified_pick、同一批函数(F_master_sweep_cache,即真正跑出缓存预测的那份
  代码,不是 enhance.py)、同一个 J.enhance_batch、同一个 classify 路径。
  新增的只有 scope。

  取样池改为【12 挑战 × 5 严重度 = 60 个退化 cell】,并单独取一份 ChallengeFree,
  因为 V-B 在干净图上的代价是一条独立的部署论证,必须单独量。

SCOPES
  stats          F.compute_stats                                   (特征)
  route          F.compute_stats + F.route_decision                (只到决策)
  op_gamma       F.apply_branch(img, "gamma")
  op_clahe       F.apply_branch(img, "clahe")
  op_stretch     F.apply_branch(img, "stretch")
  op_dcp         Q_dcp_branch.dcp_enhance(img)[0]                  (返回三元组)
  va_rule        stats + route + apply_branch        (复现 L 的 va_rule scope)
  vb_selector    stats + route + apply_branch,但 clahe 分支改由 DCP 服务
                 【直接实测,不用分支占比去合成】
  adair_enhance  J.enhance_batch(net, [img], "cpu")
  cidnet_enhance 同上
  classify32     cv2.resize(32) + BGR2RGB + J.build_transform() + CompactCNN
                 (各方法共享的下游成本,单列,不计入前面任何 scope)

  va_rule 与 vb_selector 在【退化样本】与【干净样本】上各测一次。

估计量与 L 完全一致:逐图 perf_counter,单遍,前 warmup 张丢弃,报 mean/median/p95。

USAGE
  python L2_timing_full_pool.py
  python L2_timing_full_pool.py --models va-only        (跳过两个深度模型)
  python L2_timing_full_pool.py --n 200 --n-clean 100

OUTPUT (outputs_revision\)
  L2_timing_full_pool.results.json
  L2_timing_full_pool.execution_log.txt
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import F_master_sweep_cache as F          # noqa: E402
import J_local_deep_eval as J             # noqa: E402
from revision_utils import load_gtsrb_compactcnn   # noqa: E402

# DCP 加入算子池晚于 L,所以 L 里没有它。名字取自 Q_dcp_branch.py:
# dcp_enhance(img_bgr) -> (enhanced_bgr, t_clip_frac, out_std_norm)。
try:
    from Q_dcp_branch import dcp_enhance   # noqa: E402
except ImportError:
    import Q_dcp_branch as _q
    _pub = [n for n in dir(_q) if not n.startswith("_") and callable(getattr(_q, n))]
    sys.exit("[FATAL] Q_dcp_branch 里没有 dcp_enhance。\n"
             f"        它有的是: {', '.join(_pub)}\n"
             "        改本文件顶部那一行 import 即可,其余部分不依赖这个名字。")

OUT_DIR = PROJECT_ROOT / "outputs_revision"
LOG_PATH = OUT_DIR / "L2_timing_full_pool.execution_log.txt"

# L 用的是 J.EVAL_CHALLENGES = [4, 8, 9, 11, 12](5 挑战时代的遗留)。
# 论文评的是 12 挑战,取样池必须与证据基座一致。显式覆盖,并在日志里喊出来。
PAPER_CHALLENGES = list(range(1, 13))


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stratified_pick(samples, n, seed):
    """Round-robin across (ch, sev) groups. Verbatim from L_timing."""
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[(s["ch"], s["sev"])].append(s)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    picked, i = [], 0
    while len(picked) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            picked.append(groups[k].pop())
        i += 1
    return picked


def pstats(ts):
    """Verbatim from L_timing."""
    a = np.asarray(ts, dtype=np.float64) * 1000.0
    return {"n_timed": int(a.size),
            "mean_ms": round(float(a.mean()), 3),
            "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


def time_scope(fn, imgs, warmup):
    """L_timing 的逐图循环,一遍,前 warmup 张丢弃。"""
    ts = []
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        fn(img)
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return pstats(ts)


def time_rule(imgs, warmup, dcp_for_clahe):
    """va_rule(dcp_for_clahe=False)与 V-B(True)。分支直方图一并返回。"""
    ts, branches = [], Counter()
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        b, c, e = F.compute_stats(img)
        br = F.route_decision(b, c, e, F.THRESHOLDS)
        if dcp_for_clahe and br == "clahe":
            _ = dcp_enhance(img)[0]
        else:
            _ = F.apply_branch(img, br)
        dt = time.perf_counter() - t0
        branches[br] += 1
        if i >= warmup:
            ts.append(dt)
    out = pstats(ts)
    out["branch_histogram"] = dict(branches)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="both",
                    choices=["both", "adair", "cidnet", "va-only"])
    ap.add_argument("--n", type=int, default=200,
                    help="timed DEGRADED images per scope (after warmup)")
    ap.add_argument("--n-clean", type=int, default=100,
                    help="timed CLEAN images for the rule and V-B scopes")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cure-root", default=str(J.CURE_TSR_DIR_DEFAULT))
    ap.add_argument("--adair-weight",
                    default=str(PROJECT_ROOT / "models" / "adair5d.ckpt"))
    ap.add_argument("--cidnet-weight", default="")
    ap.add_argument("--merged", default=str(OUT_DIR / "merged_per_image.csv"),
                    help="only used to cross-check the branch mix; optional")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    log(f"threads pinned: torch=1 cv2=1; torch {torch.__version__}, "
        f"opencv {cv2.__version__}")
    import Q_dcp_branch as _Q
    guided = bool(getattr(_Q, "HAS_GUIDED", False))
    log("DCP refinement: guided filter " +
        ("available and used" if guided else "NOT available, frozen fallback") +
        ". This changes the cost materially, so it is recorded in the output.")

    old = list(getattr(J, "EVAL_CHALLENGES", []))
    J.EVAL_CHALLENGES = PAPER_CHALLENGES
    log(f"J.EVAL_CHALLENGES overridden: {old} -> {PAPER_CHALLENGES}. "
        f"L_timing sampled from the 5-challenge era; the paper evaluates 12, so "
        f"the timing pool must match the evidence base.")

    samples, n_files = J.collect_samples(Path(args.cure_root))
    if not samples:
        sys.exit(f"[FATAL] no CURE images under {args.cure_root}")
    deg = [s for s in samples if s["ch"] != 0]
    cle = [s for s in samples if s["ch"] == 0]
    cells = sorted({(s["ch"], s["sev"]) for s in deg})
    log(f"scanned {n_files} files; {len(deg)} degraded over {len(cells)} cells, "
        f"{len(cle)} ChallengeFree")
    if len(cells) != 60:
        log(f"[warn] expected 60 degraded cells (12 x 5), found {len(cells)}. "
            f"Check that J.collect_samples honours the override.")

    def load(picked, tag):
        imgs = []
        for s in picked:
            im = cv2.imread(str(s["path"]))
            if im is None:
                log(f"[warn] unreadable, skipped: {s['path']}")
                continue
            imgs.append(im)
        hs = [im.shape[0] for im in imgs]
        ws = [im.shape[1] for im in imgs]
        log(f"{tag}: {len(imgs)} images, H mean {np.mean(hs):.1f} "
            f"[{min(hs)}, {max(hs)}], W mean {np.mean(ws):.1f} "
            f"[{min(ws)}, {max(ws)}]")
        return imgs, {"n_loaded": len(imgs),
                      "h_mean": round(float(np.mean(hs)), 1),
                      "h_min": int(min(hs)), "h_max": int(max(hs)),
                      "w_mean": round(float(np.mean(ws)), 1),
                      "w_min": int(min(ws)), "w_max": int(max(ws))}

    imgs_d, size_d = load(stratified_pick(deg, args.n + args.warmup, args.seed),
                          "degraded sample")
    imgs_c, size_c = load(
        stratified_pick(cle, args.n_clean + args.warmup, args.seed),
        "clean sample")

    results = {}
    W = args.warmup

    log("scope: stats (F.compute_stats)")
    results["stats"] = time_scope(F.compute_stats, imgs_d, W)

    def route_only(img):
        b, c, e = F.compute_stats(img)
        return F.route_decision(b, c, e, F.THRESHOLDS)
    log("scope: route (stats + route_decision, no operator)")
    results["route"] = time_scope(route_only, imgs_d, W)

    for br in ("gamma", "clahe", "stretch"):
        log(f"scope: op_{br} (F.apply_branch)")
        results[f"op_{br}"] = time_scope(
            lambda im, b=br: F.apply_branch(im, b), imgs_d, W)

    log("scope: op_dcp (Q_dcp_branch.dcp_enhance)")
    results["op_dcp"] = time_scope(lambda im: dcp_enhance(im)[0], imgs_d, W)

    log("scope: va_rule on DEGRADED (stats + route + apply_branch)")
    results["va_rule_degraded"] = time_rule(imgs_d, W, dcp_for_clahe=False)
    log("scope: va_rule on CLEAN")
    results["va_rule_clean"] = time_rule(imgs_c, W, dcp_for_clahe=False)

    log("scope: vb_selector on DEGRADED (clahe branch served by DCP), MEASURED")
    results["vb_degraded"] = time_rule(imgs_d, W, dcp_for_clahe=True)
    log("scope: vb_selector on CLEAN")
    results["vb_clean"] = time_rule(imgs_c, W, dcp_for_clahe=True)

    log("scope: classify32 (shared downstream, excluded from every other scope)")
    clf = load_gtsrb_compactcnn("cpu")
    tfm = J.build_transform()

    def classify(img):
        r = cv2.resize(img, (J.INPUT_SIZE, J.INPUT_SIZE))
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        x = tfm(rgb).unsqueeze(0)
        with torch.no_grad():
            clf(x)
    results["classify32"] = time_scope(classify, imgs_d, W)

    if args.models in ("both", "adair"):
        net, _ = J.load_adair("cpu", args.adair_weight)
        log("scope: adair_enhance (J.enhance_batch, single image)")
        results["adair_enhance"] = time_scope(
            lambda im: J.enhance_batch(net, [im], "cpu"), imgs_d, W)
        del net
    if args.models in ("both", "cidnet"):
        net, _ = J.load_cidnet("cpu", args.cidnet_weight)
        log("scope: cidnet_enhance (J.enhance_batch, single image)")
        results["cidnet_enhance"] = time_scope(
            lambda im: J.enhance_batch(net, [im], "cpu"), imgs_d, W)
        del net

    print("\n=== per-image latency, batch=1, 1 CPU thread (ms) ===")
    print(f"{'scope':20s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'n':>5s}")
    for k, r in results.items():
        print(f"{k:20s} {r['mean_ms']:8.3f} {r['median_ms']:8.3f} "
              f"{r['p95_ms']:8.3f} {r['n_timed']:5d}")

    print("\n=== branch mix in the timing sample ===")
    for k in ("va_rule_degraded", "va_rule_clean"):
        h = results[k]["branch_histogram"]
        tot = sum(h.values())
        print(f"  {k:18s} " + "  ".join(f"{a} {100*b/tot:.1f}%"
                                        for a, b in sorted(h.items())))
    print("  (cross-check against merged_per_image.csv: degraded is passthrough "
          "75.46, clahe 11.92, gamma 5.63, stretch 6.99 per cent; clean is "
          "passthrough 100.00)")

    if "adair_enhance" in results:
        vb = results["vb_degraded"]["mean_ms"]
        ad = results["adair_enhance"]["mean_ms"]
        print(f"\n  selector V-B {vb:.2f} ms against always-AdaIR {ad:.2f} ms on "
              f"degraded input: {ad/vb:.0f} times cheaper.")
        print(f"  On clean input V-B costs {results['vb_clean']['mean_ms']:.2f} ms, "
              f"because the rule never leaves passthrough there.")

    cfg = {
        "script": "L2_timing_full_pool.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": args.models, "n_degraded": args.n, "n_clean": args.n_clean,
        "warmup": args.warmup, "seed": args.seed,
        "cure_root": str(args.cure_root),
        "eval_challenges": PAPER_CHALLENGES,
        "eval_challenges_before_override": old,
        "estimator": "per-image perf_counter, single pass, first `warmup` "
                     "discarded; mean / median / p95 (identical to "
                     "L_timing_enhance_only)",
        "functions": "F_master_sweep_cache (the code that produced the cached "
                     "predictions), J.enhance_batch, J.build_transform",
        "adair_weight": args.adair_weight
        if args.models in ("both", "adair") else None,
        "cidnet_weight": (args.cidnet_weight or "HF cache")
        if args.models in ("both", "cidnet") else None,
        "threads": {"torch": 1, "cv2": 1},
        "dcp_guided_filter": guided,
        "supersedes": [
            "L_timing_enhance_only (2026-07-04): sampled from the 5-challenge "
            "era and has no DCP row",
            "DCP 11.1 ms/img, read off the batch throughput line 89.9 img/s",
            "V-B front end 1.52 ms", "always-AdaIR 333 ms", "the 219x ratio"],
        "torch": torch.__version__, "opencv": cv2.__version__,
    }
    payload = {"config": cfg,
               "image_sizes": {"degraded": size_d, "clean": size_c},
               "results": results}
    res_path = OUT_DIR / "L2_timing_full_pool.results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[out] wrote {res_path}")


if __name__ == "__main__":
    main()
