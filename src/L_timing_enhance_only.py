r"""
L_timing_enhance_only.py  对称口径的 per-image 增强延迟测量

WHY
  K 的 efficiency 行里, VA 的 <0.2 ms 是"仅路由+算子"(Table 4 口径),
  AdaIR ~333 ms / CIDNet ~89 ms 是 J 全量跑的均摊值(端到端, 含磁盘 IO,
  批处理与分类器)。两个口径不能直接相除。本脚本在同一 harness 下, 用同一批
  真实 CURE 图, batch=1, torch 与 cv2 都固定单线程, 分别测量四个 scope:

     va_rule   compute_stats + route_decision + apply_branch
               (与 F_master_sweep_cache.py 完全同一份函数, 原始分辨率)
     adair     J_local_deep_eval.enhance_batch 单图
               (uint8 BGR 进出, 含 BGR/RGB 转换, pad/8, 张量转换, forward)
     cidnet    同上
     classify  cv2.resize(32) + BGR2RGB + ToTensor/Normalize + CompactCNN
               forward(各方法共享的下游成本, 单独列出, 不计入前三行)

  报告 mean / median / p95 (ms/img)。取样对 25 个退化 cell + ChallengeFree
  做 round-robin 分层(seed 固定), 所有 scope 用同一份图像序列, 磁盘 IO 在
  计时前一次性完成, 不进入任何 scope。

USAGE
  python L_timing_enhance_only.py                     (两个深度模型, n=200)
  python L_timing_enhance_only.py --models cidnet --n 100
  python L_timing_enhance_only.py --models va-only    (只测 VA 与 classify)
  python L_timing_enhance_only.py --adair-weight D:\path\to\adair5d.ckpt

OUTPUT (outputs_revision\)
  L_timing_enhance_only.results.json      结果 + 配置(单一 JSON 工件)
  L_timing_enhance_only.run_config.json   运行配置
  L_timing_enhance_only.execution_log.txt 控制台日志副本
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

OUT_DIR = PROJECT_ROOT / "outputs_revision"
LOG_PATH = OUT_DIR / "L_timing_enhance_only.execution_log.txt"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stratified_pick(samples, n, seed):
    """Round-robin across (ch, sev) groups so every cell contributes."""
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
    a = np.asarray(ts, dtype=np.float64) * 1000.0
    return {"n_timed": int(a.size),
            "mean_ms": round(float(a.mean()), 3),
            "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


def time_va(imgs, warmup):
    ts, branches = [], Counter()
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        b, c, e = F.compute_stats(img)
        br = F.route_decision(b, c, e, F.THRESHOLDS)
        _ = F.apply_branch(img, br)
        dt = time.perf_counter() - t0
        branches[br] += 1
        if i >= warmup:
            ts.append(dt)
    out = pstats(ts)
    out["branch_histogram"] = dict(branches)
    return out


def time_deep(net, imgs, warmup):
    ts = []
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        _ = J.enhance_batch(net, [img], "cpu")
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return pstats(ts)


def time_classify(model, tfm, imgs, warmup):
    ts = []
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        r = cv2.resize(img, (J.INPUT_SIZE, J.INPUT_SIZE))
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        x = tfm(rgb).unsqueeze(0)
        with torch.no_grad():
            _ = model(x)
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return pstats(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="both",
                    choices=["both", "adair", "cidnet", "va-only"])
    ap.add_argument("--n", type=int, default=200,
                    help="timed images per scope (after warmup)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="leading images excluded from statistics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cure-root", default=str(J.CURE_TSR_DIR_DEFAULT))
    ap.add_argument("--adair-weight",
                    default=str(PROJECT_ROOT / "models" / "adair5d.ckpt"))
    ap.add_argument("--cidnet-weight", default="",
                    help="optional local CIDNet weight; default = HF cache")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    log(f"threads pinned: torch=1 cv2=1; torch {torch.__version__}, "
        f"opencv {cv2.__version__}")

    samples, n_files = J.collect_samples(Path(args.cure_root))
    if not samples:
        sys.exit(f"[FATAL] no CURE images under {args.cure_root}")
    picked = stratified_pick(samples, args.n + args.warmup, args.seed)
    log(f"scanned {n_files} files; picked {len(picked)} "
        f"({args.warmup} warmup + {args.n} timed), seed={args.seed}")

    imgs, shapes = [], []
    for s in picked:
        im = cv2.imread(str(s["path"]))
        if im is None:
            log(f"[warn] unreadable, skipped: {s['path']}")
            continue
        imgs.append(im)
        shapes.append(im.shape[:2])
    hs = [h for h, w in shapes]
    ws = [w for h, w in shapes]
    size_info = {"n_loaded": len(imgs),
                 "h_mean": round(float(np.mean(hs)), 1),
                 "h_min": int(min(hs)), "h_max": int(max(hs)),
                 "w_mean": round(float(np.mean(ws)), 1),
                 "w_min": int(min(ws)), "w_max": int(max(ws))}
    log(f"image sizes: H mean {size_info['h_mean']} "
        f"[{size_info['h_min']}, {size_info['h_max']}], "
        f"W mean {size_info['w_mean']} "
        f"[{size_info['w_min']}, {size_info['w_max']}]")

    results = {}

    log("timing scope: va_rule (stats + route + operator, original size)")
    results["va_rule"] = time_va(imgs, args.warmup)
    log(f"  va_rule: {results['va_rule']}")

    log("timing scope: classify (resize32 + BGR2RGB + transform + "
        "CompactCNN forward, batch=1)")
    clf = load_gtsrb_compactcnn("cpu")
    tfm = J.build_transform()
    results["classify32"] = time_classify(clf, tfm, imgs, args.warmup)
    log(f"  classify32: {results['classify32']}")

    if args.models in ("both", "adair"):
        net, _ = J.load_adair("cpu", args.adair_weight)
        log("timing scope: adair enhance (uint8 BGR in/out, single image)")
        results["adair_enhance"] = time_deep(net, imgs, args.warmup)
        log(f"  adair_enhance: {results['adair_enhance']}")
        del net

    if args.models in ("both", "cidnet"):
        net, _ = J.load_cidnet("cpu", args.cidnet_weight)
        log("timing scope: cidnet enhance (uint8 BGR in/out, single image)")
        results["cidnet_enhance"] = time_deep(net, imgs, args.warmup)
        log(f"  cidnet_enhance: {results['cidnet_enhance']}")
        del net

    cfg = {
        "script": "L_timing_enhance_only.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": args.models, "n": args.n, "warmup": args.warmup,
        "seed": args.seed, "cure_root": str(args.cure_root),
        "adair_weight": args.adair_weight
        if args.models in ("both", "adair") else None,
        "cidnet_weight": (args.cidnet_weight or "HF cache")
        if args.models in ("both", "cidnet") else None,
        "threads": {"torch": 1, "cv2": 1},
        "scope_semantics": {
            "va_rule": "compute_stats + route_decision + apply_branch, "
                       "original resolution, same functions as F cache run",
            "adair_enhance": "J.enhance_batch single image: BGR2RGB, /255, "
                             "pad reflect to /8, forward, clamp, unpad, "
                             "back to uint8 BGR",
            "cidnet_enhance": "same wrapper as adair_enhance",
            "classify32": "shared downstream stage, excluded from the "
                          "three enhancement scopes",
        },
        "torch": torch.__version__, "opencv": cv2.__version__,
    }
    payload = {"config": cfg, "image_sizes": size_info, "results": results}

    res_path = OUT_DIR / "L_timing_enhance_only.results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "L_timing_enhance_only.run_config.json", "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("\n=== per-image latency, batch=1, 1 CPU thread (ms) ===")
    print(f"{'scope':16s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'n':>5s}")
    for k in ("va_rule", "adair_enhance", "cidnet_enhance", "classify32"):
        if k in results:
            r = results[k]
            print(f"{k:16s} {r['mean_ms']:8.3f} {r['median_ms']:8.3f} "
                  f"{r['p95_ms']:8.3f} {r['n_timed']:5d}")
    if "branch_histogram" in results.get("va_rule", {}):
        print(f"va branch mix: {results['va_rule']['branch_histogram']}")
    print(f"\n[out] wrote {res_path}")


if __name__ == "__main__":
    main()
