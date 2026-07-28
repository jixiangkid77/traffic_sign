r"""
N_backbone_significance.py  三 backbone 上 VA 增益的显著性补齐

WHY
  backbone_ablation.json 只存了聚合 acc/f1, 做 McNemar 需要逐图对错。本脚本
  重跑 evaluate_backbone_ablation.py 的评估阶段(GTSRB 合成退化, 三 backbone),
  唯一区别是记录逐图 baseline_correct / va_correct, 据此对每个 backbone 检验
  "VA 相对 no-enhancement 的退化平均增益"是否统计显著。不重训, 复用三个现成
  checkpoint 与 enhance 的原函数, 结果与原 ablation 构造性一致。

DESIGN
  - 复用 evaluate_backbone_ablation 的 BACKBONES / TESTSETS / 三个 builder,
    enhance 的 no_enhance / adaptive_enhance, model.build_model, 以及逐字复制
    的阈值/标签/测试集路径/transform 装配, 保证前向路径与原 ablation 一致。
  - 增强只依赖图像不依赖 backbone: 每张图只增强一次并缓存(raw = 恒等,
    va = adaptive_enhance 一次), 三个 backbone 复用同一缓存, 把增强量从
    3x6x12630 降到 6x12630。前向仍逐 backbone 独立跑, 与原 cell 结果一致。
  - point estimate 用退化平均(5 个退化 testset 各自 acc 再平均, 与
    backbone_ablation.csv 的 DegradedAvg 同口径)以便交叉核对。
  - 主判决 McNemar exact(配对二元, 精确二项, 逐字复用 M 脚本里已对过 scipy
    的实现); 佐证 paired bootstrap 95% CI, 按 testset 分组使 bootstrap 的
    度量与退化平均一致(每个退化 testset 等权)。
  - clean 作为次要行一并报告(VA 在 clean 的代价是否显著), 与 M1 处理 GTSDB
    clean 的方式对称。

USAGE
    python N_backbone_significance.py
    python N_backbone_significance.py --boot 10000

OUTPUT (results\)
    N_backbone_significance.results.json / .run_config.json / .execution_log.txt
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

import evaluate_backbone_ablation as EBA          # noqa: E402
from enhance import no_enhance, adaptive_enhance  # noqa: E402

PROJECT_ROOT = THIS_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_PATH = RESULTS_DIR / "N_backbone_significance.execution_log.txt"

DEG_TESTSETS = ["lowlight", "foggy", "lowcontrast", "noisy", "mixed"]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------
# paired-test helpers (verbatim from M_significance_addenda, scipy-checked)
# ------------------------------------------------------------------
def _log_binom_tail(k, n):
    ln2 = math.log(2.0)
    lg = math.lgamma
    logs = [lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * ln2
            for i in range(k + 1)]
    m = max(logs)
    return m + math.log(sum(math.exp(x - m) for x in logs))


def mcnemar_exact(a_correct, b_correct):
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0,
                "p_two_sided": 1.0, "z_normal_cc": 0.0}
    k = min(n01, n10)
    p = min(1.0, 2.0 * math.exp(_log_binom_tail(k, n)))
    z = (abs(n01 - n10) - 1.0) / math.sqrt(n)
    return {"n01": n01, "n10": n10, "n_discordant": n,
            "p_two_sided": p, "z_normal_cc": round(z, 4)}


def paired_bootstrap_diff(a_correct, b_correct, groups=None, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a_correct, dtype=np.float64)
    b = np.asarray(b_correct, dtype=np.float64)
    n = a.size
    idx_all = np.arange(n)

    def metric(idx):
        if groups is None:
            return a[idx].mean() - b[idx].mean()
        g = groups[idx]
        return float(np.mean([a[idx][g == gv].mean() - b[idx][g == gv].mean()
                              for gv in np.unique(g)]))

    point = metric(idx_all) * 100.0
    boot = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        boot[i] = metric(s) * 100.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point_pp": round(point, 3),
            "ci95_pp": [round(float(lo), 3), round(float(hi), 3)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ------------------------------------------------------------------
# setup replicated verbatim from evaluate_backbone_ablation.main()
# ------------------------------------------------------------------
def load_setup():
    with open(RESULTS_DIR / "thresholds.json") as f:
        th = json.load(f)
    T = (th["T1_brightness_low"], th["T2_contrast_low"],
         th["T3_edge_low"], th["T4_brightness_high"])

    csv_candidates = [DATA_ROOT / "gtsrb" / "GT-final_test.csv",
                      DATA_ROOT / "gtsrb" / "Final_Test" / "GT-final_test.csv"]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        for p in DATA_ROOT.rglob("GT-final_test.csv"):
            csv_path = p
            break
    labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            labels[row["Filename"].rsplit(".", 1)[0]] = int(row["ClassId"])

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.3401, 0.3120, 0.3212],
                             [0.2725, 0.2609, 0.2669])])

    dirs = {"clean": DATA_ROOT / "gtsrb" / "GTSRB" / "Final_Test" / "Images",
            "lowlight": DATA_ROOT / "gtsrb_lowlight",
            "foggy": DATA_ROOT / "gtsrb_foggy",
            "lowcontrast": DATA_ROOT / "gtsrb_lowcontrast",
            "noisy": DATA_ROOT / "gtsrb_noisy",
            "mixed": DATA_ROOT / "gtsrb_mixed"}
    if not dirs["clean"].exists():
        for p in DATA_ROOT.rglob("Final_Test"):
            if p.is_dir():
                dirs["clean"] = p / "Images"
                break
    return T, labels, tf, dirs


def list_samples(image_dir, labels):
    out = []
    for p in sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.ppm")):
        if p.stem in labels:
            out.append((p, labels[p.stem]))
    return out


def cache_testset(samples, T):
    """Enhance each image once. Returns (raw_rgb_list, va_rgb_list, y)."""
    raw, va, y = [], [], []
    for path, label in samples:
        bgr = cv2.imread(str(path))
        raw.append(cv2.cvtColor(no_enhance(bgr), cv2.COLOR_BGR2RGB))
        va.append(cv2.cvtColor(adaptive_enhance(bgr, *T), cv2.COLOR_BGR2RGB))
        y.append(label)
    return raw, va, np.asarray(y, dtype=np.int64)


def forward_correct(model, rgb_list, y, tf, device, bs=256):
    preds = np.empty(len(rgb_list), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(rgb_list), bs):
            batch = torch.stack([tf(Image.fromarray(im))
                                 for im in rgb_list[s:s + bs]]).to(device)
            preds[s:s + bs] = model(batch).argmax(1).cpu().numpy()
    return (preds == y).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    log(f"N_backbone_significance start; device={device}; "
        f"boot={args.boot} seed={args.seed}")

    T, labels, tf, dirs = load_setup()
    log(f"thresholds T1..T4 = {tuple(round(x, 4) for x in T)}; "
        f"labels={len(labels)}")

    all_ts = ["clean"] + DEG_TESTSETS
    # cache enhancement once per testset (raw + va), reused by all backbones
    cache = {}
    for ts in all_ts:
        samples = list_samples(dirs[ts], labels)
        cache[ts] = cache_testset(samples, T)
        log(f"cached {ts}: {len(samples)} images (enhanced once)")

    # per-backbone correctness, then significance
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "boot": args.boot, "seed": args.seed,
           "testbed": "GTSRB synthetic degradations (12630-image test set)",
           "deg_testsets": DEG_TESTSETS, "backbones": {}}

    for name, builder, weight_file in EBA.BACKBONES:
        wp = MODELS_DIR / weight_file
        if not wp.exists():
            log(f"[{name}] SKIP: missing {wp}")
            out["backbones"][name] = {"status": "skipped"}
            continue
        model = builder(num_classes=43)
        model.load_state_dict(torch.load(wp, map_location=device))
        model = model.to(device).eval()
        log(f"[{name}] loaded {weight_file}")

        base_c, va_c, grp = {}, {}, {}
        for ts in all_ts:
            raw_rgb, va_rgb, y = cache[ts]
            base_c[ts] = forward_correct(model, raw_rgb, y, tf, device)
            va_c[ts] = forward_correct(model, va_rgb, y, tf, device)

        # degraded-avg (mean of 5 per-testset accuracies) -- sanity anchor
        base_deg = float(np.mean([base_c[ts].mean() for ts in DEG_TESTSETS]))
        va_deg = float(np.mean([va_c[ts].mean() for ts in DEG_TESTSETS]))

        # pool degraded images + testset group id for grouped bootstrap
        base_pool = np.concatenate([base_c[ts] for ts in DEG_TESTSETS])
        va_pool = np.concatenate([va_c[ts] for ts in DEG_TESTSETS])
        gid = np.concatenate([np.full(len(base_c[ts]), i)
                              for i, ts in enumerate(DEG_TESTSETS)])

        entry = {
            "status": "ok",
            "baseline_deg_avg": round(base_deg * 100, 2),
            "va_deg_avg": round(va_deg * 100, 2),
            "delta_deg_pp": round((va_deg - base_deg) * 100, 2),
            "va_vs_baseline_degraded": {
                "mcnemar": mcnemar_exact(va_pool, base_pool),
                "bootstrap": paired_bootstrap_diff(
                    va_pool, base_pool, gid, args.boot, args.seed)},
            "clean": {
                "baseline_acc": round(base_c["clean"].mean() * 100, 2),
                "va_acc": round(va_c["clean"].mean() * 100, 2),
                "delta_pp": round((va_c["clean"].mean()
                                   - base_c["clean"].mean()) * 100, 2),
                "mcnemar": mcnemar_exact(va_c["clean"], base_c["clean"]),
                "bootstrap": paired_bootstrap_diff(
                    va_c["clean"], base_c["clean"], None,
                    args.boot, args.seed)}}
        out["backbones"][name] = entry
        log(f"[{name}] deg: base={entry['baseline_deg_avg']} "
            f"va={entry['va_deg_avg']} (Δ{entry['delta_deg_pp']:+.2f}pp)  "
            f"McNemar p={entry['va_vs_baseline_degraded']['mcnemar']['p_two_sided']:.2e}  "
            f"boot {entry['va_vs_baseline_degraded']['bootstrap']['ci95_pp']}")
        log(f"[{name}] clean: base={entry['clean']['baseline_acc']} "
            f"va={entry['clean']['va_acc']} (Δ{entry['clean']['delta_pp']:+.2f}pp)  "
            f"McNemar p={entry['clean']['mcnemar']['p_two_sided']:.2e}")
        del model

    out["elapsed_min"] = round((time.time() - t0) / 60, 2)
    res_path = RESULTS_DIR / "N_backbone_significance.results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "N_backbone_significance.run_config.json", "w",
              encoding="utf-8") as f:
        json.dump({"timestamp": out["timestamp"], "boot": args.boot,
                   "seed": args.seed, "device": str(device),
                   "reuses": ["evaluate_backbone_ablation", "enhance",
                              "model.build_model"],
                   "method": "McNemar exact (primary) + paired bootstrap "
                             "95% CI grouped by testset (corroboration)",
                   "point_metric": "degraded-average (5 testsets equally "
                                   "weighted), matches backbone_ablation.csv"},
                  f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY: VA vs baseline, degraded-avg per backbone ===")
    print(f"{'backbone':14s} {'base':>6s} {'va':>6s} {'Δpp':>6s} "
          f"{'McNemar p':>11s} {'boot 95% CI':>18s}")
    for name, e in out["backbones"].items():
        if e.get("status") != "ok":
            print(f"{name:14s} skipped")
            continue
        d = e["va_vs_baseline_degraded"]
        print(f"{name:14s} {e['baseline_deg_avg']:6.2f} {e['va_deg_avg']:6.2f} "
              f"{e['delta_deg_pp']:+6.2f} {d['mcnemar']['p_two_sided']:11.2e} "
              f"{str(d['bootstrap']['ci95_pp']):>18s}")
    print(f"\n[out] wrote {res_path}  ({out['elapsed_min']} min)")


if __name__ == "__main__":
    main()
