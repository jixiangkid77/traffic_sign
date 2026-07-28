r"""
bootstrap_ci.py: CURE-TSR 真实退化结果的配对自助置信区间 (实验 2)

目的
  VA-Adaptive 对 FFA-Net 的 macro-F1 优势只有 0.38pp (35.42 vs 35.04),
  审稿人和你的二审 AI 会质疑这个"赢"是不是噪声。本脚本对 CURE-TSR 真实退化集
  做配对 (paired)、按 challenge x severity 分层 (stratified) 的自助重采样,
  给出每个方法 degraded-average acc / macro-F1 的 95% CI, 以及关键差值
  (va_adaptive - ffa_net, va_adaptive - baseline) 的 95% CI 和显著性判断。

输入
  outputs_cure_tsr\cure_tsr_per_image_predictions.csv  (先跑 dump_cure_predictions.py)
  列: split, challenge, severity, method, idx, true_label, pred_label, correct

degraded-average 的定义
  对 5 challenges x 5 severities = 25 个 cell, 先各自算 cell 内指标, 再对 25 个 cell
  取等权平均。脚本会同时打印 pooled (把全部退化图汇在一起) 的点估计做交叉核对:
  哪个点估计与正文的 57.32 / 35.42 对得上, 就说明正文用的是哪种口径。
  默认对 cell-averaged 口径做自助; 若你正文用的是 pooled, 把 AGG 改成 "pooled"。

自助过程
  配对: 同一 cell 内对所有方法用同一组重采样下标 (同一批图喂给所有方法)。
  分层: 在每个 cell 内独立重采样, 保持 5x5 设计结构。
  B 次重采样, 每次得各方法的 degraded-avg 与差值; 取 2.5 / 97.5 百分位为 95% CI。
  差值 95% CI 不跨 0 => 该差异在 95% 水平显著。

用法
  conda activate pcm_sim
  python src\bootstrap_ci.py
B=5000 时约几十秒到一两分钟。
"""
import csv
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
OUTPUT_DIR = PROJECT_ROOT / "outputs_cure_tsr"
PRED_CSV = OUTPUT_DIR / "cure_tsr_per_image_predictions.csv"

# 5 个映射目标 GTSRB 类 (与评测脚本一致): {3:9, 6:14, 11:12, 12:17, 13:13}
MAPPED_CLASSES = sorted({9, 14, 12, 17, 13})        # [9, 12, 13, 14, 17]

B = 5000                 # 自助次数; 想更稳可加到 10000
SEED = 42                # 与项目其他抽样一致
AGG = "cell"             # "cell" = 25 个 cell 等权平均 (与正文 avg-deg 一致); "pooled" = 全图汇总
KEY_DIFFS = [("va_adaptive", "ffa_net"), ("va_adaptive", "baseline")]


def macro_f1_fast(true, pred, classes):
    """与 sklearn macro-F1 (labels=classes, zero_division=0) 等价的 numpy 实现。"""
    f1s = []
    for c in classes:
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {PRED_CSV}\n请先跑: python src\\dump_cure_predictions.py")

    rows = list(csv.DictReader(open(PRED_CSV, encoding="utf-8")))
    methods = sorted({r["method"] for r in rows})

    # 退化 cell (排除 ChallengeFree)
    cells = sorted({(r["challenge"], int(r["severity"]))
                    for r in rows
                    if r["split"] != "challengefree"
                    and r["challenge"].lower() != "challengefree"})

    # data[cell][method] = (true[], pred[]), 各方法按共有 idx 对齐
    data = {}
    for ch, sev in cells:
        per_method = {m: {} for m in methods}
        for r in rows:
            if (r["split"] != "challengefree" and r["challenge"] == ch
                    and int(r["severity"]) == sev):
                per_method[r["method"]][int(r["idx"])] = (
                    int(r["true_label"]), int(r["pred_label"]))
        common = None
        for m in methods:
            ks = set(per_method[m].keys())
            common = ks if common is None else (common & ks)
        common = sorted(common or [])
        if not common:
            continue
        # 一致性检查: 同一 (cell, idx) 的 true_label 必须跨方法相同, 否则配对错位
        ref = methods[0]
        for i in common:
            t_ref = per_method[ref][i][0]
            for m in methods[1:]:
                if per_method[m][i][0] != t_ref:
                    raise ValueError(
                        f"配对错位: cell=({ch},{sev}) idx={i} 的 true_label 跨方法不一致 "
                        f"({ref}={t_ref}, {m}={per_method[m][i][0]})。"
                        f"说明各方法图序不一致, 不能配对自助。")
        data[(ch, sev)] = {
            m: (np.array([per_method[m][i][0] for i in common], dtype=np.int64),
                np.array([per_method[m][i][1] for i in common], dtype=np.int64))
            for m in methods
        }

    used_cells = sorted(data.keys())
    classes = np.array(MAPPED_CLASSES, dtype=np.int64)

    def cell_metrics(true, pred):
        acc = float(np.mean(pred == true)) * 100.0
        f1 = macro_f1_fast(true, pred, classes) * 100.0
        return acc, f1

    def degraded_avg(resample=False, rng=None):
        """返回 {method: (acc, f1)}; AGG 决定 cell-averaged 还是 pooled。"""
        if AGG == "cell":
            acc_cells = {m: [] for m in methods}
            f1_cells = {m: [] for m in methods}
            for cell in used_cells:
                n = len(next(iter(data[cell].values()))[0])
                pos = rng.integers(0, n, size=n) if resample else None
                for m in methods:
                    true, pred = data[cell][m]
                    if resample:
                        true, pred = true[pos], pred[pos]
                    a, f = cell_metrics(true, pred)
                    acc_cells[m].append(a)
                    f1_cells[m].append(f)
            return {m: (float(np.mean(acc_cells[m])), float(np.mean(f1_cells[m])))
                    for m in methods}
        else:  # pooled
            out = {}
            # 配对池: 需要每个 cell 同步重采样后再汇总
            pooled = {m: ([], []) for m in methods}
            for cell in used_cells:
                n = len(next(iter(data[cell].values()))[0])
                pos = rng.integers(0, n, size=n) if resample else np.arange(n)
                for m in methods:
                    true, pred = data[cell][m]
                    pooled[m][0].append(true[pos])
                    pooled[m][1].append(pred[pos])
            for m in methods:
                true = np.concatenate(pooled[m][0])
                pred = np.concatenate(pooled[m][1])
                out[m] = cell_metrics(true, pred)
            return out

    # ---- 点估计 (cell 与 pooled 都打印, 用来核对正文口径) ----
    rng0 = np.random.default_rng(SEED)
    saved_agg = AGG
    pt = {}
    for mode in ("cell", "pooled"):
        globals()["AGG"] = mode
        pt[mode] = degraded_avg(resample=False, rng=rng0)
    globals()["AGG"] = saved_agg

    # ---- 自助 ----
    rng = np.random.default_rng(SEED)
    boot_acc = {m: np.empty(B) for m in methods}
    boot_f1 = {m: np.empty(B) for m in methods}
    for b in range(B):
        res = degraded_avg(resample=True, rng=rng)
        for m in methods:
            boot_acc[m][b], boot_f1[m][b] = res[m]

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    log("=" * 80)
    log(f"  CURE-TSR degraded-average bootstrap  (B={B}, seed={SEED}, agg={AGG})")
    log(f"  cells used: {len(used_cells)} (challenges x severities, ChallengeFree 排除)")
    log(f"  methods: {', '.join(methods)}")
    log("=" * 80)
    log("  [点估计核对] 与正文 avg-deg 对得上的那一列即为正文口径:")
    log(f"  {'method':<14}{'cell acc':>10}{'cell F1':>10}{'pooled acc':>12}{'pooled F1':>11}")
    for m in methods:
        log(f"  {m:<14}{pt['cell'][m][0]:>10.2f}{pt['cell'][m][1]:>10.2f}"
            f"{pt['pooled'][m][0]:>12.2f}{pt['pooled'][m][1]:>11.2f}")

    log("\n" + "=" * 80)
    log(f"  Degraded-average with 95% CI  (agg={AGG})")
    log("=" * 80)
    log(f"  {'method':<14}{'acc':>8}{'acc 95% CI':>22}{'F1':>8}{'F1 95% CI':>22}")
    pe = pt[AGG]
    for m in methods:
        la, ua = ci(boot_acc[m])
        lf, uf = ci(boot_f1[m])
        log(f"  {m:<14}{pe[m][0]:>8.2f} {f'[{la:.2f}, {ua:.2f}]':>21}"
            f"{pe[m][1]:>8.2f} {f'[{lf:.2f}, {uf:.2f}]':>21}")

    log("\n" + "=" * 80)
    log("  Paired differences  (正数 = 前者更高; 95% CI 不跨 0 即显著)")
    log("=" * 80)
    for m1, m2 in KEY_DIFFS:
        if m1 not in methods or m2 not in methods:
            log(f"  [skip] {m1} - {m2}: 缺方法")
            continue
        da, df = boot_acc[m1] - boot_acc[m2], boot_f1[m1] - boot_f1[m2]
        la, ua = ci(da)
        lf, uf = ci(df)
        sa = "SIGNIFICANT" if (la > 0 or ua < 0) else "not sig (CI 跨 0)"
        sf = "SIGNIFICANT" if (lf > 0 or uf < 0) else "not sig (CI 跨 0)"
        log(f"  {m1} - {m2}:")
        log(f"      acc diff = {pe[m1][0]-pe[m2][0]:+6.2f}   95% CI [{la:+.2f}, {ua:+.2f}]   -> {sa}")
        log(f"      F1  diff = {pe[m1][1]-pe[m2][1]:+6.2f}   95% CI [{lf:+.2f}, {uf:+.2f}]   -> {sf}")

    summary = {
        "script": "bootstrap_ci.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "B": B, "seed": SEED, "agg": AGG,
        "cells_used": len(used_cells),
        "mapped_classes": MAPPED_CLASSES,
        "methods": methods,
        "point_estimate_cell": {m: pt["cell"][m] for m in methods},
        "point_estimate_pooled": {m: pt["pooled"][m] for m in methods},
        "acc_ci": {m: ci(boot_acc[m]) for m in methods},
        "f1_ci": {m: ci(boot_f1[m]) for m in methods},
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (OUTPUT_DIR / "bootstrap_ci_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "bootstrap_ci_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\n  -> {OUTPUT_DIR / 'bootstrap_ci_summary.json'}")
    log(f"  -> {OUTPUT_DIR / 'bootstrap_ci_log.txt'}")


if __name__ == "__main__":
    main()
