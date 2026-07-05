r"""
M_significance_addenda.py  两个遗留论断的显著性补齐

WHY
  投 IEEE Access 前, 论文里两句措辞尚无 CI/显著性支撑, 而我们对 AdaIR-VA
  已用 paired bootstrap 定过标准, 不能对自己双标:

    M1  GTSDB clean 上 "VA 比 no-enhancement 低 1.39pp" 这句。是 slightly
        还是 significantly? 需要对 va_A 与 passthrough 在 eval split 上做
        paired 检验。
    M2  "learned router 在真实退化上优势消失/有限"(6 月记录里 MLP 58.05 vs
        VA-rule 57.32, +0.73pp, 当时 significance pending)。+0.73 显著与否,
        决定这句是 "advantage disappears" 还是 "small but significant gain
        at the cost of 3e4 synthetic training rows"。logreg 56.04 也一并测。

DESIGN (defensible, no re-implementation)
  两个分析都直接 import 既有脚本的原函数, 复用同一份路由/模型代码, 因此
  与 I / H 的结果构造性一致, 不引入新实现带来的偏差:
    - M1  import I_gtsdb_eval: route_decision + GTSRB_THRESHOLDS, 读
          gtsdb_master_cache.csv (含 split 列), 只取 eval split。
    - M2  import H_learned_router: load_train / load_cache / LogReg / MLP,
          用 H 的默认超参与 seed=42 原样重训, 应用到 cure_master_cache.csv。
  检验方法:
    - 主判决 McNemar exact(配对二元结果的标准检验, 精确二项而非卡方近似)。
    - 佐证 paired bootstrap 95% CI(B=5000, seed=42), 与 K 对 AdaIR-VA 同法。
  point estimate 用 cell-averaged deg-avg(与 H/K 报的 57.32 / 58.05 同口径)
  以便交叉核对; GTSDB 无 cell 结构, 用 crop-level accuracy。

USAGE
    python M_significance_addenda.py
    python M_significance_addenda.py --boot 10000 --skip-m1   (仅 M2)

OUTPUT (outputs_revision\)
    M_significance_addenda.results.json / .run_config.json / .execution_log.txt
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

OUT_DIR = PROJECT_ROOT / "outputs_revision"
LOG_PATH = OUT_DIR / "M_significance_addenda.execution_log.txt"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------
# paired-test helpers
# ------------------------------------------------------------------
def _log_binom_tail(k, n):
    """log P(X<=k) for X~Binom(n,0.5), numerically stable."""
    ln2 = math.log(2.0)
    lg = math.lgamma
    logs = [lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * ln2
            for i in range(k + 1)]
    m = max(logs)
    return m + math.log(sum(math.exp(x - m) for x in logs))


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar on paired binary outcomes (a vs b, same items).
    Two-sided exact-binomial p on the discordant pairs (p=0.5), computed
    in log space to avoid overflow; a continuity-corrected normal z is
    reported alongside as a cross-check."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n01 = int(np.sum(~a & b))          # a wrong, b right
    n10 = int(np.sum(a & ~b))          # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0,
                "p_two_sided": 1.0, "z_normal_cc": 0.0}
    k = min(n01, n10)
    log_tail = _log_binom_tail(k, n)
    p = min(1.0, 2.0 * math.exp(log_tail))
    z = (abs(n01 - n10) - 1.0) / math.sqrt(n) if n > 0 else 0.0
    return {"n01": n01, "n10": n10, "n_discordant": n,
            "p_two_sided": p, "z_normal_cc": round(z, 4)}


def paired_bootstrap_diff(a_correct, b_correct, groups=None,
                          B=5000, seed=42):
    """95% CI for mean(a)-mean(b) by resampling items with replacement.
    If groups is given, the point estimate and each replicate are
    group-averaged first (cell-averaged), else plain item mean."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a_correct, dtype=np.float64)
    b = np.asarray(b_correct, dtype=np.float64)
    n = a.size
    idx_all = np.arange(n)

    def metric(idx):
        if groups is None:
            return a[idx].mean() - b[idx].mean()
        g = groups[idx]
        diffs = []
        for gv in np.unique(g):
            m = g == gv
            diffs.append(a[idx][m].mean() - b[idx][m].mean())
        return float(np.mean(diffs))

    point = metric(idx_all) * 100.0
    boot = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        boot[i] = metric(s) * 100.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point_pp": round(point, 3),
            "ci95_pp": [round(float(lo), 3), round(float(hi), 3)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def cell_avg_acc(correct, groups):
    correct = np.asarray(correct, dtype=np.float64)
    return float(np.mean([correct[groups == g].mean()
                          for g in np.unique(groups)]) * 100.0)


# ------------------------------------------------------------------
# M1  GTSDB: va_A vs passthrough (and va_A vs va_B) on eval split
# ------------------------------------------------------------------
def run_m1(boot, seed):
    import I_gtsdb_eval as I
    cache = I.OUT_CSV_DEFAULT
    if not Path(cache).exists():
        log(f"[M1] SKIP: cache not found: {cache}")
        return {"status": "skipped", "reason": f"missing {cache}"}
    rows = []
    with open(cache, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split") != "eval":
                continue
            r["true"] = int(r["true"])
            for br in I.BRANCHES:
                r[f"pred_{br}"] = int(r[f"pred_{br}"])
            r["b"], r["c"], r["e"] = float(r["b"]), float(r["c"]), float(r["e"])
            rows.append(r)
    log(f"[M1] GTSDB eval crops: {len(rows)}")

    # recompute threshold set B exactly as I does (calib split, P15/P70)
    calib = []
    with open(cache, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split") == "calib":
                calib.append((float(r["b"]), float(r["c"]), float(r["e"])))
    cb = np.array([x[0] for x in calib])
    cc = np.array([x[1] for x in calib])
    ce = np.array([x[2] for x in calib])
    T_B = {"T1": float(np.percentile(cb, 15)),
           "T2": float(np.percentile(cc, 15)),
           "T3": float(np.percentile(ce, 15)),
           "T4": float(np.percentile(cb, 70))}

    def preds(name):
        out = []
        for r in rows:
            if name == "passthrough":
                p = r["pred_passthrough"]
            elif name == "va_A":
                br = I.route_decision(r["b"], r["c"], r["e"],
                                      I.GTSRB_THRESHOLDS)
                p = r[f"pred_{br}"]
            elif name == "va_B":
                br = I.route_decision(r["b"], r["c"], r["e"], T_B)
                p = r[f"pred_{br}"]
            out.append(1 if p == r["true"] else 0)
        return np.asarray(out, dtype=np.int64)

    pass_c = preds("passthrough")
    vaA_c = preds("va_A")
    vaB_c = preds("va_B")
    acc = {"passthrough": round(100 * pass_c.mean(), 2),
           "va_A": round(100 * vaA_c.mean(), 2),
           "va_B": round(100 * vaB_c.mean(), 2)}
    log(f"[M1] acc: {acc}")

    res = {"status": "ok", "n_eval": len(rows), "acc": acc}
    # headline: does VA (va_A) differ from no-enhancement (passthrough)?
    res["va_A_vs_passthrough"] = {
        "mcnemar": mcnemar_exact(vaA_c, pass_c),
        "bootstrap": paired_bootstrap_diff(vaA_c, pass_c, None, boot, seed)}
    # secondary: threshold-transfer (A vs B)
    res["va_A_vs_va_B"] = {
        "mcnemar": mcnemar_exact(vaA_c, vaB_c),
        "bootstrap": paired_bootstrap_diff(vaA_c, vaB_c, None, boot, seed)}
    log(f"[M1] va_A - passthrough: {res['va_A_vs_passthrough']}")
    log(f"[M1] va_A - va_B: {res['va_A_vs_va_B']}")
    return res


# ------------------------------------------------------------------
# M2  CURE degraded: learned router (logreg, mlp) vs VA-rule
# ------------------------------------------------------------------
def run_m2(boot, seed):
    import H_learned_router as H
    train_csv = H.TRAIN_CSV_DEFAULT
    cure_csv = H.CACHE_CSV_DEFAULT
    for p in (train_csv, cure_csv):
        if not Path(p).exists():
            log(f"[M2] SKIP: missing {p}")
            return {"status": "skipped", "reason": f"missing {p}"}

    # replicate H's training split + standardization exactly (seed=42)
    X, y, ach, rule_idx_tr, cond = H.load_train(train_csv)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * 0.1)))
    tr_i = perm[n_val:]
    mu, sd = X[tr_i].mean(axis=0), X[tr_i].std(axis=0) + 1e-9

    models = {
        "logreg": H.LogReg(seed=seed).fit((X[tr_i] - mu) / sd, y[tr_i],
                                          epochs=300, lr=0.05),
        "mlp": H.MLP(hidden=16, seed=seed).fit((X[tr_i] - mu) / sd, y[tr_i],
                                               epochs=400, lr=0.01)}

    Xc, corr, rule_idx, ch_id, ch_name, sev = H.load_cache(cure_csv)
    Xcs = (Xc - mu) / sd
    deg = ch_id != 0
    cells = np.array([f"{c}_{s}" for c, s in zip(ch_id, sev)])

    # per-image correctness for VA-rule (lookup of the rule's chosen branch)
    rule_c = corr[np.arange(len(corr)), rule_idx]

    # cell-balance disclosure (image-mean == cell-mean iff balanced)
    _, counts = np.unique(cells[deg], return_counts=True)
    balanced = bool(np.all(counts == counts[0]))
    log(f"[M2] degraded images: {int(deg.sum())}; cells: {len(counts)}; "
        f"balanced={balanced} (sizes {counts.min()}..{counts.max()})")

    va_deg_cell = cell_avg_acc(rule_c[deg], cells[deg])
    res = {"status": "ok", "n_deg": int(deg.sum()),
           "cells_balanced": balanced,
           "va_rule_deg_cellavg": round(va_deg_cell, 2),
           "selectors": {}}
    log(f"[M2] VA-rule deg-avg (cell) = {va_deg_cell:.2f}  "
        f"(expect ~57.32)")

    for name, m in models.items():
        idx = m.predict(Xcs)
        sel_c = corr[np.arange(len(corr)), idx]
        sel_deg_cell = cell_avg_acc(sel_c[deg], cells[deg])
        entry = {
            "deg_cellavg": round(sel_deg_cell, 2),
            "point_diff_pp_cellavg": round(sel_deg_cell - va_deg_cell, 3),
            "mcnemar": mcnemar_exact(sel_c[deg], rule_c[deg]),
            "bootstrap": paired_bootstrap_diff(
                sel_c[deg], rule_c[deg], cells[deg], boot, seed)}
        res["selectors"][name] = entry
        log(f"[M2] {name}: deg-avg={sel_deg_cell:.2f} "
            f"(diff {entry['point_diff_pp_cellavg']:+.2f}pp)  "
            f"McNemar p={entry['mcnemar']['p_two_sided']:.2e}  "
            f"boot {entry['bootstrap']['ci95_pp']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-m1", action="store_true")
    ap.add_argument("--skip-m2", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"M_significance_addenda start; boot={args.boot} seed={args.seed}")

    out = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "boot": args.boot, "seed": args.seed}
    out["M1_gtsdb"] = ({"status": "skipped", "reason": "--skip-m1"}
                       if args.skip_m1 else run_m1(args.boot, args.seed))
    out["M2_router"] = ({"status": "skipped", "reason": "--skip-m2"}
                        if args.skip_m2 else run_m2(args.boot, args.seed))

    res_path = OUT_DIR / "M_significance_addenda.results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "M_significance_addenda.run_config.json", "w",
              encoding="utf-8") as f:
        json.dump({"timestamp": out["timestamp"], "boot": args.boot,
                   "seed": args.seed,
                   "imports": ["I_gtsdb_eval", "H_learned_router"],
                   "method": "McNemar exact (primary) + paired bootstrap "
                             "95% CI (corroboration)",
                   "point_metric": "cell-averaged deg-avg (CURE) / "
                                   "crop accuracy (GTSDB)"}, f,
                  indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    for tag, block in [("M1 GTSDB va_A vs passthrough", out["M1_gtsdb"]),
                       ("M2 router vs VA-rule", out["M2_router"])]:
        print(f"[{tag}] {block.get('status')}")
    print(f"[out] wrote {res_path}")


if __name__ == "__main__":
    main()
