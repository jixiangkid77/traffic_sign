"""
H_learned_router.py -- learned router (logistic regression + small MLP)
                       vs the hand-designed rule router

WHAT IT ANSWERS (reviewer's inevitable question)
  "Why fixed rules? Would LEARNING the branch selection from the same three
  statistics do better?"  This script trains two selectors on synthetic
  GTSRB training data (G_synth_router_data.py output) and evaluates them,
  by pure table lookup, on the authoritative CURE-TSR master cache
  (F_master_sweep_cache.py output).  Everything is deterministic and
  self-contained: numpy only, no torch, no sklearn.

DESIGN DECISIONS (documented for audit)
  - Features: exactly the router's three statistics (b, c, e), standardized
    with TRAIN-split mean/std.  Same information as the rule router.
  - Label: best branch per training image = argmax over the four branches of
    the frozen CNN's softmax probability of the TRUE class; ties resolve to
    the earliest branch in [passthrough, gamma, clahe, stretch], i.e. prefer
    doing nothing when equal (conservative).
  - Models: (1) multinomial logistic regression, (2) MLP 3-16-16-4 (ReLU),
    both trained full-batch with Adam, fixed seeds.
  - Deployment evaluation NEVER re-processes images: the CURE cache already
    stores each branch's prediction per image, so a router is evaluated by
    choosing a branch and looking its outcome up.

USAGE (after G has produced the training CSV):
    python H_learned_router.py
    python H_learned_router.py --train-csv ... --cure-cache ... --seed 42

OUTPUT
    outputs_revision\\learned_router_results.json
    console tables: synthetic validation + CURE deployment comparison
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
TRAIN_CSV_DEFAULT = PROJECT_ROOT / "outputs_revision" / "gtsrb_router_train.csv"
CACHE_CSV_DEFAULT = PROJECT_ROOT / "outputs_revision" / "cure_master_cache.csv"
OUT_JSON_DEFAULT = PROJECT_ROOT / "outputs_revision" / "learned_router_results.json"

BRANCHES = ["passthrough", "gamma", "clahe", "stretch"]


# ------------------------------------------------------------------
# data loading
# ------------------------------------------------------------------
def load_train(path):
    X, y, ach, rule_idx, cond = [], [], [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            X.append([float(r["b"]), float(r["c"]), float(r["e"])])
            ptrue = [float(r[f"ptrue_{br}"]) for br in BRANCHES]
            y.append(int(np.argmax(ptrue)))          # ties -> earliest branch
            t = int(r["true"])
            ach.append([1 if int(r[f"pred_{br}"]) == t else 0
                        for br in BRANCHES])
            rule_idx.append(BRANCHES.index(r["rule_branch"]))
            cond.append(r["condition"])
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64),
            np.asarray(ach, dtype=np.int64), np.asarray(rule_idx, np.int64),
            np.asarray(cond))


def load_cache(path):
    feats, corr, rule_idx, ch_id, ch_name, sev = [], [], [], [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            feats.append([float(r["b"]), float(r["c"]), float(r["e"])])
            t = int(r["gtsrb_true"])
            corr.append([1 if int(r[f"pred_{br}"]) == t else 0
                         for br in BRANCHES])
            rule_idx.append(BRANCHES.index(r["rule_branch"]))
            ch_id.append(int(r["ch_id"]))
            ch_name.append(r["ch_name"])
            sev.append(int(r["sev"]))
    return (np.asarray(feats, np.float64), np.asarray(corr, np.int64),
            np.asarray(rule_idx, np.int64), np.asarray(ch_id, np.int64),
            np.asarray(ch_name), np.asarray(sev, np.int64))


# ------------------------------------------------------------------
# numpy models (deterministic)
# ------------------------------------------------------------------
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class AdamState:
    def __init__(self, shapes, lr):
        self.lr = lr
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + eps))
        return out


class LogReg:
    name = "logreg"

    def __init__(self, d_in=3, d_out=4, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (d_in, d_out))
        self.b = np.zeros(d_out)

    def fit(self, X, y, epochs=300, lr=0.05):
        n = len(X)
        Y = np.eye(self.b.size)[y]
        adam = AdamState([self.W.shape, self.b.shape], lr)
        for _ in range(epochs):
            P = _softmax(X @ self.W + self.b)
            G = (P - Y) / n
            gW = X.T @ G
            gb = G.sum(axis=0)
            self.W, self.b = adam.step([self.W, self.b], [gW, gb])
        return self

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)


class MLP:
    name = "mlp"

    def __init__(self, d_in=3, hidden=16, d_out=4, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, d_out))
        self.b3 = np.zeros(d_out)

    def _forward(self, X):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h1, h2, h2 @ self.W3 + self.b3

    def fit(self, X, y, epochs=400, lr=0.01):
        n = len(X)
        Y = np.eye(self.b3.size)[y]
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = AdamState([p.shape for p in params], lr)
        for _ in range(epochs):
            h1, h2, z = self._forward(X)
            P = _softmax(z)
            G3 = (P - Y) / n
            gW3 = h2.T @ G3
            gb3 = G3.sum(axis=0)
            G2 = (G3 @ self.W3.T) * (h2 > 0)
            gW2 = h1.T @ G2
            gb2 = G2.sum(axis=0)
            G1 = (G2 @ self.W2.T) * (h1 > 0)
            gW1 = X.T @ G1
            gb1 = G1.sum(axis=0)
            params = adam.step(params, [gW1, gb1, gW2, gb2, gW3, gb3])
            (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3) = params
        return self

    def predict(self, X):
        _, _, z = self._forward(X)
        return np.argmax(z, axis=1)


# ------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------
def achieved_acc(choice_idx, corr):
    return 100.0 * corr[np.arange(len(choice_idx)), choice_idx].mean()


def cell_avg(choice_idx, corr, cells):
    """cells: array of hashable cell keys; returns mean over cells of acc."""
    ok = corr[np.arange(len(choice_idx)), choice_idx]
    accs = []
    for cell in np.unique(cells):
        m = cells == cell
        accs.append(100.0 * ok[m].mean())
    return float(np.mean(accs))


def branch_distribution(choice_idx, mask):
    n = mask.sum()
    return {br: round(100.0 * float((choice_idx[mask] == k).sum()) / n, 1)
            for k, br in enumerate(BRANCHES)}


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(TRAIN_CSV_DEFAULT))
    ap.add_argument("--cure-cache", default=str(CACHE_CSV_DEFAULT))
    ap.add_argument("--out", default=str(OUT_JSON_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs-lr", type=int, default=300)
    ap.add_argument("--epochs-mlp", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=16)
    args = ap.parse_args()

    # ---------------- training data ----------------
    X, y, ach, rule_idx, cond = load_train(args.train_csv)
    n = len(X)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    mu, sd = X[tr_i].mean(axis=0), X[tr_i].std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    print(f"[train] {len(tr_i)} rows train / {len(val_i)} rows val "
          f"(seed={args.seed})")
    print(f"[train] best-branch label distribution: "
          + "  ".join(f"{br}={100.0*(y==k).mean():.1f}%"
                      for k, br in enumerate(BRANCHES)))

    models = [
        LogReg(seed=args.seed).fit(Xs[tr_i], y[tr_i],
                                   epochs=args.epochs_lr, lr=0.05),
        MLP(hidden=args.hidden, seed=args.seed).fit(
            Xs[tr_i], y[tr_i], epochs=args.epochs_mlp, lr=0.01),
    ]

    print("\n=== synthetic GTSRB validation split ===")
    print(f"{'selector':10s} {'label-acc':>9s} {'achieved':>9s}")
    oracle_val = 100.0 * ach[val_i].max(axis=1).mean()
    rule_val = achieved_acc(rule_idx[val_i], ach[val_i])
    base_val = 100.0 * ach[val_i, 0].mean()
    val_results = {}
    for m in models:
        pred = m.predict(Xs[val_i])
        lab = 100.0 * (pred == y[val_i]).mean()
        got = achieved_acc(pred, ach[val_i])
        val_results[m.name] = {"label_acc": lab, "achieved": got}
        print(f"{m.name:10s} {lab:8.2f}% {got:8.2f}%")
    print(f"{'rule':10s} {'-':>9s} {rule_val:8.2f}%")
    print(f"{'baseline':10s} {'-':>9s} {base_val:8.2f}%")
    print(f"{'oracle4':10s} {'-':>9s} {oracle_val:8.2f}%")

    # ---------------- CURE deployment ----------------
    Xc, corr, rule_c, ch_id, ch_name, sev = load_cache(args.cure_cache)
    Xcs = (Xc - mu) / sd                      # TRAIN standardization
    deg = ch_id != 0
    cf = ch_id == 0
    cells = np.array([f"{c}_{s}" for c, s in zip(ch_id, sev)])

    print("\n=== CURE-TSR deployment (cell-averaged deg-avg, %) ===")
    print(f"{'selector':10s} {'deg-avg':>8s} {'CF acc':>8s}")
    base_idx = np.zeros(len(Xc), dtype=np.int64)
    rows_out = {}

    def report(name, idx):
        d = cell_avg(idx[deg], corr[deg], cells[deg])
        c = achieved_acc(idx[cf], corr[cf])
        rows_out[name] = {"deg_avg": round(d, 2), "cf_acc": round(c, 2),
                          "dist_deg": branch_distribution(idx, deg),
                          "dist_cf": branch_distribution(idx, cf)}
        print(f"{name:10s} {d:8.2f} {c:8.2f}")
        return d

    report("baseline", base_idx)
    report("rule", rule_c)
    for m in models:
        idx = m.predict(Xcs)
        report(m.name, idx)
    oracle_idx = np.argmax(corr, axis=1)      # any correct branch if exists
    d_or = cell_avg(oracle_idx[deg], corr[deg], cells[deg])
    print(f"{'oracle4':10s} {d_or:8.2f} {'-':>8s}")

    print("\nper-challenge deg-avg:")
    names = ["baseline", "rule"] + [m.name for m in models]
    idxs = [base_idx, rule_c] + [m.predict(Xcs) for m in models]
    hdr = f"{'challenge':12s}" + "".join(f"{nm:>10s}" for nm in names + ["oracle4"])
    print(hdr)
    for ch in sorted(set(ch_name[deg])):
        m_ch = deg & (ch_name == ch)
        line = f"{ch:12s}"
        for idx in idxs:
            line += f"{cell_avg(idx[m_ch], corr[m_ch], cells[m_ch]):10.1f}"
        line += f"{cell_avg(oracle_idx[m_ch], corr[m_ch], cells[m_ch]):10.1f}"
        print(line)

    print("\nrouting distribution on degraded CURE (learned selectors):")
    for m, idx in zip(models, idxs[2:]):
        dist = branch_distribution(idx, deg)
        print(f"  {m.name:8s} " + "  ".join(f"{k[:5]}={v:4.1f}%"
                                            for k, v in dist.items()))
    print("  (safety check) ChallengeFree distribution:")
    for m, idx in zip(models, idxs[2:]):
        dist = branch_distribution(idx, cf)
        print(f"  {m.name:8s} " + "  ".join(f"{k[:5]}={v:4.1f}%"
                                            for k, v in dist.items()))

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed, "val_frac": args.val_frac,
        "train_csv": str(args.train_csv), "cure_cache": str(args.cure_cache),
        "feature_standardization": {"mean": mu.tolist(), "std": sd.tolist()},
        "label_rule": "argmax ptrue over branches; ties -> earliest "
                      "(passthrough first)",
        "epochs": {"logreg": args.epochs_lr, "mlp": args.epochs_mlp},
        "hidden": args.hidden,
        "synthetic_val": {"oracle4": round(oracle_val, 2),
                          "rule": round(rule_val, 2),
                          "baseline": round(base_val, 2),
                          **{k: {kk: round(vv, 2) for kk, vv in v.items()}
                             for k, v in val_results.items()}},
        "cure_deployment": rows_out,
        "cure_oracle4_degavg": round(d_or, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
