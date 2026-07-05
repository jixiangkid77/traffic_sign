"""
evaluate_cure_tsr_external.py — CURE-TSR external validation（路线 A）

核心原则：
  - 零训练（不在 CURE-TSR 上训练任何东西）
  - 零再校准（GTSRB 训练时校的阈值不动）
  - 只评估 high-confidence mapped subset

这种设计直接回应评审 Comment 2:
  "Real degraded traffic sign images should be added to verify
   practical effectiveness in ADAS scenarios."

且回应得最强 — paper 可以写：
  "No retraining or fine-tuning is performed on CURE-TSR; thresholds
   are not re-tuned. Our GTSRB-trained model is applied directly."

5 个对比方法：
  1. baseline (no preprocessing)
  2. fixed CLAHE
  3. fixed gamma
  4. fixed linear stretch  ← 新增（回应"comparisons incomplete"）
  5. VA-Adaptive (ours)

输出：
  outputs_cure_tsr/
    cure_tsr_main_results.json
    cure_tsr_main_results.csv
    cure_tsr_intensity_curves.png
    cure_tsr_routing_stats.json
    cure_tsr_eval_log.txt

Sanity check:
  脚本第一步会先在 ChallengeFree 上跑 baseline，如果 acc < 50%，
  自动停下并提醒：你的 GTSRB 模型与 CURE-TSR 域差距太大，需要
  重新校准映射或换路线 B。

运行：
  conda activate pcm_sim
  python src/evaluate_cure_tsr_external.py

依赖：
  src/model.py 里的 CompactCNN
  src/enhance.py 里的 apply_clahe / apply_gamma / adaptive_enhance
  models/mbnetv3_baseline.pth (你的 GTSRB-trained 模型)
"""

import os
import sys
import json
import csv
import re
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# 引入 paper 里的 model 和 enhance 实现
sys.path.insert(0, r"D:\Project\traffic_sign\src")
from model import CompactCNN  # noqa
from enhance import apply_clahe, apply_gamma, adaptive_enhance  # noqa
# 学习型增强器: 复用 evaluate_learned_baselines 里已修好的实现 (FFA 输入归一化已包含)
from evaluate_learned_baselines import (  # noqa
    ZeroDCEEnhancer, FFANetEnhancer, ZERO_DCE_WEIGHTS, FFA_NET_WEIGHTS,
)

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
CURE_TSR_DIR = PROJECT_ROOT / "datasets" / "CURE-TSR"
OUTPUT_DIR = PROJECT_ROOT / "outputs_cure_tsr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GTSRB_MODEL_PATH = PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"

INPUT_SIZE = 32
GTSRB_MEAN = [0.3401, 0.3120, 0.3212]
GTSRB_STD  = [0.2725, 0.2609, 0.2669]

# CURE-TSR filename pattern (5 段)
FILENAME_PATTERN = re.compile(
    r'(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d+)\.(bmp|png|jpg|jpeg)',
    re.IGNORECASE
)

CURE_CLASSES = {
    1: "speed_limit",   2: "goods_vehicles", 3: "no_overtaking",
    4: "no_stopping",   5: "no_parking",     6: "stop",
    7: "bicycle",       8: "hump",           9: "no_left",
    10:"no_right",     11: "priority_to",   12: "no_entry",
    13:"yield",        14: "parking",
}

CHALLENGE_TYPES = {
    0:  "ChallengeFree",   1: "Decolorization",  2: "LensBlur",
    3:  "CodecError",      4: "Darkening",       5: "DirtyLens",
    6:  "Exposure",        7: "GaussianBlur",    8: "Noise",
    9:  "Rain",           10: "Shadow",         11: "Snow",
    12: "Haze",
}

# ============================================================
# 关键: CURE → GTSRB 映射
# 默认仅用 high-confidence 子集 (4 类一对一映射)
# 若你目测确认其他类也可加入，修改这里即可
# ============================================================
DEFAULT_CURE_TO_GTSRB = {
    # CURE_id : GTSRB_class_id (单值即一对一)
    3:  9,   # no_overtaking
    6:  14,  # stop
    11: 12,  # priority_to
    12: 17,  # no_entry
    13: 13,  # yield
}

# Paper-relevant challenges
EVAL_CHALLENGES = [4, 8, 9, 11, 12]  # Darkening, Noise, Rain, Snow, Haze
EVAL_SEVERITIES = [1, 2, 3, 4, 5]


# ============================================================
# Dataset
# ============================================================
class CureTSRMappedDataset(Dataset):
    """
    加载 CURE-TSR 真实数据 (sequenceType=01)，
    过滤到 mapped subset，
    可选应用 enhancement。
    """
    def __init__(self, root, mapping, challenge_type=None, severity=None,
                 enhancement_fn=None, transform=None):
        self.mapping = mapping
        self.enhancement_fn = enhancement_fn
        self.transform = transform

        self.samples = []
        for fpath in root.rglob('*.bmp'):
            m = FILENAME_PATTERN.match(fpath.name)
            if not m:
                continue
            seq = int(m.group(1))
            sign = int(m.group(2))
            ch = int(m.group(3))
            sev = int(m.group(4))

            # 过滤
            if seq != 1:                  # 仅 real
                continue
            if sign not in mapping:        # 仅 mapped subset
                continue
            if challenge_type is not None and ch != challenge_type:
                continue
            if severity is not None and sev != severity:
                continue

            self.samples.append({
                "path": fpath, "sign": sign, "ch": ch, "sev": sev
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(s["path"]))
        if img is None:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), -1

        if self.enhancement_fn is not None:
            img = self.enhancement_fn(img)        # 原始尺寸增强, 与 GTSRB 主管线一致

        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_t = self.transform(img_rgb)
        else:
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        target_gtsrb = self.mapping[s["sign"]]
        return img_t, target_gtsrb


# ============================================================
# Enhancement functions
# ============================================================
def fn_baseline(img):
    return img

def fn_fixed_clahe(img):
    return apply_clahe(img, clip_limit=3.0)

def fn_fixed_gamma(img):
    return apply_gamma(img, gamma=0.5)

def fn_fixed_stretch(img):
    """新增: fixed linear stretch baseline"""
    f = img.astype(np.float32) / 255.0
    out = np.clip((f - 0.5) * 1.5 + 0.5, 0, 1)
    return (out * 255).astype(np.uint8)

def fn_va_adaptive(img):
    t = GTSRB_THRESHOLDS
    return adaptive_enhance(img, t["T1"], t["T2"], t["T3"], t["T4"])

METHODS = {
    "baseline":      fn_baseline,
    "fixed_clahe":   fn_fixed_clahe,
    "fixed_gamma":   fn_fixed_gamma,
    "fixed_stretch": fn_fixed_stretch,
    "va_adaptive":   fn_va_adaptive,
}


# ============================================================
# Routing 统计 (只对 va_adaptive 用)
# ============================================================
def compute_stats(img_bgr):
    if img_bgr.dtype == np.uint8:
        f = img_bgr.astype(np.float32) / 255.0
    else:
        f = img_bgr
    gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gn = gray.astype(np.float32) / 255.0
    b = float(np.mean(gn))
    c = float(2 * np.std(gn))
    edges = cv2.Canny(gray, 50, 150)
    e = float(np.mean(edges > 0))
    return b, c, e

def route_decision(b, c, e, T1, T2, T3, T4):
    if b < T1: return "gamma"
    elif c < T2: return "clahe"
    elif e < T3 and b > T4: return "stretch"
    else: return "passthrough"

GTSRB_THRESHOLDS = {"T1": 0.1206, "T2": 0.1061, "T3": 0.0726, "T4": 0.4085}


# ============================================================
# 模型加载
# ============================================================
def load_gtsrb_model(device):
    print(f"\n[Load] GTSRB-trained CompactCNN from {GTSRB_MODEL_PATH}")
    if not GTSRB_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {GTSRB_MODEL_PATH}")

    model = CompactCNN(num_classes=43)
    ckpt = torch.load(GTSRB_MODEL_PATH, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Params: {n_params:,}")
    return model


# ============================================================
# 单次评估
# ============================================================
def evaluate(model, device, dataset, batch_size=64):
    if len(dataset) == 0:
        return {"acc": float('nan'), "macro_f1": float('nan'), "n": 0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds) * 100
    # macro-F1 只在 5 个映射目标类上算; 否则模型在 43 个 GTSRB 类上的杂散预测会把它摊薄
    macro_f1 = f1_score(labels, preds, average='macro',
                        labels=sorted(set(DEFAULT_CURE_TO_GTSRB.values())),
                        zero_division=0) * 100
    return {"acc": acc, "macro_f1": macro_f1, "n": len(dataset)}


# ============================================================
# Sanity check (在 ChallengeFree 上跑 baseline)
# ============================================================
def sanity_check(model, device, transform):
    print("\n" + "=" * 70)
    print("  Sanity Check: GTSRB model on CURE-TSR ChallengeFree (real)")
    print("=" * 70)

    ds = CureTSRMappedDataset(
        root=CURE_TSR_DIR,
        mapping=DEFAULT_CURE_TO_GTSRB,
        challenge_type=0,  # ChallengeFree
        enhancement_fn=None,
        transform=transform,
    )
    print(f"  Mapped subset (ChallengeFree real): {len(ds)} images")
    if len(ds) == 0:
        print(f"  ✗ 没有可评估样本。检查映射或数据是否完整。")
        return False, 0.0

    r = evaluate(model, device, ds)
    print(f"  Baseline acc on CURE-TSR ChallengeFree: {r['acc']:.2f}%")
    print(f"  Macro-F1: {r['macro_f1']:.2f}%")

    if r['acc'] < 50.0:
        print(f"\n  ⚠ Acc < 50% — GTSRB ↔ CURE-TSR 域差距太大")
        print(f"  → 路线 A 不可行")
        print(f"  → 选项 1: 重新检查映射 (sample_thumbnails.png 上确认)")
        print(f"  → 选项 2: 降级到路线 B (qualitative-only)")
        return False, r['acc']
    elif r['acc'] < 70.0:
        print(f"\n  ⚠ Acc 中等 ({r['acc']:.1f}%) — 路线 A 可行但 paper 要解释域差异")
        return True, r['acc']
    else:
        print(f"\n  ✓ Acc {r['acc']:.1f}% — 路线 A 完全可行")
        return True, r['acc']


# ============================================================
# 完整评估
# ============================================================
def run_full_evaluation(model, device, transform):
    print("\n" + "=" * 70)
    print("  Full Evaluation")
    print("=" * 70)

    all_results = {
        "config": {
            "mapping": DEFAULT_CURE_TO_GTSRB,
            "mapped_class_names": [CURE_CLASSES[c] for c in DEFAULT_CURE_TO_GTSRB],
            "n_mapped_classes": len(DEFAULT_CURE_TO_GTSRB),
            "challenges": EVAL_CHALLENGES,
            "challenge_names": [CHALLENGE_TYPES[c] for c in EVAL_CHALLENGES],
            "severities": EVAL_SEVERITIES,
            "methods": list(METHODS.keys()),
            "thresholds": GTSRB_THRESHOLDS,
        },
        "challengefree": {},
        "challenges": {},
    }

    # ChallengeFree (clean reference)
    print("\n[Reference] ChallengeFree:")
    for method_name, fn in METHODS.items():
        ds = CureTSRMappedDataset(
            root=CURE_TSR_DIR, mapping=DEFAULT_CURE_TO_GTSRB,
            challenge_type=0, enhancement_fn=fn, transform=transform,
        )
        r = evaluate(model, device, ds)
        all_results["challengefree"][method_name] = r
        print(f"  {method_name:15s} acc={r['acc']:5.2f}%  f1={r['macro_f1']:5.2f}%  n={r['n']}")

    # 每个 challenge × severity
    for ch_id in EVAL_CHALLENGES:
        ch_name = CHALLENGE_TYPES[ch_id]
        print(f"\n[Challenge {ch_id:02d}] {ch_name}:")
        all_results["challenges"][ch_name] = {}

        for sev in EVAL_SEVERITIES:
            line = f"  Sev {sev}: "
            sev_results = {}
            for method_name, fn in METHODS.items():
                ds = CureTSRMappedDataset(
                    root=CURE_TSR_DIR, mapping=DEFAULT_CURE_TO_GTSRB,
                    challenge_type=ch_id, severity=sev,
                    enhancement_fn=fn, transform=transform,
                )
                r = evaluate(model, device, ds)
                sev_results[method_name] = r
                line += f"{method_name[:6]}={r['acc']:.1f}  "
            all_results["challenges"][ch_name][sev] = sev_results
            print(line)

    return all_results


# ============================================================
# Routing 分布统计 (仅 va_adaptive)
# ============================================================
def compute_routing_stats():
    print("\n" + "=" * 70)
    print("  Routing Distribution (VA-Adaptive only)")
    print("=" * 70)

    routing = {}
    T = GTSRB_THRESHOLDS
    for ch_id in [0] + EVAL_CHALLENGES:
        ch_name = CHALLENGE_TYPES[ch_id]
        counts = {"gamma": 0, "clahe": 0, "stretch": 0, "passthrough": 0}
        n_total = 0

        for fpath in CURE_TSR_DIR.rglob('*.bmp'):
            m = FILENAME_PATTERN.match(fpath.name)
            if not m: continue
            if int(m.group(1)) != 1: continue        # real only
            if int(m.group(2)) not in DEFAULT_CURE_TO_GTSRB: continue
            if int(m.group(3)) != ch_id: continue

            img = cv2.imread(str(fpath))
            if img is None: continue
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            b, c, e = compute_stats(img)
            br = route_decision(b, c, e, T["T1"], T["T2"], T["T3"], T["T4"])
            counts[br] += 1
            n_total += 1

        if n_total == 0:
            continue
        pct = {k: v / n_total * 100 for k, v in counts.items()}
        routing[ch_name] = {"counts": counts, "pct": pct, "total": n_total}

        print(f"  {ch_name:18s} n={n_total:>5}: ", end="")
        for br, p in sorted(pct.items(), key=lambda x: -x[1]):
            print(f"{br[:4]}={p:.0f}% ", end="")
        print()

    out = OUTPUT_DIR / "cure_tsr_routing_stats.json"
    with open(out, 'w') as f:
        json.dump(routing, f, indent=2)
    print(f"\n  [✓] {out}")
    return routing


# ============================================================
# 输出 CSV + 强度曲线
# ============================================================
def write_csv(all_results):
    out = OUTPUT_DIR / "cure_tsr_main_results.csv"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["challenge", "severity", "method", "acc", "macro_f1", "n"])
        for method, r in all_results["challengefree"].items():
            w.writerow(["ChallengeFree", 0, method, r["acc"], r["macro_f1"], r["n"]])
        for ch_name, by_sev in all_results["challenges"].items():
            for sev, by_method in by_sev.items():
                for method, r in by_method.items():
                    w.writerow([ch_name, sev, method, r["acc"], r["macro_f1"], r["n"]])
    print(f"\n  [✓] {out}")


def plot_intensity_curves(all_results):
    methods = list(METHODS.keys())
    colors = ['#666', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#2ca02c', '#8c564b']

    n_chal = len(EVAL_CHALLENGES)
    fig, axes = plt.subplots(1, n_chal, figsize=(n_chal * 3.5, 3.8))
    if n_chal == 1:
        axes = [axes]

    for ax, ch_id in zip(axes, EVAL_CHALLENGES):
        ch_name = CHALLENGE_TYPES[ch_id]
        if ch_name not in all_results["challenges"]:
            continue

        for method, color in zip(methods, colors):
            xs, ys = [], []
            for sev in EVAL_SEVERITIES:
                if sev in all_results["challenges"][ch_name]:
                    r = all_results["challenges"][ch_name][sev][method]
                    if not np.isnan(r["acc"]):
                        xs.append(sev)
                        ys.append(r["acc"])

            ax.plot(xs, ys, '-o', color=color, label=method,
                    linewidth=1.5, markersize=5)

        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(ch_name)
        ax.grid(alpha=0.3)
        ax.set_xticks(EVAL_SEVERITIES)
        if ch_id == EVAL_CHALLENGES[0]:
            ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "cure_tsr_intensity_curves.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [✓] {out}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"\nCURE-TSR External Evaluation (Path A)")
    print(f"Mapping (CURE → GTSRB): {DEFAULT_CURE_TO_GTSRB}")
    print(f"Mapped classes: {[CURE_CLASSES[c] for c in DEFAULT_CURE_TO_GTSRB]}")

    if not CURE_TSR_DIR.exists():
        print(f"\n[!] CURE-TSR 数据未找到: {CURE_TSR_DIR}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])

    model = load_gtsrb_model(device)

    # Sanity check
    ok, sc_acc = sanity_check(model, device, transform)
    if not ok:
        print(f"\n[STOP] Sanity check failed (acc={sc_acc:.2f}%)")
        print(f"建议：1) 用 sample_thumbnails.png 重新审查映射")
        print(f"      2) 降级到路线 B (evaluate_cure_tsr_qualitative.py)")
        return

    # 接入学习型基线 (sanity check 通过后再加载, 各自仅在权重存在时), 与 GTSRB 实验同一套增强器
    if ZERO_DCE_WEIGHTS.exists():
        METHODS["zero_dce"] = ZeroDCEEnhancer(ZERO_DCE_WEIGHTS, device)
        print(f"  + learned baseline: zero_dce ({ZERO_DCE_WEIGHTS.name})")
    else:
        print(f"  [skip] 未找到 {ZERO_DCE_WEIGHTS.name}, 跳过 zero_dce")
    if FFA_NET_WEIGHTS.exists():
        METHODS["ffa_net"] = FFANetEnhancer(FFA_NET_WEIGHTS, device)
        print(f"  + learned baseline: ffa_net ({FFA_NET_WEIGHTS.name})")
    else:
        print(f"  [skip] 未找到 {FFA_NET_WEIGHTS.name}, 跳过 ffa_net")

    # Full evaluation
    t0 = time.time()
    results = run_full_evaluation(model, device, transform)
    t1 = time.time()
    print(f"\nFull evaluation: {(t1-t0)/60:.1f} min")

    # Routing stats
    routing = compute_routing_stats()

    # 写 JSON
    out_json = OUTPUT_DIR / "cure_tsr_main_results.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  [✓] {out_json}")

    # 写 CSV
    write_csv(results)

    # 画曲线
    plot_intensity_curves(results)

    print("\n" + "=" * 70)
    print("  CURE-TSR external evaluation 完成")
    print("=" * 70)
    print(f"\n  关键摘要 (写进 paper Section IV.F):")
    print(f"  - GTSRB-trained CompactCNN, no retraining")
    print(f"  - {len(DEFAULT_CURE_TO_GTSRB)} CURE classes mapped to GTSRB:")
    for c in DEFAULT_CURE_TO_GTSRB:
        print(f"      CURE {c} ({CURE_CLASSES[c]}) → GTSRB {DEFAULT_CURE_TO_GTSRB[c]}")
    print(f"  - {len(EVAL_CHALLENGES)} challenges × {len(EVAL_SEVERITIES)} severities × {len(METHODS)} methods")
    print(f"  - Avg-degraded acc per method (across all challenges & severities):")

    method_avgs = defaultdict(list)
    for ch_name, by_sev in results["challenges"].items():
        for sev, by_method in by_sev.items():
            for method, r in by_method.items():
                if not np.isnan(r["acc"]):
                    method_avgs[method].append(r["acc"])

    print(f"\n  {'method':<15} {'avg-deg-acc':>12} {'clean':>10}")
    for method in METHODS:
        if method_avgs[method]:
            avg = np.mean(method_avgs[method])
            clean = results["challengefree"][method]["acc"]
            marker = " ←" if method == "va_adaptive" else ""
            print(f"  {method:<15} {avg:>10.2f}%   {clean:>8.2f}%{marker}")


if __name__ == "__main__":
    main()
