"""
D_quick_noise_branch_sanity.py — v7

★ v7 关键修复（pipeline order）:
  v6 错: cv2.imread → cv2.resize(32x32) → enhance → torchvision transform
  v7 对: cv2.imread → enhance (at original size) → BGR2RGB → PIL → transforms.Resize → ToTensor → Normalize
       (与 evaluate_all.py 完全一致)

Calibration 也改成在原始尺寸上做 (不再 cv2.resize)
"""

import sys
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

sys.path.insert(0, r"D:\Project\traffic_sign\src")
from revision_utils import (
    OUTPUTS_DIR, INPUT_SIZE,
    PAPER_THRESHOLDS, init_run_config, finalize_run,
    load_gtsrb_compactcnn, compute_noise_stat,
    verify_data_paths,
    collect_gtsrb_test_samples,
    collect_gtsrb_train_samples_stratified,
    build_paper_test_transform, cv2_to_pil_rgb,
)
from enhance import adaptive_enhance

TESTSETS_FOR_SANITY = ["clean", "lowcontrast", "noisy"]
SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42
NOISE_PERCENTILE = 85


def _T():
    return (PAPER_THRESHOLDS["T1"], PAPER_THRESHOLDS["T2"],
            PAPER_THRESHOLDS["T3"], PAPER_THRESHOLDS["T4"])


def calibrate_noise_threshold(percentile=NOISE_PERCENTILE):
    """★ v7: 不再 cv2.resize，在原始尺寸上计算 noise_stat"""
    samples = collect_gtsrb_train_samples_stratified(
        samples_per_class=SAMPLES_PER_CLASS, seed=RANDOM_SEED)
    
    stats = []
    for path, _ in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        # ★ no resize — operate at original size
        stats.append(compute_noise_stat(img))
    
    if not stats:
        return 0.05, 0
    threshold = float(np.percentile(stats, percentile))
    print(f"  Stratified sampling: {len(stats)} images (original size), P{percentile} = {threshold:.4f}")
    return threshold, len(stats)


def adaptive_v1(img):
    return adaptive_enhance(img, *_T())


def adaptive_v2_median(img, T_noise):
    n = compute_noise_stat(img)
    if n > T_noise:
        img = cv2.medianBlur(img, 3)
    return adaptive_enhance(img, *_T())


def adaptive_v2_gaussian(img, T_noise):
    n = compute_noise_stat(img)
    if n > T_noise:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return adaptive_enhance(img, *_T())


# ============================================================
# Dataset (★ v7: paper-consistent pipeline)
# ============================================================
class TestsetDataset(Dataset):
    def __init__(self, testset_name, enhancement_fn, transform):
        self.fn = enhancement_fn
        self.transform = transform
        self.samples = collect_gtsrb_test_samples(testset_name)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path)            # original size
        if img_bgr is None:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), -1
        # ★ 1. enhance at ORIGINAL size
        img_bgr = self.fn(img_bgr)
        # ★ 2. BGR → PIL RGB
        pil_img = cv2_to_pil_rgb(img_bgr)
        # ★ 3. transform: PIL Resize(32x32) → ToTensor → Normalize
        x = self.transform(pil_img)
        return x, label


def evaluate(model, device, ds):
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return {
        "n": len(ds),
        "acc": accuracy_score(labels, preds) * 100,
        "macro_f1": f1_score(labels, preds, average='macro', zero_division=0) * 100,
    }


def measure_routing_trigger_rate(testset_name, T_noise, max_samples=500):
    """★ v7: 在原始尺寸上计算 noise_stat"""
    samples = collect_gtsrb_test_samples(testset_name)[:max_samples]
    stats = []
    triggered = 0
    for path, _ in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        # ★ no resize
        ns = compute_noise_stat(img)
        stats.append(ns)
        if ns > T_noise:
            triggered += 1
    n = len(stats)
    return {
        "n": n,
        "trigger_rate_pct": triggered / n * 100 if n > 0 else 0,
        "mean_noise_stat": float(np.mean(stats)) if stats else 0,
        "median_noise_stat": float(np.median(stats)) if stats else 0,
    }


def main():
    verify_data_paths()
    init_run_config("D_noise_branch_sanity", {
        "version": "v7 (paper-consistent pipeline)",
        "testsets": TESTSETS_FOR_SANITY,
        "noise_threshold_percentile": NOISE_PERCENTILE,
        "thresholds": PAPER_THRESHOLDS,
        "pipeline": "cv2.imread → enhance@original → BGR2RGB → PIL → Resize(32) → ToTensor → Normalize",
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_paper_test_transform()
    model = load_gtsrb_compactcnn(device)
    print(f"Model loaded.")
    
    print("\n" + "=" * 70)
    print("  Step 1: Calibrate noise threshold (at original sizes)")
    print("=" * 70)
    T_noise, n_calib = calibrate_noise_threshold()
    
    print("\n" + "=" * 70)
    print("  Step 2: Noise branch trigger rates")
    print("=" * 70)
    print(f"  Noise threshold = {T_noise:.4f}")
    
    trigger_rates = {}
    for ts in TESTSETS_FOR_SANITY:
        r = measure_routing_trigger_rate(ts, T_noise)
        trigger_rates[ts] = r
        print(f"  {ts:13s}: triggered={r['trigger_rate_pct']:5.1f}%  mean_stat={r['mean_noise_stat']:.4f}")
    
    print("\n" + "=" * 70)
    print("  Step 3: Evaluate 4 methods × 3 testsets")
    print("  (★ v7: paper-consistent pipeline, expect v1_paper to match paper's 91.66/65.10/67.36)")
    print("=" * 70)
    
    methods = {
        "no_prep":      lambda img: img,
        "v1_paper":     lambda img: adaptive_v1(img),
        "v2_median":    lambda img: adaptive_v2_median(img, T_noise),
        "v2_gaussian":  lambda img: adaptive_v2_gaussian(img, T_noise),
    }
    
    results = {}
    for ts in TESTSETS_FOR_SANITY:
        results[ts] = {}
        print(f"\n[Testset] {ts}")
        for method_name, fn in methods.items():
            ds = TestsetDataset(ts, fn, transform)
            r = evaluate(model, device, ds)
            if r is None:
                continue
            results[ts][method_name] = r
            print(f"  {method_name:13s} acc={r['acc']:5.2f}%  f1={r['macro_f1']:5.2f}%  n={r['n']}")
    
    print("\n" + "=" * 70)
    print("  Step 4: 自动决策建议")
    print("=" * 70)
    
    decision_lines = []
    for variant in ["v2_median", "v2_gaussian"]:
        d_clean = (results.get("clean", {}).get(variant, {}).get("acc", 0)
                   - results.get("clean", {}).get("v1_paper", {}).get("acc", 0))
        d_lc    = (results.get("lowcontrast", {}).get(variant, {}).get("acc", 0)
                   - results.get("lowcontrast", {}).get("v1_paper", {}).get("acc", 0))
        d_noisy = (results.get("noisy", {}).get(variant, {}).get("acc", 0)
                   - results.get("noisy", {}).get("v1_paper", {}).get("acc", 0))
        decision_lines.append(f"\n--- {variant} (vs v1_paper) ---")
        decision_lines.append(f"  Δ clean       = {d_clean:+.2f} pp")
        decision_lines.append(f"  Δ lowcontrast = {d_lc:+.2f} pp")
        decision_lines.append(f"  Δ noisy       = {d_noisy:+.2f} pp")
        if d_noisy > 5 and d_clean > -1.0 and d_lc > -1.0:
            verdict = "✓ 推荐用作主方法"
        elif d_noisy > 2 and d_clean > -2.0 and d_lc > -2.0:
            verdict = "○ 仅作 ablation"
        elif d_noisy < 2:
            verdict = "✗ 收益不足"
        else:
            verdict = "✗ 伤害其他 testset"
        decision_lines.append(f"  Verdict: {verdict}")
    
    decision_lines.append("\n" + "=" * 50)
    decision_lines.append("  vs no_prep")
    decision_lines.append("=" * 50)
    for variant in ["v1_paper", "v2_median", "v2_gaussian"]:
        d_n = (results.get("noisy", {}).get(variant, {}).get("acc", 0)
               - results.get("noisy", {}).get("no_prep", {}).get("acc", 0))
        d_c = (results.get("clean", {}).get(variant, {}).get("acc", 0)
               - results.get("clean", {}).get("no_prep", {}).get("acc", 0))
        decision_lines.append(f"  {variant:13s}: noisy {d_n:+.2f}, clean {d_c:+.2f}")
    
    final_lines = ["\n" + "=" * 50, "  最终推荐", "=" * 50]
    median_d_noisy = (results.get("noisy", {}).get("v2_median", {}).get("acc", 0)
                      - results.get("noisy", {}).get("v1_paper", {}).get("acc", 0))
    gauss_d_noisy = (results.get("noisy", {}).get("v2_gaussian", {}).get("acc", 0)
                     - results.get("noisy", {}).get("v1_paper", {}).get("acc", 0))
    median_d_clean = (results.get("clean", {}).get("v2_median", {}).get("acc", 0)
                      - results.get("clean", {}).get("v1_paper", {}).get("acc", 0))
    gauss_d_clean = (results.get("clean", {}).get("v2_gaussian", {}).get("acc", 0)
                     - results.get("clean", {}).get("v1_paper", {}).get("acc", 0))
    
    if max(median_d_noisy, gauss_d_noisy) < 2:
        final_lines.append("  → 两个 variant 在 noisy 上都未显著提升")
        final_lines.append("  → 推荐：不加 noise branch，作为 limitation")
    elif median_d_noisy > gauss_d_noisy and median_d_clean >= gauss_d_clean - 0.5:
        final_lines.append(f"  → Median 3×3 (Δnoisy=+{median_d_noisy:.2f}pp, Δclean={median_d_clean:+.2f}pp)")
        final_lines.append("  → 推荐 Median 3×3")
    elif gauss_d_noisy > median_d_noisy + 1.0:
        final_lines.append(f"  → Gaussian (Δnoisy=+{gauss_d_noisy:.2f}pp)")
        final_lines.append("  → 推荐 Gaussian 3×3")
    else:
        final_lines.append("  → 两者接近，按 latency 选轻的（Median）")
    decision_lines += final_lines
    
    csv_path = OUTPUTS_DIR / "D_noise_branch_sanity.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["=== Noise Branch Sanity Check (v7) ==="])
        w.writerow(["pipeline", "paper-consistent (enhance@original size, PIL resize)"])
        w.writerow(["noise_threshold", T_noise])
        w.writerow(["calibration_samples", n_calib])
        w.writerow(["random_seed", RANDOM_SEED])
        w.writerow([])
        w.writerow(["=== Trigger rates ==="])
        w.writerow(["testset", "trigger_rate_pct", "mean_noise_stat", "median_noise_stat"])
        for ts, r in trigger_rates.items():
            w.writerow([ts, f"{r['trigger_rate_pct']:.2f}",
                        f"{r['mean_noise_stat']:.4f}", f"{r['median_noise_stat']:.4f}"])
        w.writerow([])
        w.writerow(["=== Accuracy ==="])
        w.writerow(["testset", "method", "n", "accuracy", "macro_f1"])
        for ts, by_m in results.items():
            for m, r in by_m.items():
                w.writerow([ts, m, r["n"], f"{r['acc']:.4f}", f"{r['macro_f1']:.4f}"])
    print(f"\n[✓] {csv_path}")
    
    txt_path = OUTPUTS_DIR / "D_noise_branch_decision.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  Noise Branch Sanity Decision (v7 paper-consistent pipeline)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Pipeline: cv2.imread → enhance(orig) → BGR2RGB → PIL → Resize(32) → tensor\n")
        f.write(f"Noise threshold: {T_noise:.4f} (from {n_calib} stratified train images at original size)\n\n")
        f.write("Trigger rates:\n")
        for ts, r in trigger_rates.items():
            f.write(f"  {ts:13s}: {r['trigger_rate_pct']:5.1f}% triggered\n")
        f.write("\nAccuracy (expect v1_paper to match paper's reported numbers):\n")
        f.write("  Paper reference: clean=91.66%, lowcontrast=65.10%, noisy=67.36%\n\n")
        for ts, by_m in results.items():
            f.write(f"  [{ts}]\n")
            for m, r in by_m.items():
                f.write(f"    {m:13s}: acc={r['acc']:.2f}%  f1={r['macro_f1']:.2f}%\n")
        f.write("\n".join(decision_lines))
    print(f"[✓] {txt_path}")
    
    for line in decision_lines:
        print(line)
    
    finalize_run("D_noise_branch_sanity",
                 f"v7 paper-consistent pipeline\n"
                 f"T_noise={T_noise:.4f}\n"
                 f"v1_paper: clean={results.get('clean', {}).get('v1_paper', {}).get('acc', 0):.2f}%, "
                 f"lc={results.get('lowcontrast', {}).get('v1_paper', {}).get('acc', 0):.2f}%, "
                 f"noisy={results.get('noisy', {}).get('v1_paper', {}).get('acc', 0):.2f}%\n"
                 f"Median Δnoisy={median_d_noisy:+.2f}pp\n"
                 f"Gaussian Δnoisy={gauss_d_noisy:+.2f}pp")


if __name__ == "__main__":
    main()
