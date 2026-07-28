"""
B_threshold_sensitivity.py — v5

★ v5 关键修复:
  Pipeline: cv2.imread → enhance@original → BGR2RGB → PIL → Resize(32) → ToTensor → Normalize
  Calibration: 在原始尺寸上计算 b/c/e (与 paper 阈值校准方式一致)
"""

import sys
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, r"D:\Project\traffic_sign\src")
from revision_utils import (
    OUTPUTS_DIR, INPUT_SIZE,
    PAPER_THRESHOLDS, init_run_config, finalize_run,
    load_gtsrb_compactcnn, compute_image_stats,
    verify_data_paths, collect_gtsrb_test_samples,
    collect_gtsrb_train_samples_stratified,
    build_paper_test_transform, cv2_to_pil_rgb,
)
from enhance import adaptive_enhance

DEGRADED_TESTSETS = ["lowlight", "foggy", "lowcontrast", "noisy", "mixed"]
T_LOW_PCTS = [10, 15, 20]
T_HIGH_PCTS = [60, 70, 80]
SAMPLES_PER_CLASS = 50
RANDOM_SEED = 42


def calibrate_from_train(t_low_pct, t_high_pct):
    """★ v5: 在原始尺寸上计算 stats（不再 cv2.resize）"""
    samples = collect_gtsrb_train_samples_stratified(
        samples_per_class=SAMPLES_PER_CLASS, seed=RANDOM_SEED)
    
    bs, cs, es = [], [], []
    for path, _ in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        # ★ no resize
        b, c, e = compute_image_stats(img)
        bs.append(b); cs.append(c); es.append(e)
    
    if not bs:
        raise RuntimeError("No images processed")
    
    return {
        "T1": float(np.percentile(bs, t_low_pct)),
        "T2": float(np.percentile(cs, t_low_pct)),
        "T3": float(np.percentile(es, t_low_pct)),
        "T4": float(np.percentile(bs, t_high_pct)),
    }, len(bs)


class TestsetDataset(Dataset):
    def __init__(self, testset_name, T, transform):
        self.T = T
        self.transform = transform
        self.samples = collect_gtsrb_test_samples(testset_name)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), -1
        # ★ enhance at ORIGINAL size with calibrated thresholds
        img_bgr = adaptive_enhance(img_bgr, self.T["T1"], self.T["T2"], self.T["T3"], self.T["T4"])
        pil_img = cv2_to_pil_rgb(img_bgr)
        x = self.transform(pil_img)
        return x, label


def evaluate_testset(model, device, testset_name, T, transform):
    ds = TestsetDataset(testset_name, T, transform)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds) * 100


def main():
    verify_data_paths()
    init_run_config("B_threshold_sensitivity", {
        "version": "v5 (paper-consistent pipeline)",
        "T_low_percentiles": T_LOW_PCTS,
        "T_high_percentiles": T_HIGH_PCTS,
        "testsets": DEGRADED_TESTSETS,
        "default_thresholds_paper": PAPER_THRESHOLDS,
        "calibration": {"method": "stratified_sampling@original_size",
                        "samples_per_class": SAMPLES_PER_CLASS,
                        "n_classes": 43, "seed": RANDOM_SEED},
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_paper_test_transform()
    model = load_gtsrb_compactcnn(device)
    print(f"Model loaded.")
    
    print("\n" + "=" * 70)
    print(f"  Threshold sensitivity (v5 paper-consistent): 9 cells × 5 testsets")
    print("=" * 70)
    
    cell_results = {}
    for t_low in T_LOW_PCTS:
        for t_high in T_HIGH_PCTS:
            print(f"\n[Cell] T_low={t_low}, T_high={t_high}")
            T, n_train = calibrate_from_train(t_low, t_high)
            print(f"  T1={T['T1']:.4f}, T2={T['T2']:.4f}, T3={T['T3']:.4f}, T4={T['T4']:.4f}")
            
            per_testset = {}
            for ts in DEGRADED_TESTSETS:
                acc = evaluate_testset(model, device, ts, T, transform)
                if acc is None:
                    continue
                per_testset[ts] = acc
                print(f"  {ts:13s} {acc:.2f}%")
            
            avg = float(np.mean(list(per_testset.values()))) if per_testset else float('nan')
            cell_results[(t_low, t_high)] = {
                "thresholds": T, "per_testset": per_testset, "avg_degraded": avg,
            }
            print(f"  → AVG = {avg:.2f}%")
    
    csv_path = OUTPUTS_DIR / "B_threshold_sensitivity.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["=== Avg-Degraded Accuracy (%) ==="])
        w.writerow(["T_low \\ T_high"] + [f"P{p}" for p in T_HIGH_PCTS])
        for t_low in T_LOW_PCTS:
            row = [f"P{t_low}"]
            for t_high in T_HIGH_PCTS:
                r = cell_results.get((t_low, t_high))
                row.append(f"{r['avg_degraded']:.2f}" if r else "—")
            w.writerow(row)
        w.writerow([])
        all_avgs = [r["avg_degraded"] for r in cell_results.values() if not np.isnan(r["avg_degraded"])]
        if all_avgs:
            w.writerow(["std", f"{np.std(all_avgs):.4f}"])
            w.writerow(["range", f"{max(all_avgs) - min(all_avgs):.4f}"])
        w.writerow([])
        w.writerow(["=== Detailed ==="])
        w.writerow(["T_low", "T_high", "T1", "T2", "T3", "T4"] + DEGRADED_TESTSETS + ["avg"])
        for (t_low, t_high), r in cell_results.items():
            row = [t_low, t_high,
                   f"{r['thresholds']['T1']:.4f}", f"{r['thresholds']['T2']:.4f}",
                   f"{r['thresholds']['T3']:.4f}", f"{r['thresholds']['T4']:.4f}"]
            for ts in DEGRADED_TESTSETS:
                row.append(f"{r['per_testset'].get(ts, float('nan')):.2f}")
            row.append(f"{r['avg_degraded']:.2f}")
            w.writerow(row)
    print(f"\n[✓] {csv_path}")
    
    if cell_results:
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.zeros((len(T_LOW_PCTS), len(T_HIGH_PCTS)))
        for i, t_low in enumerate(T_LOW_PCTS):
            for j, t_high in enumerate(T_HIGH_PCTS):
                r = cell_results.get((t_low, t_high))
                matrix[i, j] = r["avg_degraded"] if r else float('nan')
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(T_HIGH_PCTS)))
        ax.set_xticklabels([f"P{p}" for p in T_HIGH_PCTS])
        ax.set_yticks(range(len(T_LOW_PCTS)))
        ax.set_yticklabels([f"P{p}" for p in T_LOW_PCTS])
        ax.set_xlabel("T4 percentile")
        ax.set_ylabel("T1/T2/T3 percentile")
        ax.set_title("Avg-Degraded Accuracy (%)")
        for i in range(len(T_LOW_PCTS)):
            for j in range(len(T_HIGH_PCTS)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha='center', va='center',
                            color='black', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        out_png = OUTPUTS_DIR / "figures" / "B_threshold_sensitivity.png"
        plt.savefig(out_png, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"[✓] {out_png}")
    
    summary = ["B. Threshold sensitivity:"]
    if all_avgs:
        summary.append(f"  Range: [{min(all_avgs):.2f}, {max(all_avgs):.2f}]")
        summary.append(f"  Std: {np.std(all_avgs):.4f}")
    finalize_run("B_threshold_sensitivity", "\n".join(summary))


if __name__ == "__main__":
    main()
