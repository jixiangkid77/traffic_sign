"""
A_evaluate_with_metrics.py — v5

★ v5 关键修复（pipeline order）:
  v4 错: cv2.imread → cv2.resize(32x32) → enhance → torchvision transform
  v5 对: cv2.imread → enhance@original → BGR2RGB → PIL → Resize(32) → ToTensor → Normalize
       (与 evaluate_all.py 完全一致)
"""

import sys
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, r"D:\Project\traffic_sign\src")
from revision_utils import (
    OUTPUTS_DIR, INPUT_SIZE,
    PAPER_THRESHOLDS, PAPER_CLEAN_BASELINE_ACC,
    init_run_config, finalize_run, load_gtsrb_compactcnn,
    per_class_accuracy, verify_data_paths,
    collect_gtsrb_test_samples,
    build_paper_test_transform, cv2_to_pil_rgb,
)
from enhance import apply_clahe, apply_gamma, adaptive_enhance

TESTSETS = ["clean", "lowlight", "foggy", "lowcontrast", "noisy", "mixed"]


def _T():
    return (PAPER_THRESHOLDS["T1"], PAPER_THRESHOLDS["T2"],
            PAPER_THRESHOLDS["T3"], PAPER_THRESHOLDS["T4"])


def fn_baseline(img):
    return img

def fn_fixed_clahe(img):
    return apply_clahe(img, clip_limit=3.0)

def fn_fixed_gamma(img):
    return apply_gamma(img, gamma=0.5)

def fn_fixed_stretch(img):
    f = img.astype(np.float32) / 255.0
    out = np.clip((f - 0.5) * 1.5 + 0.5, 0, 1)
    return (out * 255).astype(np.uint8)

def fn_va_adaptive(img):
    return adaptive_enhance(img, *_T())

METHODS = {
    "baseline":      fn_baseline,
    "fixed_clahe":   fn_fixed_clahe,
    "fixed_gamma":   fn_fixed_gamma,
    "fixed_stretch": fn_fixed_stretch,
    "va_adaptive":   fn_va_adaptive,
}

N_CLASSES = 43


# ============================================================
# Dataset (★ v5: paper-consistent pipeline)
# ============================================================
class TestsetDataset(Dataset):
    def __init__(self, testset_name, enhancement_fn=None, transform=None):
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
        if self.fn is not None:
            img_bgr = self.fn(img_bgr)        # enhance at ORIGINAL size
        pil_img = cv2_to_pil_rgb(img_bgr)
        x = self.transform(pil_img)            # PIL Resize(32) → ToTensor → Normalize
        return x, label


def evaluate_pair(model, device, testset_name, method_name, transform):
    fn = METHODS[method_name]
    ds = TestsetDataset(testset_name, enhancement_fn=fn, transform=transform)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())
    acc = accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100
    pcacc = per_class_accuracy(labels, preds, N_CLASSES)
    min_class_acc = float(np.nanmin(pcacc))
    return {
        "testset": testset_name, "method": method_name,
        "n_samples": len(ds), "acc": acc,
        "macro_f1": macro_f1, "min_class_acc": min_class_acc,
        "per_class_acc": pcacc.tolist(),
    }


def main():
    verify_data_paths()
    init_run_config("A_metrics", {
        "version": "v5 (paper-consistent pipeline)",
        "testsets": TESTSETS,
        "methods": list(METHODS.keys()),
        "n_classes": N_CLASSES,
        "pipeline": "cv2.imread → enhance@original → BGR2RGB → PIL → Resize(32) → ToTensor → Normalize",
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    transform = build_paper_test_transform()
    model = load_gtsrb_compactcnn(device)
    print("Model loaded.")
    
    # === Sanity check ===
    print("\n" + "=" * 70)
    print("  Sanity check: clean baseline (expect ~93.92% per paper)")
    print("=" * 70)
    sanity = evaluate_pair(model, device, "clean", "baseline", transform)
    if sanity is None:
        return
    diff = sanity["acc"] - PAPER_CLEAN_BASELINE_ACC
    print(f"  Clean baseline acc: {sanity['acc']:.2f}%  (paper: {PAPER_CLEAN_BASELINE_ACC}%, Δ={diff:+.2f}pp)")
    if abs(diff) > 1.5:
        print(f"  ⚠ Warning: difference > 1.5pp")
    else:
        print(f"  ✓ Within tolerance, paper-consistent")
    
    all_results = [sanity]
    print("\n" + "=" * 70)
    print(f"  Evaluating remaining cells")
    print("=" * 70)
    
    for testset in TESTSETS:
        print(f"\n[Testset] {testset}")
        for method in METHODS:
            if testset == "clean" and method == "baseline":
                r = sanity
                print(f"  {method:15s} acc={r['acc']:5.2f}%  f1={r['macro_f1']:5.2f}%  min_cls={r['min_class_acc']:5.2f}%  n={r['n_samples']} (cached)")
                continue
            r = evaluate_pair(model, device, testset, method, transform)
            if r is None:
                continue
            print(f"  {method:15s} acc={r['acc']:5.2f}%  f1={r['macro_f1']:5.2f}%  min_cls={r['min_class_acc']:5.2f}%  n={r['n_samples']}")
            all_results.append(r)
    
    csv_path = OUTPUTS_DIR / "A_metrics_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["testset", "method", "n_samples", "accuracy", "macro_f1", "min_class_accuracy"])
        for r in all_results:
            w.writerow([r["testset"], r["method"], r["n_samples"],
                        f"{r['acc']:.4f}", f"{r['macro_f1']:.4f}", f"{r['min_class_acc']:.4f}"])
    print(f"\n[✓] {csv_path}")
    
    pc_csv = OUTPUTS_DIR / "A_per_class_accuracy.csv"
    with open(pc_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ["testset", "method"] + [f"cls_{i:02d}" for i in range(N_CLASSES)]
        w.writerow(header)
        for r in all_results:
            row = [r["testset"], r["method"]] + [f"{a:.2f}" for a in r["per_class_acc"]]
            w.writerow(row)
    print(f"[✓] {pc_csv}")
    
    pretty_path = OUTPUTS_DIR / "A_per_condition_table.csv"
    with open(pretty_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for metric_name, metric_key in [
            ("Top-1 Accuracy (%)", "acc"),
            ("Macro-F1 (%)", "macro_f1"),
            ("Min-class Accuracy (%)", "min_class_acc"),
        ]:
            w.writerow([f"=== {metric_name} ==="])
            w.writerow(["testset"] + list(METHODS.keys()))
            for ts in TESTSETS:
                row = [ts]
                for m in METHODS:
                    r = next((x for x in all_results if x["testset"] == ts and x["method"] == m), None)
                    row.append(f"{r[metric_key]:.2f}" if r else "—")
                w.writerow(row)
            w.writerow([])
    print(f"[✓] {pretty_path}")
    
    base_mixed = next((x for x in all_results if x["testset"] == "mixed" and x["method"] == "baseline"), None)
    va_mixed = next((x for x in all_results if x["testset"] == "mixed" and x["method"] == "va_adaptive"), None)
    if base_mixed and va_mixed:
        fig, ax = plt.subplots(figsize=(14, 4))
        x = np.arange(N_CLASSES)
        ax.bar(x - 0.2, base_mixed["per_class_acc"], width=0.4, label="Baseline", color="#888")
        ax.bar(x + 0.2, va_mixed["per_class_acc"], width=0.4, label="VA-Adaptive", color="#d62728")
        ax.set_xlabel("GTSRB Class ID")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Per-class accuracy on mixed degradation set")
        ax.set_xticks(np.arange(0, N_CLASSES, 2))
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_png = OUTPUTS_DIR / "figures" / "A_per_class_bar.png"
        plt.savefig(out_png, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"[✓] {out_png}")
    
    summary_lines = ["A. Metrics summary:"]
    for ts in TESTSETS:
        line_parts = [f"  {ts:13s}"]
        for m in METHODS:
            r = next((x for x in all_results if x["testset"] == ts and x["method"] == m), None)
            if r:
                line_parts.append(f"{m[:6]}={r['acc']:.1f}/{r['macro_f1']:.1f}")
        summary_lines.append("  ".join(line_parts))
    finalize_run("A_metrics", "\n".join(summary_lines))


if __name__ == "__main__":
    main()
