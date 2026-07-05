"""
C_measure_latency.py — v5

★ v5 关键修复:
  v4 错: cv2.imread → cv2.resize(32x32) → 测延迟 (在 32x32 上 enhance)
  v5 对: cv2.imread → 测延迟 (在原始尺寸上 enhance, 与 paper deployment 一致)

理由：paper 的 evaluate_all.py pipeline 是在 ORIGINAL 尺寸上 enhance，
然后 PIL Resize 到 32x32。所以延迟测量也应该在原始尺寸做。

GTSRB test 图像尺寸不一（约 25×25 ~ 250×250），平均 ~50×50。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import csv
import platform
import numpy as np

import cv2
cv2.setNumThreads(1)

import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, r"D:\Project\traffic_sign\src")
from revision_utils import (
    OUTPUTS_DIR, init_run_config, finalize_run,
    PAPER_THRESHOLDS, verify_data_paths,
    collect_gtsrb_test_samples,
)
from enhance import apply_clahe, apply_gamma, adaptive_enhance

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    torch = None


N_IMAGES = 1000
N_WARMUP = 100
N_REPEAT = 5


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


def load_test_images(n_images=N_IMAGES):
    """★ v5: 加载原始尺寸图片 (无 cv2.resize)"""
    samples = collect_gtsrb_test_samples("clean")
    images = []
    for path, _ in samples[:n_images]:
        img = cv2.imread(path)
        if img is None:
            continue
        # ★ no resize
        images.append(img)
    if len(images) == 0:
        raise RuntimeError("No test images loaded")
    return images


def measure_method(fn, images, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    for i in range(n_warmup):
        _ = fn(images[i % len(images)])
    
    results = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for img in images:
            _ = fn(img)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        results.append({
            "elapsed_s": elapsed,
            "per_image_ms": elapsed * 1000 / len(images),
            "fps": len(images) / elapsed,
        })
    return results


def main():
    verify_data_paths(check_train=False, check_degraded=False)
    
    init_run_config("C_latency", {
        "version": "v5 (original size, paper-consistent deployment)",
        "n_images": N_IMAGES, "n_warmup": N_WARMUP, "n_repeat": N_REPEAT,
        "image_size": "original (variable, mean ~50x50)",
        "mode": "single-image",
        "methods": list(METHODS.keys()),
        "thread_config": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "cv2_threads": cv2.getNumThreads(),
            "torch_threads": torch.get_num_threads() if torch else "n/a",
        },
    })
    
    print("\n" + "=" * 70)
    print("  Latency Measurement (single-image, original size, 1 thread)")
    print("=" * 70)
    print(f"  CPU: {platform.processor() or 'unknown'}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  N images: {N_IMAGES} (original sizes)")
    print(f"  Threads: OMP={os.environ.get('OMP_NUM_THREADS')}, "
          f"cv2={cv2.getNumThreads()}, "
          f"torch={torch.get_num_threads() if torch else 'n/a'}")
    
    print("\n[Loading] Test images (at original sizes)...")
    images = load_test_images(N_IMAGES)
    print(f"  Loaded {len(images)} images")
    sizes = [img.shape[:2] for img in images]
    h_arr, w_arr = zip(*sizes)
    print(f"  Size range: H={min(h_arr)}-{max(h_arr)}, W={min(w_arr)}-{max(w_arr)}, "
          f"mean=({np.mean(h_arr):.1f}, {np.mean(w_arr):.1f})")
    
    print("\n" + "=" * 70)
    print("  Measuring...")
    print("=" * 70)
    
    all_method_results = {}
    for method_name, fn in METHODS.items():
        print(f"\n[{method_name}]")
        repeats = measure_method(fn, images, N_WARMUP, N_REPEAT)
        per_img_means = [r["per_image_ms"] for r in repeats]
        fps_means = [r["fps"] for r in repeats]
        mean_ms = np.mean(per_img_means)
        std_ms = np.std(per_img_means)
        mean_fps = np.mean(fps_means)
        all_method_results[method_name] = {
            "per_image_ms_mean": float(mean_ms),
            "per_image_ms_std": float(std_ms),
            "fps_mean": float(mean_fps),
        }
        print(f"  per-image: {mean_ms:.4f} ± {std_ms:.4f} ms  ({mean_fps:.0f} FPS)")
    
    csv_path = OUTPUTS_DIR / "C_latency_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["=== Latency Measurement (v5, single-image, original size) ==="])
        w.writerow([])
        w.writerow(["CPU", platform.processor() or "unknown"])
        w.writerow(["Python", sys.version.split()[0]])
        w.writerow(["OpenCV", cv2.__version__])
        w.writerow(["N images", N_IMAGES])
        w.writerow(["Image size", f"variable original (mean H={np.mean(h_arr):.1f}, W={np.mean(w_arr):.1f})"])
        w.writerow(["N warmup", N_WARMUP])
        w.writerow(["N repeat", N_REPEAT])
        w.writerow(["OMP threads", os.environ.get("OMP_NUM_THREADS")])
        w.writerow(["cv2 threads", cv2.getNumThreads()])
        w.writerow([])
        w.writerow(["Method", "Per-image (ms)", "Std (ms)", "FPS"])
        for m, r in all_method_results.items():
            w.writerow([m, f"{r['per_image_ms_mean']:.4f}",
                        f"{r['per_image_ms_std']:.4f}", f"{r['fps_mean']:.1f}"])
        w.writerow([])
        baseline_ms = all_method_results["baseline"]["per_image_ms_mean"]
        w.writerow(["=== Overhead vs Baseline ==="])
        w.writerow(["Method", "Overhead (ms)", "Overhead (×)"])
        for m, r in all_method_results.items():
            if m == "baseline":
                w.writerow([m, "0", "1.00x"])
            else:
                ovh = r["per_image_ms_mean"] - baseline_ms
                ratio = r["per_image_ms_mean"] / baseline_ms if baseline_ms > 0 else float('nan')
                w.writerow([m, f"{ovh:.4f}", f"{ratio:.2f}x"])
    print(f"\n[✓] {csv_path}")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(all_method_results.keys())
    means = [all_method_results[m]["per_image_ms_mean"] for m in methods]
    stds = [all_method_results[m]["per_image_ms_std"] for m in methods]
    colors = ['#888', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
    ax.bar(methods, means, yerr=stds, color=colors, capsize=5)
    ax.set_ylabel("Per-image latency (ms)")
    ax.set_title("Enhancement Latency (single-image, original size, 1 CPU thread)")
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label="1 ms")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    for i, (mn, s) in enumerate(zip(means, stds)):
        ax.text(i, mn + s + 0.02, f"{mn:.3f} ms", ha='center', fontsize=9)
    plt.tight_layout()
    out_png = OUTPUTS_DIR / "figures" / "C_latency_bar.png"
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[✓] {out_png}")
    
    va_ms = all_method_results["va_adaptive"]["per_image_ms_mean"]
    summary_lines = ["C. Latency (v5):"]
    for m, r in all_method_results.items():
        summary_lines.append(f"  {m}: {r['per_image_ms_mean']:.4f} ms")
    summary_lines.append(f"\n  VA-Adaptive: {va_ms:.4f} ms (at original sizes)")
    finalize_run("C_latency", "\n".join(summary_lines))
    
    print("\n" + "=" * 70)
    if va_ms < 1.0:
        print(f"  ✓ VA-Adaptive {va_ms:.3f} ms < 1.0 ms → 'sub-millisecond' OK")
    else:
        print(f"  ⚠ VA-Adaptive {va_ms:.3f} ms ≥ 1.0 ms")


if __name__ == "__main__":
    main()
