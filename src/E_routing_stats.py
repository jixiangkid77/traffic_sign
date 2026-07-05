"""
E_routing_stats.py — v1

Reports routing-branch proportions per testset to verify decision-rule behavior.
Addresses CVIDL2026 round-2 reviewer comment 2:
  "report routing-branch proportions under each degradation type to verify
   whether the decision rule behaves as expected"

Pipeline (consistent with A/B/D v5/v7):
  cv2.imread → compute b/c/e on ORIGINAL size → priority-ordered routing decision

Output:
  outputs_revision/E_routing_stats.csv
  outputs_revision/figures/E_routing_distribution.png
"""

import sys
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\Project\traffic_sign\src")
from revision_utils import (
    OUTPUTS_DIR,
    PAPER_THRESHOLDS, init_run_config, finalize_run,
    compute_image_stats,
    verify_data_paths,
    collect_gtsrb_test_samples,
)

TESTSETS = ["clean", "lowlight", "foggy", "lowcontrast", "noisy", "mixed"]
BRANCHES = ["gamma", "clahe", "stretch", "passthrough"]


def _T():
    return (PAPER_THRESHOLDS["T1"], PAPER_THRESHOLDS["T2"],
            PAPER_THRESHOLDS["T3"], PAPER_THRESHOLDS["T4"])


def route(b, c, e, T1, T2, T3, T4):
    """Priority-ordered routing decision.
    Mirrors the dispatch logic in enhance.adaptive_enhance — returns the
    BRANCH NAME that adaptive_enhance would dispatch to.

    Priority order (from paper Section III-C):
        1. b < T1                     → gamma
        2. else if c < T2             → clahe
        3. else if e < T3 AND b > T4  → stretch
        4. else                       → passthrough
    """
    if b < T1:
        return "gamma"
    if c < T2:
        return "clahe"
    if (e < T3) and (b > T4):
        return "stretch"
    return "passthrough"


def measure_routing_for_testset(testset_name, T):
    """Iterate every image in the testset, tally which branch fires."""
    samples = collect_gtsrb_test_samples(testset_name)
    counter = Counter({b: 0 for b in BRANCHES})
    failed = 0
    for path, _ in samples:
        img = cv2.imread(path)
        if img is None:
            failed += 1
            continue
        # ★ original size, consistent with A/B/D
        b, c, e = compute_image_stats(img)
        counter[route(b, c, e, *T)] += 1
    return counter, len(samples) - failed, failed


def main():
    verify_data_paths(check_train=False, check_degraded=True)

    init_run_config("E_routing_stats", {
        "version": "v1",
        "purpose": "round-2 reviewer comment 2: routing-branch verification",
        "testsets": TESTSETS,
        "branches": BRANCHES,
        "thresholds": PAPER_THRESHOLDS,
        "pipeline": "cv2.imread → compute b/c/e @ original size → priority-ordered routing",
    })

    T = _T()
    print(f"\nThresholds: T1={T[0]:.4f}, T2={T[1]:.4f}, T3={T[2]:.4f}, T4={T[3]:.4f}")

    print("\n" + "=" * 70)
    print("  Routing-branch proportions per testset")
    print("=" * 70)

    all_results = {}
    for ts in TESTSETS:
        print(f"\n[Testset] {ts}")
        counter, n_total, n_failed = measure_routing_for_testset(ts, T)
        if n_failed:
            print(f"  ⚠ {n_failed} images failed to load")
        if n_total == 0:
            print(f"  ⚠ No images, skipping")
            continue
        all_results[ts] = {
            "n": n_total,
            "counter": counter,
            "pct": {b: counter[b] / n_total * 100 for b in BRANCHES},
        }
        for b in BRANCHES:
            cnt = counter[b]
            pct = cnt / n_total * 100
            print(f"  {b:13s}: {cnt:5d} ({pct:5.1f}%)")

    # === CSV ===
    csv_path = OUTPUTS_DIR / "E_routing_stats.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["=== Routing-branch proportions (%) ==="])
        w.writerow(["testset"] + BRANCHES + ["n_images"])
        for ts in TESTSETS:
            r = all_results.get(ts)
            if r is None:
                continue
            row = [ts] + [f"{r['pct'][b]:.2f}" for b in BRANCHES] + [r["n"]]
            w.writerow(row)
        w.writerow([])
        w.writerow(["=== Raw counts ==="])
        w.writerow(["testset"] + BRANCHES + ["n_images"])
        for ts in TESTSETS:
            r = all_results.get(ts)
            if r is None:
                continue
            row = [ts] + [r["counter"][b] for b in BRANCHES] + [r["n"]]
            w.writerow(row)
    print(f"\n[✓] {csv_path}")

    # === Figure: stacked bar of routing distribution ===
    if all_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ts_list = [ts for ts in TESTSETS if ts in all_results]
        x = np.arange(len(ts_list))
        bottom = np.zeros(len(ts_list))
        colors = {
            "gamma":       "#ff7f0e",
            "clahe":       "#1f77b4",
            "stretch":     "#9467bd",
            "passthrough": "#888888",
        }
        for b in BRANCHES:
            heights = [all_results[ts]["pct"][b] for ts in ts_list]
            ax.bar(x, heights, bottom=bottom, label=b, color=colors[b],
                   edgecolor='black', linewidth=0.5)
            for i, (h, btm) in enumerate(zip(heights, bottom)):
                if h > 5:
                    ax.text(i, btm + h / 2, f"{h:.0f}%",
                            ha='center', va='center', fontsize=10, fontweight='bold')
            bottom += np.array(heights)
        ax.set_xticks(x)
        ax.set_xticklabels(ts_list, fontsize=11)
        ax.set_ylabel("Routing share (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.13),
                  fontsize=10, frameon=False)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        out_png = OUTPUTS_DIR / "figures" / "E_routing_distribution.png"
        plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[✓] {out_png}")

    # === Summary ===
    summary = ["E. Routing-branch proportions:"]
    for ts in TESTSETS:
        r = all_results.get(ts)
        if r is None:
            continue
        dom = max(BRANCHES, key=lambda b: r["pct"][b])
        summary.append(f"  {ts:13s}: {dom} dominant ({r['pct'][dom]:.1f}%) | "
                       + ", ".join(f"{b}={r['pct'][b]:.0f}%" for b in BRANCHES))
    finalize_run("E_routing_stats", "\n".join(summary))


if __name__ == "__main__":
    main()