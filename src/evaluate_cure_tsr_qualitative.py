"""
evaluate_cure_tsr_qualitative.py — CURE-TSR 快速定性 sanity check

目标：在投入 2 天做方案 B（重训分类器）之前，先用 30 分钟
确认 VA-Adaptive 在 CURE-TSR 真实退化图上"看起来"有效。

输出：
  - cure_tsr_qualitative.png — 4×N grid，每行：
      [original ChallengeFree] | [degraded] | [VA-Adaptive output] | [routing branch]
  - cure_tsr_routing_stats.json — VA-Adaptive 在每种 challenge 上的 branch 选择分布

关键判断（运行后看输出）：
  ✓ 如果 VA-Adaptive 在 DarkenChallenge 上**主要选 gamma**
    在 Haze 上**主要选 stretch**
    在 Noise 上**保持原图**（因为没噪声分支）
    → 说明你的 thresholds 在真实数据上**也 work**，方案 B 值得做
  
  ✗ 如果 routing 看起来随机或全部走 pass-through
    → 阈值在 cross-dataset 不工作，需要重新校准（在 CURE-TSR 上）
       或者诚实承认作为 limitation

运行：
  conda activate pcm_sim
  python evaluate_cure_tsr_qualitative.py
"""

import os
import sys
import json
import random
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# 注意：这里假设你的 src/enhance.py 里有 v1 的 VA-Adaptive 实现
sys.path.insert(0, r"D:\Project\traffic_sign\src")
from enhance import adaptive_enhance, apply_clahe, apply_gamma  # noqa

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
CURE_TSR_DIR = PROJECT_ROOT / "datasets" / "CURE-TSR" / "Real_Test"
OUTPUT_DIR = PROJECT_ROOT / "outputs_cure_tsr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 你 paper 训练数据校准的阈值（来自 thresholds.json）
THRESHOLDS = {
    "T1": 0.1206,  # brightness lower
    "T2": 0.1061,  # contrast lower
    "T3": 0.0726,  # edge lower
    "T4": 0.4085,  # brightness upper (foggy detection)
}

# 要展示的 4 个 challenge types（对应你 paper 5 种合成退化的 4 个）
CHALLENGES_TO_EVAL = {
    "DarkenChallenge-3": "lowlight (real)",
    "Haze-3":            "foggy (real)",
    "Noise-3":           "noisy (real)",
    "Rain-3":            "rainy (real, bonus)",
}

# 每个 challenge 取 N 个样本展示
N_SAMPLES_PER_CHALLENGE = 5

# 随机种子（保证 figure 可复现）
random.seed(42)
np.random.seed(42)


# ============================================================
# 计算图像统计量（与 paper 一致）
# ============================================================
def compute_stats(img_bgr):
    """图像统计：brightness, contrast, edge density"""
    if img_bgr.dtype == np.uint8:
        img_norm = img_bgr.astype(np.float32) / 255.0
    else:
        img_norm = img_bgr
    
    gray = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_norm = gray.astype(np.float32) / 255.0
    
    b = float(np.mean(gray_norm))
    c = float(2 * np.std(gray_norm))
    edges = cv2.Canny(gray, 50, 150)
    e = float(np.mean(edges > 0))
    
    return b, c, e


def route_decision(b, c, e, T):
    """与 enhance.py v1 一致的 4 分支路由"""
    if b < T["T1"]:
        return "gamma"
    elif c < T["T2"]:
        return "clahe"
    elif e < T["T3"] and b > T["T4"]:
        return "stretch"
    else:
        return "passthrough"


def apply_va_adaptive(img_bgr, T):
    """模拟你 enhance.py 里的 adaptive_enhance"""
    b, c, e = compute_stats(img_bgr)
    branch = route_decision(b, c, e, T)
    
    if branch == "gamma":
        out = apply_gamma(img_bgr, gamma=0.5)
    elif branch == "clahe":
        out = apply_clahe(img_bgr, clip_limit=3.0)
    elif branch == "stretch":
        # 线性 stretch
        img_f = img_bgr.astype(np.float32) / 255.0
        out_f = np.clip((img_f - 0.5) * 1.5 + 0.5, 0, 1)
        out = (out_f * 255).astype(np.uint8)
    else:
        out = img_bgr.copy()
    
    return out, branch, (b, c, e)


# ============================================================
# 核心：处理一个 challenge type
# ============================================================
def process_challenge(challenge_name, label, n_samples):
    """对一个 challenge folder 里随机抽 n_samples 张图，跑 VA-Adaptive"""
    folder = CURE_TSR_DIR / challenge_name
    if not folder.exists():
        print(f"[!] 跳过：{folder} 不存在")
        return None
    
    files = [f for f in os.listdir(folder) if f.endswith(('.bmp', '.png', '.jpg'))]
    if not files:
        print(f"[!] 跳过：{folder} 是空的")
        return None
    
    selected = random.sample(files, min(n_samples, len(files)))
    
    results = []
    branch_counts = {"gamma": 0, "clahe": 0, "stretch": 0, "passthrough": 0}
    
    for fname in selected:
        img_path = folder / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # CURE-TSR 图片大小通常是 28×28 或 64×64，统一 resize 到 64×64
        img_resized = cv2.resize(img, (64, 64))
        
        enhanced, branch, stats = apply_va_adaptive(img_resized, THRESHOLDS)
        branch_counts[branch] += 1
        
        results.append({
            "filename": fname,
            "input": cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),
            "output": cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
            "branch": branch,
            "stats": stats,
        })
    
    # 统计完整 challenge 上的 branch 分布（不止抽样，全跑一遍）
    full_branch_counts = {"gamma": 0, "clahe": 0, "stretch": 0, "passthrough": 0}
    for fname in files:
        img = cv2.imread(str(folder / fname))
        if img is None:
            continue
        img_resized = cv2.resize(img, (64, 64))
        b, c, e = compute_stats(img_resized)
        branch = route_decision(b, c, e, THRESHOLDS)
        full_branch_counts[branch] += 1
    
    total = sum(full_branch_counts.values())
    full_branch_pct = {k: v / total * 100 for k, v in full_branch_counts.items()}
    
    print(f"  [{challenge_name:25s}] {label}")
    print(f"    Total images: {total}")
    print(f"    Branch distribution:")
    for branch, pct in sorted(full_branch_pct.items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"      {branch:12s} {pct:5.1f}%  {bar}")
    
    return {
        "challenge": challenge_name,
        "label": label,
        "samples": results,
        "full_branch_pct": full_branch_pct,
        "total_images": total,
    }


# ============================================================
# 可视化（关键 figure）
# ============================================================
def make_figure(all_results, output_path):
    """每个 challenge 一行，每行 N 列 (input | output | branch label)"""
    n_challenges = len(all_results)
    n_cols = N_SAMPLES_PER_CHALLENGE * 2 + 1  # input/output 配对 + 标签列
    
    fig, axes = plt.subplots(n_challenges, n_cols, 
                              figsize=(n_cols * 1.4, n_challenges * 1.6))
    
    if n_challenges == 1:
        axes = axes[None, :]
    
    for row, result in enumerate(all_results):
        # 第一列：challenge 名字
        axes[row, 0].text(0.5, 0.5, result["label"],
                          ha='center', va='center',
                          fontsize=10, fontweight='bold',
                          transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')
        
        # 后面成对的 input/output
        for col_idx, sample in enumerate(result["samples"][:N_SAMPLES_PER_CHALLENGE]):
            in_col = 1 + col_idx * 2
            out_col = 2 + col_idx * 2
            
            axes[row, in_col].imshow(sample["input"])
            axes[row, in_col].set_title("input", fontsize=8)
            axes[row, in_col].axis('off')
            
            axes[row, out_col].imshow(sample["output"])
            axes[row, out_col].set_title(f"→ {sample['branch']}", fontsize=8)
            axes[row, out_col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] 定性图已保存：{output_path}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"CURE-TSR 定性 sanity check")
    print(f"阈值（来自 GTSRB 训练集校准）: {THRESHOLDS}")
    print(f"Real_Test 路径: {CURE_TSR_DIR}")
    print()
    
    if not CURE_TSR_DIR.exists():
        print(f"[!] CURE-TSR 数据未找到。先运行 download_cure_tsr.py")
        sys.exit(1)
    
    print(f"扫描 {len(CHALLENGES_TO_EVAL)} 个 challenge types...\n")
    
    all_results = []
    routing_stats = {}
    
    for challenge_name, label in CHALLENGES_TO_EVAL.items():
        result = process_challenge(challenge_name, label, N_SAMPLES_PER_CHALLENGE)
        if result is not None:
            all_results.append(result)
            routing_stats[challenge_name] = {
                "label": label,
                "branch_distribution_pct": result["full_branch_pct"],
                "total_images": result["total_images"],
            }
        print()
    
    # 写 routing 统计
    out_json = OUTPUT_DIR / "cure_tsr_routing_stats.json"
    with open(out_json, 'w') as f:
        json.dump(routing_stats, f, indent=2)
    print(f"[✓] Routing 统计已保存：{out_json}")
    
    # 画图
    if all_results:
        out_png = OUTPUT_DIR / "cure_tsr_qualitative.png"
        make_figure(all_results, out_png)
    
    # 关键解读（自动判断 sanity check 结果）
    print("\n" + "=" * 70)
    print("  Sanity check 自动判读")
    print("=" * 70)
    
    for challenge_name, stats in routing_stats.items():
        pct = stats["branch_distribution_pct"]
        label = stats["label"]
        
        # 期望的"正确路由"
        if "DarkenChallenge" in challenge_name:
            expected = "gamma"
            criterion = pct.get("gamma", 0) > 50
        elif "Haze" in challenge_name:
            expected = "stretch"
            criterion = pct.get("stretch", 0) > 30 or pct.get("clahe", 0) > 30
        elif "Noise" in challenge_name:
            expected = "passthrough or clahe"
            criterion = pct.get("passthrough", 0) > 30 or pct.get("clahe", 0) > 30
        elif "Rain" in challenge_name:
            expected = "varies"
            criterion = max(pct.values()) > 30  # 至少有一个分支主导
        else:
            expected = "any"
            criterion = True
        
        verdict = "✓ PASS" if criterion else "✗ FAIL"
        print(f"  {verdict}  {label:20s} -- expected: {expected}")
        print(f"         actual: gamma={pct.get('gamma', 0):.0f}%, "
              f"clahe={pct.get('clahe', 0):.0f}%, "
              f"stretch={pct.get('stretch', 0):.0f}%, "
              f"passthrough={pct.get('passthrough', 0):.0f}%")
    
    print("\n" + "=" * 70)
    print("  下一步建议：")
    print("=" * 70)
    
    pass_count = 0
    for challenge_name, stats in routing_stats.items():
        pct = stats["branch_distribution_pct"]
        if "DarkenChallenge" in challenge_name and pct.get("gamma", 0) > 50:
            pass_count += 1
        elif "Haze" in challenge_name and (pct.get("stretch", 0) > 30 or pct.get("clahe", 0) > 30):
            pass_count += 1
    
    if pass_count >= 2:
        print(f"  ✓ Routing 在 real-world 数据上看起来有效（{pass_count}/2+ 关键 case 通过）")
        print(f"  → 投入方案 B（重训 14 类分类器）值得做")
        print(f"  → 下一步：python train_cure_classifier.py")
    else:
        print(f"  ⚠ Routing 在 real-world 数据上不太对（{pass_count}/2 关键 case 通过）")
        print(f"  → 选项 1：在 CURE-TSR 上重新校准 thresholds")
        print(f"  → 选项 2：仅做 qualitative validation，方案 B 不投入")


if __name__ == "__main__":
    main()
