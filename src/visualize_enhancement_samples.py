"""
Day 4-B: 增强效果定性对比可视化 (v2)。

v2 改进：自动选择会触发"正确分支"的样例图，保证：
  Low-light    行 → VA-Adaptive 选 Gamma 分支
  Foggy        行 → VA-Adaptive 选 Stretch 分支
  Low-contrast 行 → VA-Adaptive 选 CLAHE 分支

输出（论文 Fig 5）：
  - Fig5_enhancement_samples.png

时间：约 30 秒（搜索图片 + 画图）
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from enhance import (  # noqa: E402
    apply_clahe, apply_gamma, adaptive_enhance, compute_image_stats,
)
from degrade import (  # noqa: E402
    degrade_lowlight, degrade_foggy, degrade_lowcontrast,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# 加载阈值
with open(RESULTS_DIR / 'thresholds.json') as f:
    th = json.load(f)
T1, T2 = th['T1_brightness_low'], th['T2_contrast_low']
T3, T4 = th['T3_edge_low'], th['T4_brightness_high']
print(f"阈值: T1={T1:.4f}, T2={T2:.4f}, T3={T3:.4f}, T4={T4:.4f}")


def detect_branch(img_bgr):
    """复现 adaptive_enhance 内部判定逻辑。"""
    b, c, e = compute_image_stats(img_bgr)
    if b < T1:
        return 'gamma', 'Gamma (low-light)'
    elif c < T2:
        return 'clahe', 'CLAHE (low-contrast)'
    elif e < T3 and b > T4:
        return 'stretch', 'Stretch (foggy)'
    else:
        return 'normal', 'None (normal)'


def find_image_for_branch(all_imgs, degrade_fn, target_branch, start_idx=0):
    """在测试集中搜索能触发 target_branch 的图。

    Args:
        target_branch: 'gamma' / 'clahe' / 'stretch'

    Returns:
        (Path, dict) — 图路径和该图退化后的统计量；找不到返回 (None, None)
    """
    n_total = len(all_imgs)
    # 从 start_idx 开始往后扫，最多扫 2000 张
    for i in range(start_idx, min(start_idx + 2000, n_total)):
        p = all_imgs[i]
        img = cv2.imread(str(p))
        if img is None or min(img.shape[:2]) < 48:
            continue
        # 先 resize 到 64x64（和最终展示一致）
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        img_deg = degrade_fn(img_resized)
        branch, _ = detect_branch(img_deg)
        if branch == target_branch:
            b, c, e = compute_image_stats(img_deg)
            return p, {'b': b, 'c': c, 'e': e, 'idx': i}
    return None, None


# ============== 找图 ==============
img_dir = DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Test' / 'Images'
if not img_dir.exists():
    for p in DATA_ROOT.rglob('Final_Test'):
        if p.is_dir():
            img_dir = p / 'Images'
            break

all_imgs = sorted(img_dir.glob('*.ppm'))
print(f"找到 {len(all_imgs)} 张测试图\n")

print("智能选图（找会触发对应分支的样例）...")

# 三个场景，分别找会触发对应分支的图
search_plan = [
    ('Low-light',     degrade_lowlight,    'gamma',   100),
    ('Foggy',         degrade_foggy,       'stretch', 2000),
    ('Low-contrast',  degrade_lowcontrast, 'clahe',   5000),
]

scenarios = []
for scene_name, degrade_fn, target_branch, start_idx in search_plan:
    img_path, stats = find_image_for_branch(
        all_imgs, degrade_fn, target_branch, start_idx
    )
    if img_path is None:
        print(f"  ✗ {scene_name:13s} 找不到触发 {target_branch} 分支的图，"
              f"放宽搜索范围 ...")
        # 回退：找任意能触发的图（从开头）
        img_path, stats = find_image_for_branch(
            all_imgs, degrade_fn, target_branch, 0
        )
    if img_path is None:
        # 实在找不到，用默认 fallback
        print(f"  ✗ {scene_name:13s} 仍找不到，使用第 {start_idx} 张")
        img_path = all_imgs[min(start_idx, len(all_imgs) - 1)]
    else:
        print(f"  ✓ {scene_name:13s} → {img_path.name}  "
              f"(idx={stats['idx']}, b={stats['b']:.3f}, "
              f"c={stats['c']:.3f}, e={stats['e']:.3f})")
    scenarios.append((scene_name, degrade_fn, img_path))


# ============== 画图 ==============
n_rows = len(scenarios)
n_cols = 5

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.6))

col_titles = ['Original', 'Degraded', '+ Fixed CLAHE', '+ Fixed Gamma',
              '+ VA-Adaptive (Ours)']

for i, (scene_name, degrade_fn, img_path) in enumerate(scenarios):
    img_orig = cv2.imread(str(img_path))
    img_orig = cv2.resize(img_orig, (64, 64), interpolation=cv2.INTER_LINEAR)

    img_deg = degrade_fn(img_orig)
    img_clahe = apply_clahe(img_deg)
    img_gamma = apply_gamma(img_deg)
    img_adapt = adaptive_enhance(img_deg, T1, T2, T3, T4)

    _, branch_str = detect_branch(img_deg)

    cells = [img_orig, img_deg, img_clahe, img_gamma, img_adapt]

    for j, img in enumerate(cells):
        ax = axes[i, j]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_title(col_titles[j], fontsize=11, fontweight='bold', pad=8)

        if j == 0:
            ax.text(-0.18, 0.5, scene_name, fontsize=12, fontweight='bold',
                     ha='right', va='center', rotation=90,
                     transform=ax.transAxes)

        if j == n_cols - 1:
            color = '#27AE60' if 'None' not in branch_str else '#7F8C8D'
            ax.text(0.5, -0.10, f'Branch: {branch_str}', fontsize=9,
                     color=color, ha='center', va='top',
                     fontweight='bold', transform=ax.transAxes)

fig.suptitle(
    'Qualitative Comparison of Image Enhancement Methods on Degraded Traffic Signs',
    fontsize=13, y=1.00,
)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
fig5_path = RESULTS_DIR / 'Fig5_enhancement_samples.png'
plt.savefig(fig5_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✓ 已保存: {fig5_path}")
print("\n核心检查：每行最后一列下方的 Branch 标注应该是：")
print("  Low-light    → Gamma (low-light)")
print("  Foggy        → Stretch (foggy)")
print("  Low-contrast → CLAHE (low-contrast)")
