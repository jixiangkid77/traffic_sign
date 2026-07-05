"""
Day 3 Step 1: 在 clean GTSRB 训练集上自动标定阈值 T1-T4。

逻辑：clean 训练集代表"模型眼中的正常图像"。如果一张测试图的
brightness/contrast/edge 显著低于 clean 训练集的低分位，则认为是退化。

这种"分位数自动定阈值"的写法在论文里讲得清楚 + 可复现 + 抗审，
比手动拍脑袋设魔法数字强很多。

输出：
  results/thresholds.json              4 个阈值 + 训练集统计分布
  results/threshold_distribution.png   直方图（论文 Fig 2 候选）

跑一次约 2-3 分钟。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from enhance import compute_image_stats  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# 找训练集图片目录（兼容不同 torchvision 版本的目录结构）
TRAIN_DIR_CANDIDATES = [
    DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Training',
    DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Training' / 'Images',
    DATA_ROOT / 'gtsrb' / 'Training',
    DATA_ROOT / 'gtsrb' / 'Final_Training' / 'Images',
]
TRAIN_DIR = next((p for p in TRAIN_DIR_CANDIDATES if p.exists()), None)
if TRAIN_DIR is None:
    # 兜底：rglob 找
    for p in DATA_ROOT.rglob('Training'):
        if any(p.rglob('*.ppm')):
            TRAIN_DIR = p
            break

if TRAIN_DIR is None:
    raise FileNotFoundError(
        "找不到训练集目录。已检查:\n" +
        "\n".join(f"  - {p}" for p in TRAIN_DIR_CANDIDATES)
    )

# 训练集按类别分子文件夹，递归找所有 .ppm
train_paths = list(TRAIN_DIR.rglob('*.ppm'))
print(f"训练集目录: {TRAIN_DIR}")
print(f"找到 {len(train_paths)} 张训练图\n")

if len(train_paths) == 0:
    raise RuntimeError(f"目录 {TRAIN_DIR} 下没有 .ppm 文件")

# ============== 计算所有图的统计量 ==============
print("计算 clean 训练集的统计量分布 ...")
stats = []
for p in tqdm(train_paths, desc="读图算 stats", ncols=80):
    img = cv2.imread(str(p))
    if img is None:
        continue
    stats.append(compute_image_stats(img))
stats = np.array(stats)  # shape (N, 3): [brightness, contrast, edge]

print(f"\nclean 训练集统计 (N={len(stats)}):")
print(f"  brightness: mean={stats[:, 0].mean():.3f}  std={stats[:, 0].std():.3f}")
print(f"  contrast:   mean={stats[:, 1].mean():.3f}  std={stats[:, 1].std():.3f}")
print(f"  edge:       mean={stats[:, 2].mean():.3f}  std={stats[:, 2].std():.3f}")

# ============== 阈值取分位数 ==============
T1 = float(np.percentile(stats[:, 0], 15))   # brightness 低于此 → low_light
T2 = float(np.percentile(stats[:, 1], 15))   # contrast 低于此 → low_contrast
T3 = float(np.percentile(stats[:, 2], 15))   # edge 弱于此 → 候选 foggy
T4 = float(np.percentile(stats[:, 0], 70))   # 同时 brightness 高于此 → 确认 foggy

print(f"\n确定阈值（基于 clean 训练集分布）：")
print(f"  T1 (brightness 15th pct) = {T1:.4f}  → 低于此判定 low_light")
print(f"  T2 (contrast   15th pct) = {T2:.4f}  → 低于此判定 low_contrast")
print(f"  T3 (edge       15th pct) = {T3:.4f}  → 弱于此 + 高亮判定 foggy")
print(f"  T4 (brightness 70th pct) = {T4:.4f}  → 配合 T3 判 foggy")

# ============== 保存阈值 ==============
out_data = {
    'T1_brightness_low':  T1,
    'T2_contrast_low':    T2,
    'T3_edge_low':        T3,
    'T4_brightness_high': T4,
    'method': 'percentile_based',
    'percentiles': {
        'T1': '15th of brightness',
        'T2': '15th of contrast',
        'T3': '15th of edge',
        'T4': '70th of brightness',
    },
    'stats_summary': {
        'brightness_mean': float(stats[:, 0].mean()),
        'brightness_std':  float(stats[:, 0].std()),
        'contrast_mean':   float(stats[:, 1].mean()),
        'contrast_std':    float(stats[:, 1].std()),
        'edge_mean':       float(stats[:, 2].mean()),
        'edge_std':        float(stats[:, 2].std()),
        'n_samples':       len(stats),
    },
}
out_path = RESULTS_DIR / 'thresholds.json'
with open(out_path, 'w') as f:
    json.dump(out_data, f, indent=2)
print(f"\n✓ 阈值保存到: {out_path}")

# ============== 画分布直方图 ==============
print("\n画分布直方图 ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ['Brightness', 'Contrast', 'Edge Strength']
thresholds_for_plot = [(T1, T4), (T2, None), (T3, None)]
colors = ['#FF6B6B', '#4ECDC4', '#FFA500']

for i, (label, color) in enumerate(zip(labels, colors)):
    axes[i].hist(stats[:, i], bins=50, color=color, alpha=0.7, edgecolor='black')
    t_low, t_high = thresholds_for_plot[i]
    axes[i].axvline(t_low, color='blue', linestyle='--', linewidth=2,
                    label=f'15th pct = {t_low:.3f}')
    if t_high is not None:
        axes[i].axvline(t_high, color='purple', linestyle='--', linewidth=2,
                        label=f'70th pct = {t_high:.3f}')
    axes[i].set_xlabel(label, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].set_title(f'{label} on clean GTSRB training set', fontsize=12)
    axes[i].legend(fontsize=10)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
fig_path = RESULTS_DIR / 'threshold_distribution.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()

print(f"✓ 直方图保存到: {fig_path}")
print()
print("下一步：python src/evaluate_all.py")
