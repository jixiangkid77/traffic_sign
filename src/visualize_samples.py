"""
画一张样例对比图：选 3 张测试图，展示原图 + 5 种退化。

输出：sample_degradations.png（保存在项目根目录）

跑完打开图片肉眼检查每种退化是否合理。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from degrade import DEGRADATIONS, degrade_mixed  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'

# 找 GTSRB 测试集目录
SRC_DIR_CANDIDATES = [
    DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Test' / 'Images',
    DATA_ROOT / 'gtsrb' / 'Final_Test' / 'Images',
]
SRC_DIR = next((p for p in SRC_DIR_CANDIDATES if p.exists()), None)
if SRC_DIR is None:
    for p in DATA_ROOT.rglob('Final_Test'):
        if (p / 'Images').exists():
            SRC_DIR = p / 'Images'
            break

if SRC_DIR is None:
    raise FileNotFoundError("找不到 GTSRB 测试集，请先跑 download_data.py")

# 选 3 张样例（间隔取，避免都是同一类）
all_paths = sorted(SRC_DIR.glob('*.ppm'))
idx = [100, 5000, 10000]
sample_paths = [all_paths[i] for i in idx if i < len(all_paths)]

deg_names = ['original', 'lowlight', 'foggy', 'lowcontrast', 'noisy', 'mixed']
n_samples = len(sample_paths)
n_degs = len(deg_names)

fig, axes = plt.subplots(n_samples, n_degs,
                          figsize=(n_degs * 2.2, n_samples * 2.2))

# 单 sample 时 axes 是 1D，统一成 2D
if n_samples == 1:
    axes = axes.reshape(1, -1)

rng = np.random.default_rng(42)

for i, img_path in enumerate(sample_paths):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取 {img_path}")
        continue

    for j, deg_name in enumerate(deg_names):
        if deg_name == 'original':
            out = img.copy()
        elif deg_name == 'mixed':
            out = degrade_mixed(img, rng=rng)
        else:
            out = DEGRADATIONS[deg_name](img)

        # OpenCV 用 BGR，matplotlib 用 RGB
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        axes[i, j].imshow(out_rgb)
        axes[i, j].axis('off')
        if i == 0:
            axes[i, j].set_title(deg_name, fontsize=11)

plt.tight_layout()
out_path = PROJECT_ROOT / 'sample_degradations.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()

print(f"✓ 样例图保存到: {out_path}")
print(f"  打开看 5 种退化效果是否合理")
print(f"  - lowlight    : 应该明显变暗")
print(f"  - foggy       : 应该有白雾感")
print(f"  - lowcontrast : 应该灰蒙蒙")
print(f"  - noisy       : 应该有雪花点")
print(f"  - mixed       : 应该是 2-3 种叠加")
