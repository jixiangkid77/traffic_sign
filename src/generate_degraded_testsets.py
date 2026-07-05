"""
对 GTSRB 测试集应用 5 种退化，分别保存到独立文件夹。

跑一次约 5-10 分钟。每个测试集 12,630 张图（约 200 MB）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 将 src 添加到 path 以便 import 同级模块
sys.path.insert(0, str(Path(__file__).parent))
from degrade import DEGRADATIONS, degrade_mixed  # noqa: E402

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'

# GTSRB 测试集原始路径（torchvision 下载后的标准位置）
SRC_DIR_CANDIDATES = [
    DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Test' / 'Images',
    DATA_ROOT / 'gtsrb' / 'Final_Test' / 'Images',
]

SRC_DIR = None
for candidate in SRC_DIR_CANDIDATES:
    if candidate.exists():
        SRC_DIR = candidate
        break

if SRC_DIR is None:
    # 兜底：递归搜索任何包含 .ppm 的 Final_Test 目录
    for p in DATA_ROOT.rglob('Final_Test'):
        img_dir = p / 'Images'
        if img_dir.exists() and any(img_dir.glob('*.ppm')):
            SRC_DIR = img_dir
            break

if SRC_DIR is None:
    raise FileNotFoundError(
        f"找不到 GTSRB 测试集图片目录。\n"
        f"已检查的候选路径:\n"
        + "\n".join(f"  - {p}" for p in SRC_DIR_CANDIDATES)
        + f"\n请先在 {PROJECT_ROOT} 下运行: python download_data.py"
    )

img_paths = sorted(SRC_DIR.glob('*.ppm'))
print(f"源目录: {SRC_DIR}")
print(f"找到 {len(img_paths)} 张测试图\n")

if len(img_paths) == 0:
    raise RuntimeError(
        f"目录 {SRC_DIR} 下没有 .ppm 文件。请重新运行 download_data.py。"
    )

# 设全局 seed，保证 mixed 退化可复现
np.random.seed(42)
rng = np.random.default_rng(42)

print(f"开始生成 5 个退化测试集 ...\n")

for deg_name in DEGRADATIONS:
    dst_dir = DATA_ROOT / f'gtsrb_{deg_name}'
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已经生成过（断点续跑用）
    existing = len(list(dst_dir.glob('*.png')))
    if existing == len(img_paths):
        print(f"  [{deg_name}] 已存在 {existing} 张，跳过")
        continue
    elif existing > 0:
        print(f"  [{deg_name}] 已部分存在 {existing} 张，将重新生成")

    for img_path in tqdm(img_paths, desc=f"  生成 {deg_name:12s}", ncols=80):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n  警告：无法读取 {img_path}")
            continue

        if deg_name == 'mixed':
            out = degrade_mixed(img, rng=rng)
        else:
            out = DEGRADATIONS[deg_name](img)

        # 保存为 png（无损压缩、文件名兼容性好）
        dst_path = dst_dir / img_path.name.replace('.ppm', '.png')
        cv2.imwrite(str(dst_path), out)

print()
print("=" * 60)
print("✓ 所有退化测试集生成完毕！")
print("=" * 60)
for deg_name in DEGRADATIONS:
    n = len(list((DATA_ROOT / f'gtsrb_{deg_name}').glob('*.png')))
    print(f"  data/gtsrb_{deg_name}/  ({n} 张图)")
print()
print("下一步：python src/visualize_samples.py")
