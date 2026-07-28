"""
download_cure_tsr.py — 下载 CURE-TSR 真实退化数据子集

CURE-TSR 数据集结构：
  CURE-TSR/
    Real_Train/                  # 真实采集的训练图像
      ChallengeFree/             # 干净图像
      Rain-1/, Rain-2/, ...      # 5 个 severity levels
      Snow-1/, Snow-2/, ...
      Haze-1/, Haze-2/, ...
      LensBlur-1/, ...
      Exposure-1/, ...
      etc.
    Real_Test/                   # 真实采集的测试图像（结构同上）

每个 challenge 子文件夹里图片命名：
  XX_YY_ZZZZZ.bmp
  XX = challenge type ID (01–14)
  YY = severity level (01–05) 
  ZZZZZ = sample index

类别映射：
  01_*.bmp = speed_limit
  02_*.bmp = goods_vehicles
  ... (14 个类)

本脚本下载 Real_Test 子集（约 2 GB），并按 challenge type + severity 整理路径供后续评估。

运行：
  conda activate pcm_sim
  python download_cure_tsr.py
"""

import os
import sys
import urllib.request
import zipfile
import json
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
CURE_TSR_DIR = PROJECT_ROOT / "datasets" / "CURE-TSR"

# CURE-TSR 在 OLIVES 实验室页面的下载链接
# 注意：CURE-TSR 实际数据需从其 GitHub README 中链接到 Google Drive 下载
# 这里给出推荐做法
CURE_TSR_README_URL = "https://github.com/olivesgatech/CURE-TSR"

# CURE-TSR 14 类类别映射（按官方文档）
CURE_CLASSES = {
    1:  "speed_limit",
    2:  "goods_vehicles",
    3:  "no_overtaking",
    4:  "no_stopping",
    5:  "no_parking",
    6:  "stop",
    7:  "bicycle",
    8:  "hump",
    9:  "no_left",
    10: "no_right",
    11: "priority_to",
    12: "no_entry",
    13: "yield",
    14: "parking",
}

# 14 个 challenge types（除 ChallengeFree 外）
CURE_CHALLENGES = [
    "ChallengeFree",   # severity 0 (干净)
    "Decolorization",
    "LensBlur",
    "CodecError",
    "DarkenChallenge",
    "DirtyLens",
    "Exposure",
    "GaussianBlur",
    "Haze",            # 雾
    "Noise",           # 噪声 — 直接对应你 Comment 4 噪声分支
    "Rain",            # 雨
    "Shadow",
    "Snow",            # 雪
    "GrayscaleStripes",
]

# 跟你 paper 5 个合成退化最对应的 4 个：
PAPER_RELEVANT_CHALLENGES = ["DarkenChallenge", "Haze", "Noise", "Rain"]


# ============================================================
# Step 1: 创建目录 + 输出说明
# ============================================================
def setup_directories():
    CURE_TSR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[✓] CURE-TSR 目标目录创建：{CURE_TSR_DIR}")


def print_download_instructions():
    print("\n" + "=" * 70)
    print("  CURE-TSR 数据下载说明（手动步骤，无自动下载）")
    print("=" * 70)
    print(f"""
原因：CURE-TSR 数据集托管在 Google Drive 上，作者要求手动确认下载，
没有公开的直接 wget 链接。

请按以下步骤手动下载：

1. 访问 GitHub README：
   {CURE_TSR_README_URL}

2. 在 README 中找到 "Real World Data" 或 "Dataset Download" 部分。
   通常会有一个 Google Drive 链接，例如：
   https://drive.google.com/drive/folders/...

3. 下载需要的子集（推荐路径如下，节省时间）：
   
   仅下载 Real_Test 即可（约 1.5 GB），不要下载 Real_Train（10+ GB）。
   
   Real_Test/ 包含：
     ChallengeFree/                    # 必下（用作 reference clean）
     DarkenChallenge-{{1..5}}/          # 必下（对应你的 lowlight）
     Haze-{{1..5}}/                     # 必下（对应你的 foggy）
     Noise-{{1..5}}/                    # 必下（对应你的 noisy）
     Rain-{{1..5}}/                     # 推荐（real-world 卖点）
     Snow-{{1..5}}/                     # 推荐
     Exposure-{{1..5}}/                 # 推荐
     LensBlur-{{1..5}}/                 # 可选

4. 解压到：
   {CURE_TSR_DIR}

5. 解压后目录结构应该是：
   {CURE_TSR_DIR}/
     Real_Test/
       ChallengeFree/
         01_00_00001.bmp
         01_00_00002.bmp
         ...
       DarkenChallenge-1/
         01_01_00001.bmp
         ...
       Haze-1/
       Noise-1/
       Rain-1/
       ...

6. 下载完成后再次运行本脚本，会自动检测和验证：
   python download_cure_tsr.py
""")


# ============================================================
# Step 2: 验证下载（如果已下载）
# ============================================================
def verify_dataset():
    real_test = CURE_TSR_DIR / "Real_Test"
    if not real_test.exists():
        print(f"[!] {real_test} 不存在 — 数据尚未下载")
        return False
    
    # 检查必需的 challenge folders
    required_minimal = ["ChallengeFree"]
    required_minimal += [f"{c}-{s}" for c in PAPER_RELEVANT_CHALLENGES for s in range(1, 6)]
    
    missing = []
    for sub in required_minimal:
        path = real_test / sub
        if not path.exists():
            missing.append(sub)
    
    if missing:
        print(f"[!] 以下子文件夹缺失（共 {len(missing)} 个）:")
        for m in missing[:5]:
            print(f"    - {m}")
        if len(missing) > 5:
            print(f"    ... 还有 {len(missing) - 5} 个")
        return False
    
    # 统计样本数
    print(f"\n[✓] CURE-TSR Real_Test 已下载至 {real_test}")
    print(f"\n各子文件夹样本统计：")
    print("-" * 60)
    
    for sub in sorted(os.listdir(real_test)):
        sub_path = real_test / sub
        if sub_path.is_dir():
            n_imgs = len([f for f in os.listdir(sub_path) 
                         if f.endswith(('.bmp', '.png', '.jpg'))])
            marker = "★" if sub.split('-')[0] in PAPER_RELEVANT_CHALLENGES or sub == "ChallengeFree" else " "
            print(f"  {marker} {sub:30s} {n_imgs:6d} images")
    
    print("-" * 60)
    print("  ★ 标记 = 你 paper 直接用得上的 challenge type")
    return True


# ============================================================
# Step 3: 输出元数据 JSON
# ============================================================
def write_metadata():
    real_test = CURE_TSR_DIR / "Real_Test"
    if not real_test.exists():
        return
    
    metadata = {
        "dataset": "CURE-TSR Real_Test",
        "n_classes": 14,
        "classes": CURE_CLASSES,
        "filename_format": "XX_YY_ZZZZZ.bmp where XX=class(1-14), YY=severity(0-5)",
        "challenges": {},
        "paper_relevant": PAPER_RELEVANT_CHALLENGES,
    }
    
    for sub in sorted(os.listdir(real_test)):
        sub_path = real_test / sub
        if sub_path.is_dir():
            files = [f for f in os.listdir(sub_path) if f.endswith(('.bmp', '.png', '.jpg'))]
            metadata["challenges"][sub] = {
                "n_images": len(files),
                "path": str(sub_path),
            }
    
    out = CURE_TSR_DIR / "metadata.json"
    with open(out, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[✓] 元数据写入：{out}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    setup_directories()
    
    if verify_dataset():
        write_metadata()
        print("\n[✓] CURE-TSR 数据准备完成，可以开始评估实验。")
        print("    下一步：运行 evaluate_cure_tsr_qualitative.py 做快速 sanity check")
    else:
        print_download_instructions()
        sys.exit(1)
