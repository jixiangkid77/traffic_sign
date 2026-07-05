"""
Day 3 Step 3: 主对比实验 - 4 种方法 × 6 个测试集 = 24 个评估。

方法：
  1. baseline    : 不预处理
  2. fixed_clahe : 全局 CLAHE 增强
  3. fixed_gamma : 全局 Gamma 校正
  4. va_adaptive : 能见度感知自适应增强（本文方法）

测试集：
  1. clean       : 原始 GTSRB 测试集
  2. lowlight    : 低光照退化
  3. foggy       : 雾化退化
  4. lowcontrast : 低对比度退化
  5. noisy       : 噪声退化
  6. mixed       : 混合退化

输出：
  results/main_results.json    完整结果字典
  results/main_results.csv     论文用表格

CPU 时间预估：每个评估 1-2 分钟（CompactCNN 推理快），总共 25-50 分钟
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from enhance import (  # noqa: E402
    no_enhance, apply_clahe, apply_gamma, adaptive_enhance,
)
from model import build_model  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ============== 配置 ==============
IMAGE_SIZE = 32        # ← 必须和训练时一致
BATCH_SIZE = 256
NUM_CLASSES = 43

test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # 必须和训练时同一个 normalize
    transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
])


def load_test_labels():
    """从 GT-final_test.csv 加载测试集 filename → ClassId 映射。"""
    csv_candidates = [
        DATA_ROOT / 'gtsrb' / 'GT-final_test.csv',
        DATA_ROOT / 'gtsrb' / 'Final_Test' / 'GT-final_test.csv',
        DATA_ROOT / 'gtsrb' / 'GTSRB' / 'GT-final_test.csv',
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        for p in DATA_ROOT.rglob('GT-final_test.csv'):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError("找不到 GT-final_test.csv")

    print(f"加载标签: {csv_path}")
    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            stem = row['Filename'].rsplit('.', 1)[0]
            labels[stem] = int(row['ClassId'])
    return labels


class GTSRBTestDataset(Dataset):
    """从指定目录加载测试集，应用预处理函数 + 标准 transforms。"""

    def __init__(self, image_dir, labels, preprocess_fn, transform):
        self.image_dir = Path(image_dir)
        self.labels = labels
        self.preprocess_fn = preprocess_fn
        self.transform = transform

        all_files = (
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.ppm'))
        )
        self.samples = []
        for p in all_files:
            stem = p.stem
            if stem in self.labels:
                self.samples.append((p, self.labels[stem]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"目录 {self.image_dir} 下没有匹配 label 的图片。"
                f"找到 {len(all_files)} 张图，标签 {len(self.labels)} 个。"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(str(path))
        img_bgr = self.preprocess_fn(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        x = self.transform(img_pil)
        return x, label


def evaluate(model, image_dir, labels, preprocess_fn, device):
    """在指定测试集上评估，应用指定预处理函数。"""
    dataset = GTSRBTestDataset(image_dir, labels, preprocess_fn, test_tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)

    model.eval()
    all_preds, all_labels = [], []

    t0 = time.time()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  推理", ncols=80, leave=False):
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    elapsed = time.time() - t0

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'top1_acc': float(acc),
        'macro_f1': float(f1),
        'n_samples': len(all_labels),
        'total_time_s': float(elapsed),
        'time_per_img_ms': float(elapsed / len(all_labels) * 1000),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Day 3 Step 3: Main Comparison Experiment")
    print("=" * 70)
    print(f"设备: {device}\n")

    # 加载阈值
    th_path = RESULTS_DIR / 'thresholds.json'
    if not th_path.exists():
        raise FileNotFoundError(
            f"找不到阈值文件 {th_path}\n请先运行 calibrate_thresholds.py"
        )
    with open(th_path) as f:
        th = json.load(f)
    T1, T2 = th['T1_brightness_low'], th['T2_contrast_low']
    T3, T4 = th['T3_edge_low'], th['T4_brightness_high']
    print(f"加载阈值: T1={T1:.4f} T2={T2:.4f} T3={T3:.4f} T4={T4:.4f}\n")

    # 加载模型
    model_path = MODELS_DIR / 'mbnetv3_baseline.pth'
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型 {model_path}\n请先运行 train_baseline.py"
        )
    print(f"加载模型: {model_path}")
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 加载测试集标签
    labels = load_test_labels()
    print(f"测试集标签: {len(labels)} 个\n")

    # ============== 定义所有方法和测试集 ==============
    METHODS = {
        'baseline':    no_enhance,
        'fixed_clahe': apply_clahe,
        'fixed_gamma': lambda img: apply_gamma(img, gamma=0.5),
        'va_adaptive': lambda img: adaptive_enhance(img, T1, T2, T3, T4),
    }

    clean_dir_candidates = [
        DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Test' / 'Images',
        DATA_ROOT / 'gtsrb' / 'Final_Test' / 'Images',
    ]
    clean_dir = next((p for p in clean_dir_candidates if p.exists()), None)
    if clean_dir is None:
        for p in DATA_ROOT.rglob('Final_Test'):
            if (p / 'Images').exists():
                clean_dir = p / 'Images'
                break
    if clean_dir is None:
        raise FileNotFoundError("找不到 clean 测试集目录")

    TESTSETS = {
        'clean':       clean_dir,
        'lowlight':    DATA_ROOT / 'gtsrb_lowlight',
        'foggy':       DATA_ROOT / 'gtsrb_foggy',
        'lowcontrast': DATA_ROOT / 'gtsrb_lowcontrast',
        'noisy':       DATA_ROOT / 'gtsrb_noisy',
        'mixed':       DATA_ROOT / 'gtsrb_mixed',
    }

    print(f"开始主对比实验: {len(METHODS)} 方法 × {len(TESTSETS)} 测试集 "
          f"= {len(METHODS)*len(TESTSETS)} 个组合\n")
    print("=" * 70)

    results = {}
    total_t0 = time.time()
    n_done = 0
    n_total = len(METHODS) * len(TESTSETS)

    for method_name, preprocess_fn in METHODS.items():
        results[method_name] = {}
        for testset_name, testset_dir in TESTSETS.items():
            n_done += 1
            print(f"[{n_done:2d}/{n_total}] {method_name:12s} × {testset_name:12s}")
            metrics = evaluate(model, testset_dir, labels, preprocess_fn, device)
            results[method_name][testset_name] = metrics
            print(f"        → acc={metrics['top1_acc']*100:5.2f}%  "
                  f"f1={metrics['macro_f1']:.4f}  "
                  f"time={metrics['total_time_s']:.1f}s  "
                  f"({metrics['time_per_img_ms']:.2f} ms/img)\n")

            with open(RESULTS_DIR / 'main_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - total_t0

    print("=" * 70)
    print(f"✓ 全部完成！总耗时: {total_time/60:.1f} 分钟")
    print("=" * 70)

    print("\n## Top-1 Accuracy 对比表\n")
    header = ["Method"] + list(TESTSETS.keys())
    print("| " + " | ".join(f"{h:12s}" for h in header) + " |")
    print("|" + "|".join("-" * 14 for _ in header) + "|")
    for method_name in METHODS:
        row = [method_name]
        for testset_name in TESTSETS:
            acc = results[method_name][testset_name]['top1_acc']
            row.append(f"{acc*100:.2f}%")
        print("| " + " | ".join(f"{c:12s}" for c in row) + " |")

    csv_path = RESULTS_DIR / 'main_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'testset', 'top1_acc', 'macro_f1',
                         'n_samples', 'total_time_s', 'time_per_img_ms'])
        for method_name in METHODS:
            for testset_name in TESTSETS:
                m = results[method_name][testset_name]
                writer.writerow([
                    method_name, testset_name,
                    m['top1_acc'], m['macro_f1'],
                    m['n_samples'], m['total_time_s'], m['time_per_img_ms'],
                ])
    print(f"\nCSV 表格保存到: {csv_path}")
    print(f"JSON 结果保存到: {RESULTS_DIR / 'main_results.json'}")


if __name__ == '__main__':
    main()
