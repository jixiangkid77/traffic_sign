"""
Day 4-C: Backbone Ablation Evaluation。

评估 3 个骨干 × 2 方法 × 6 测试集 = 36 个 cell：
  Backbones: CompactCNN / ShuffleNetV2 (x0.5) / MobileNetV2 (w=0.5)
  Methods:   no_enhance (baseline) / va_adaptive (Ours)
  Testsets:  clean / lowlight / foggy / lowcontrast / noisy / mixed

为节省时间，跳过 fixed_clahe 和 fixed_gamma（已在主表中证明它们的 trade-off
在不同骨干下是普适的）。

使用：
  python src/evaluate_backbone_ablation.py

时间：约 12 分钟
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5, mobilenet_v2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from enhance import no_enhance, adaptive_enhance  # noqa: E402
from model import build_model  # noqa: E402


PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'


# ============== 三个骨干的构造函数 ==============
def build_shufflenet_v2_x0_5(num_classes=43):
    model = shufflenet_v2_x0_5(weights=None, num_classes=num_classes)
    model.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    return model


def build_mobilenet_v2_w0_5(num_classes=43):
    model = mobilenet_v2(weights=None, num_classes=num_classes, width_mult=0.5)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        first_conv.in_channels, first_conv.out_channels,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    return model


BACKBONES = [
    ('CompactCNN',   build_model,                 'mbnetv3_baseline.pth'),
    ('ShuffleNetV2', build_shufflenet_v2_x0_5,    'shufflenet_baseline.pth'),
    ('MobileNetV2',  build_mobilenet_v2_w0_5,     'mobilenet_baseline.pth'),
]

TESTSETS = ['clean', 'lowlight', 'foggy', 'lowcontrast', 'noisy', 'mixed']


# ============== 数据集类 ==============
class GTSRBTestDataset(Dataset):
    def __init__(self, image_dir, labels, preprocess_fn, transform):
        self.samples = []
        for p in sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.ppm')):
            stem = p.stem
            if stem in labels:
                self.samples.append((p, labels[stem]))
        self.preprocess_fn = preprocess_fn
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(str(path))
        img_bgr = self.preprocess_fn(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(Image.fromarray(img_rgb))
        return x, label


def evaluate_one_cell(model, image_dir, labels, preprocess_fn, transform, device):
    """评估一个 cell（骨干 × 方法 × 测试集），返回 acc 和 f1。"""
    dataset = GTSRBTestDataset(image_dir, labels, preprocess_fn, transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    return acc, f1, len(all_labels)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Backbone Ablation Evaluation")
    print("=" * 70)
    print(f"设备: {device}")

    # 加载阈值
    with open(RESULTS_DIR / 'thresholds.json') as f:
        th = json.load(f)
    T1, T2 = th['T1_brightness_low'], th['T2_contrast_low']
    T3, T4 = th['T3_edge_low'], th['T4_brightness_high']
    print(f"阈值: T1={T1:.4f} T2={T2:.4f} T3={T3:.4f} T4={T4:.4f}")

    # 加载标签
    csv_candidates = [
        DATA_ROOT / 'gtsrb' / 'GT-final_test.csv',
        DATA_ROOT / 'gtsrb' / 'Final_Test' / 'GT-final_test.csv',
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        for p in DATA_ROOT.rglob('GT-final_test.csv'):
            csv_path = p
            break
    labels_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            stem = row['Filename'].rsplit('.', 1)[0]
            labels_map[stem] = int(row['ClassId'])
    print(f"标签: {len(labels_map)} 个")

    test_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
    ])

    methods = {
        'baseline':    no_enhance,
        'va_adaptive': lambda img: adaptive_enhance(img, T1, T2, T3, T4),
    }

    testset_dirs = {
        'clean':       DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Test' / 'Images',
        'lowlight':    DATA_ROOT / 'gtsrb_lowlight',
        'foggy':       DATA_ROOT / 'gtsrb_foggy',
        'lowcontrast': DATA_ROOT / 'gtsrb_lowcontrast',
        'noisy':       DATA_ROOT / 'gtsrb_noisy',
        'mixed':       DATA_ROOT / 'gtsrb_mixed',
    }
    # 兼容 clean 在不同位置
    if not testset_dirs['clean'].exists():
        for p in DATA_ROOT.rglob('Final_Test'):
            if p.is_dir():
                testset_dirs['clean'] = p / 'Images'
                break

    results = {}

    total_cells = len(BACKBONES) * len(methods) * len(TESTSETS)
    cell_idx = 0
    t0 = time.time()

    for backbone_name, builder, weight_file in BACKBONES:
        weight_path = MODELS_DIR / weight_file
        if not weight_path.exists():
            print(f"\n✗ 跳过 {backbone_name}：找不到 {weight_path}")
            continue

        print(f"\n加载 {backbone_name}: {weight_path.name}")
        model = builder(num_classes=43)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device).eval()

        results[backbone_name] = {}
        for method_name, prep_fn in methods.items():
            results[backbone_name][method_name] = {}
            for testset in TESTSETS:
                cell_idx += 1
                img_dir = testset_dirs[testset]
                desc = f"[{cell_idx:2d}/{total_cells}] {backbone_name:14s} {method_name:12s} {testset}"
                print(f"  {desc}", end=' ', flush=True)
                acc, f1, n = evaluate_one_cell(
                    model, img_dir, labels_map, prep_fn, test_tf, device,
                )
                print(f"→ acc={acc*100:.2f}%  f1={f1:.4f}  (N={n})")
                results[backbone_name][method_name][testset] = {
                    'top1_acc': acc, 'macro_f1': f1, 'n_samples': n,
                }

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")

    # ============== 保存 + 打印汇总表 ==============
    with open(RESULTS_DIR / 'backbone_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ 保存到: {RESULTS_DIR / 'backbone_ablation.json'}")

    # 汇总表 + 论文要的提升数字
    print("\n" + "=" * 70)
    print("Backbone Ablation 汇总（论文 Table 2 候选）")
    print("=" * 70)
    print(f"\n{'Backbone':<14} {'Method':<12} {'Clean':>7} "
          f"{'L-light':>7} {'Foggy':>7} {'L-cont':>7} {'Noisy':>7} {'Mixed':>7}")
    print("-" * 70)

    csv_lines = ["Backbone,Method,Clean,Lowlight,Foggy,Lowcontrast,Noisy,Mixed,DegradedAvg"]
    for bb in results:
        for m in results[bb]:
            row = f"{bb:<14} {m:<12}"
            degraded_accs = []
            csv_row = [bb, m]
            for ts in TESTSETS:
                acc = results[bb][m][ts]['top1_acc'] * 100
                row += f" {acc:>6.2f}%"
                csv_row.append(f"{acc:.2f}")
                if ts != 'clean':
                    degraded_accs.append(acc)
            avg_deg = np.mean(degraded_accs)
            row += f"  | DegAvg={avg_deg:.2f}%"
            csv_row.append(f"{avg_deg:.2f}")
            csv_lines.append(",".join(csv_row))
            print(row)
        print()

    with open(RESULTS_DIR / 'backbone_ablation.csv', 'w') as f:
        f.write("\n".join(csv_lines))
    print(f"✓ CSV: {RESULTS_DIR / 'backbone_ablation.csv'}")

    # 关键提升数字
    print("\n" + "=" * 70)
    print("关键提升幅度（va_adaptive vs baseline 在退化场景平均）")
    print("=" * 70)
    for bb in results:
        if 'baseline' in results[bb] and 'va_adaptive' in results[bb]:
            base_avg = np.mean([results[bb]['baseline'][ts]['top1_acc'] * 100
                                 for ts in TESTSETS if ts != 'clean'])
            va_avg = np.mean([results[bb]['va_adaptive'][ts]['top1_acc'] * 100
                              for ts in TESTSETS if ts != 'clean'])
            print(f"  {bb:<14}: baseline={base_avg:.2f}%  "
                  f"va_adaptive={va_avg:.2f}%  "
                  f"Δ={va_avg-base_avg:+.2f} pp")


if __name__ == '__main__':
    main()
