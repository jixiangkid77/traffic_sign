"""
Day 4-A: 主结果可视化。

输出两张图（论文 Fig 3 + Fig 4）：
  - Fig3_main_comparison.png      4 方法 × 6 测试集分组柱状图
  - Fig4_confusion_matrices.png   baseline vs va_adaptive 在 mixed 上的混淆矩阵对比

使用：
  python src/visualize_main_results.py

时间：约 1-2 分钟（混淆矩阵需要重新跑 mixed 测试集的推理）
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from enhance import no_enhance, adaptive_enhance  # noqa: E402
from model import build_model  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# ============== 读取主结果 ==============
with open(RESULTS_DIR / 'main_results.json') as f:
    results = json.load(f)
print(f"加载 main_results.json")

# ============== Fig 3: 主对比柱状图 ==============
print("\n生成 Fig 3: 主对比柱状图 ...")

methods = ['baseline', 'fixed_clahe', 'fixed_gamma', 'va_adaptive']
testsets = ['clean', 'lowlight', 'foggy', 'lowcontrast', 'noisy', 'mixed']

display_names = {
    'baseline':    'Baseline (no preprocessing)',
    'fixed_clahe': 'Fixed CLAHE',
    'fixed_gamma': 'Fixed Gamma',
    'va_adaptive': 'VA-Adaptive (Ours)',
}
testset_display = {
    'clean': 'Clean', 'lowlight': 'Low-light', 'foggy': 'Foggy',
    'lowcontrast': 'Low-contrast', 'noisy': 'Noisy', 'mixed': 'Mixed',
}
colors = {
    'baseline':    '#34495E',  # 深蓝灰
    'fixed_clahe': '#3498DB',  # 蓝
    'fixed_gamma': '#27AE60',  # 绿
    'va_adaptive': '#E74C3C',  # 红（突出我们的方法）
}

acc_data = {m: [results[m][t]['top1_acc'] * 100 for t in testsets] for m in methods}

x = np.arange(len(testsets))
width = 0.2
fig, ax = plt.subplots(figsize=(13, 6))

for i, method in enumerate(methods):
    vals = acc_data[method]
    ax.bar(x + i * width, vals, width, label=display_names[method],
           color=colors[method], edgecolor='black', linewidth=0.5)
    for j, v in enumerate(vals):
        ax.text(x[j] + i * width, v + 0.8, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([testset_display[t] for t in testsets], fontsize=11)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('Comparison of Image Enhancement Strategies on GTSRB Test Sets',
             fontsize=13, pad=15)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 105])

plt.tight_layout()
fig3_path = RESULTS_DIR / 'Fig3_main_comparison.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ {fig3_path.name}")

# ============== Fig 4: 混淆矩阵对比 ==============
print("\n生成 Fig 4: 混淆矩阵 ...")
print("（需要重新对 mixed 测试集跑 baseline 和 va_adaptive 推理，约 1-2 分钟）")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = build_model(num_classes=43)
model.load_state_dict(torch.load(MODELS_DIR / 'mbnetv3_baseline.pth',
                                  map_location=device))
model = model.to(device).eval()

# 加载阈值
with open(RESULTS_DIR / 'thresholds.json') as f:
    th = json.load(f)
T1, T2 = th['T1_brightness_low'], th['T2_contrast_low']
T3, T4 = th['T3_edge_low'], th['T4_brightness_high']

# 加载 mixed 测试集 + 标签
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

mixed_dir = DATA_ROOT / 'gtsrb_mixed'
img_paths = sorted(mixed_dir.glob('*.png'))

test_tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
])


def run_inference(preprocess_fn, name):
    """对 mixed 测试集做推理，返回 y_true, y_pred。"""
    all_preds, all_labels = [], []
    batch_imgs, batch_labels = [], []
    BATCH = 256

    for p in tqdm(img_paths, desc=f"  {name:14s}", ncols=80):
        stem = p.stem
        if stem not in labels_map:
            continue
        img_bgr = cv2.imread(str(p))
        img_bgr = preprocess_fn(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = test_tf(Image.fromarray(img_rgb))
        batch_imgs.append(x)
        batch_labels.append(labels_map[stem])

        if len(batch_imgs) >= BATCH:
            xs = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                preds = model(xs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels)
            batch_imgs, batch_labels = [], []

    if batch_imgs:
        xs = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            preds = model(xs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels)

    return np.array(all_labels), np.array(all_preds)


t0 = time.time()
y_true_b, y_pred_b = run_inference(no_enhance, 'baseline')
y_true_v, y_pred_v = run_inference(
    lambda img: adaptive_enhance(img, T1, T2, T3, T4), 'va_adaptive'
)
print(f"  推理耗时 {time.time()-t0:.1f}s\n")

cm_b = confusion_matrix(y_true_b, y_pred_b)
cm_v = confusion_matrix(y_true_v, y_pred_v)

# 行归一化（每行表示该 true class 的预测分布）
cm_b_norm = cm_b / (cm_b.sum(axis=1, keepdims=True) + 1e-9)
cm_v_norm = cm_v / (cm_v.sum(axis=1, keepdims=True) + 1e-9)

acc_b = results['baseline']['mixed']['top1_acc'] * 100
acc_v = results['va_adaptive']['mixed']['top1_acc'] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_b_norm, ax=axes[0], cmap='Blues', cbar=True,
            vmin=0, vmax=1, square=True, xticklabels=5, yticklabels=5)
axes[0].set_title(f'(a) Baseline on Mixed-degradation\nAccuracy = {acc_b:.2f}%',
                   fontsize=12)
axes[0].set_xlabel('Predicted Class', fontsize=11)
axes[0].set_ylabel('True Class', fontsize=11)

sns.heatmap(cm_v_norm, ax=axes[1], cmap='Blues', cbar=True,
            vmin=0, vmax=1, square=True, xticklabels=5, yticklabels=5)
axes[1].set_title(f'(b) VA-Adaptive on Mixed-degradation\nAccuracy = {acc_v:.2f}%',
                   fontsize=12)
axes[1].set_xlabel('Predicted Class', fontsize=11)
axes[1].set_ylabel('True Class', fontsize=11)

plt.tight_layout()
fig4_path = RESULTS_DIR / 'Fig4_confusion_matrices.png'
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ {fig4_path.name}")

# 保存预测结果（便于之后复用）
preds_path = RESULTS_DIR / 'mixed_predictions.json'
with open(preds_path, 'w') as f:
    json.dump({
        'baseline':    {'y_true': y_true_b.tolist(), 'y_pred': y_pred_b.tolist()},
        'va_adaptive': {'y_true': y_true_v.tolist(), 'y_pred': y_pred_v.tolist()},
    }, f)

print(f"\n所有图已保存到: {RESULTS_DIR}")
print("  Fig3_main_comparison.png      ← 论文 Fig 3")
print("  Fig4_confusion_matrices.png   ← 论文 Fig 4")
