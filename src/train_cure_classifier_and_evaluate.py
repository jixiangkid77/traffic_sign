"""
train_cure_classifier_and_evaluate.py — CURE-TSR 14 类分类器 + 定量评估

仅在 evaluate_cure_tsr_qualitative.py 通过 sanity check 后再运行。

策略：
  1. 用 GTSRB 训好的 CompactCNN 作为 feature extractor（freeze 前 4 个 conv block）
  2. 重新训练最后的分类头（43 → 14 类）
  3. 训练数据：CURE-TSR Real_Train/ChallengeFree/ 上的干净图（仅干净，不用 challenge）
  4. 测试：4 种 challenge × 5 severity = 20 个 testset
  5. 4 种方法：baseline / fixed CLAHE / fixed gamma / fixed stretch / VA-Adaptive
  6. 报 top-1 + macro-F1 + per-class accuracy

输出：
  - cure_tsr_main_results.json — 完整数字
  - cure_tsr_main_results.csv — 表格
  - cure_tsr_intensity_curves.png — 5 levels × 5 methods 折线图

工作量：
  - 训练 last layer：~30 分钟（CPU）
  - 完整评估：~2 小时
  
运行：
  conda activate pcm_sim
  python train_cure_classifier_and_evaluate.py
"""

import os
import sys
import json
import csv
import re
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\Project\traffic_sign\src")
from model import CompactCNN
from enhance import apply_clahe, apply_gamma, adaptive_enhance  # noqa


# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
CURE_TSR_DIR = PROJECT_ROOT / "datasets" / "CURE-TSR"
OUTPUT_DIR = PROJECT_ROOT / "outputs_cure_tsr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GTSRB_MODEL_PATH = PROJECT_ROOT / "models" / "mbnetv3_baseline.pth"
CURE_MODEL_PATH = PROJECT_ROOT / "models" / "cure_classifier_14cls.pth"

CHALLENGES_AND_SEVERITY = ["DarkenChallenge", "Haze", "Noise", "Rain"]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

INPUT_SIZE = 32  # 与 GTSRB 训练一致
N_CURE_CLASSES = 14

# CURE-TSR 文件名格式：XX_YY_ZZZZZ.bmp
# XX = 类别 (1-14)
FILENAME_PATTERN = re.compile(r'(\d{2})_(\d{2})_(\d{5})\.(bmp|png|jpg)')

# GTSRB 标准化（与训练时一致）
GTSRB_MEAN = [0.3401, 0.3120, 0.3212]
GTSRB_STD = [0.2725, 0.2609, 0.2669]


# ============================================================
# Dataset
# ============================================================
class CureTSRDataset(Dataset):
    def __init__(self, folder, enhancement_fn=None, transform=None):
        """
        folder: 包含 XX_YY_ZZZZZ.bmp 的目录
        enhancement_fn: 输入 BGR uint8，返回 BGR uint8（preprocessing）
        transform: torchvision transform，作用在 enhancement 之后
        """
        self.folder = Path(folder)
        self.enhancement_fn = enhancement_fn
        self.transform = transform
        
        self.samples = []
        for f in os.listdir(self.folder):
            m = FILENAME_PATTERN.match(f)
            if m:
                cls = int(m.group(1)) - 1  # 0-indexed
                self.samples.append((f, cls))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = cv2.imread(str(self.folder / fname))
        if img is None:
            return None
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        
        if self.enhancement_fn is not None:
            img = self.enhancement_fn(img)
        
        # BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        return img_tensor, label


# ============================================================
# 模型：在 CompactCNN 上替换最后分类层
# ============================================================
class CureClassifier(nn.Module):
    def __init__(self, gtsrb_pretrained_path, n_classes=14, freeze_backbone=True):
        super().__init__()
        
        # 加载 GTSRB 预训练模型
        gtsrb_model = CompactCNN(num_classes=43)
        ckpt = torch.load(gtsrb_pretrained_path, map_location="cpu")
        if "model_state" in ckpt:
            gtsrb_model.load_state_dict(ckpt["model_state"])
        else:
            gtsrb_model.load_state_dict(ckpt)
        
        # 取除最后分类层之外的所有部分作为 backbone
        # 注意：你的 CompactCNN 结构需要这里能 attribute 访问 fc / classifier
        # 假设最后是 self.fc 或 self.classifier，根据实际改一下
        
        # 通用做法：找到最后一个 Linear layer
        all_modules = list(gtsrb_model.children())
        # 最后一个 Linear 之前的所有当作 backbone
        last_linear_idx = -1
        for i, m in enumerate(all_modules):
            if isinstance(m, nn.Linear):
                last_linear_idx = i
        
        if last_linear_idx == -1:
            # 如果是 nn.Sequential 嵌套，需要更复杂查找
            # 简单起见：用 GTSRB 模型完整 forward 到分类前
            self.backbone = gtsrb_model
            # 替换 fc 输出层（需要根据你 CompactCNN 实际属性名调整）
            if hasattr(gtsrb_model, 'fc'):
                in_features = gtsrb_model.fc.in_features
                self.backbone.fc = nn.Linear(in_features, n_classes)
            elif hasattr(gtsrb_model, 'classifier'):
                in_features = gtsrb_model.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, n_classes)
            else:
                raise AttributeError("找不到最后分类层。请根据 CompactCNN 实际结构修改本脚本")
        else:
            self.backbone = nn.Sequential(*all_modules[:last_linear_idx])
            in_features = all_modules[last_linear_idx].in_features
            self.classifier = nn.Linear(in_features, n_classes)
        
        # 冻结 backbone（feature extractor mode）
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        # 假设是 sequential 模式
        if hasattr(self, 'classifier'):
            x = self.backbone(x)
            if x.ndim > 2:
                x = x.flatten(1)
            return self.classifier(x)
        else:
            return self.backbone(x)


# ============================================================
# 训练
# ============================================================
def train_cure_classifier():
    print("Step 1: 训练 CURE-TSR 14 类分类器（freeze backbone）...")
    
    train_folder = CURE_TSR_DIR / "Real_Train" / "ChallengeFree"
    if not train_folder.exists():
        print(f"[!] 找不到 CURE-TSR Real_Train. 改用 Real_Test/ChallengeFree 做 train+val split")
        train_folder = CURE_TSR_DIR / "Real_Test" / "ChallengeFree"
    
    if not train_folder.exists():
        print(f"[!] {train_folder} 不存在。无法训练。")
        sys.exit(1)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])
    
    full_dataset = CureTSRDataset(train_folder, enhancement_fn=None, transform=transform)
    print(f"  Train+val total: {len(full_dataset)} images")
    
    # 90/10 split
    n_train = int(len(full_dataset) * 0.9)
    n_val = len(full_dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = CureClassifier(GTSRB_MODEL_PATH, n_classes=N_CURE_CLASSES, freeze_backbone=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 只训练分类层参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    
    # 训练 10 epoch（freeze backbone 收敛快）
    n_epochs = 10
    best_val_acc = 0
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_ds)
        
        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
        val_acc = val_correct / n_val * 100
        
        print(f"  Epoch {epoch+1:2d}/{n_epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CURE_MODEL_PATH)
    
    print(f"\n[✓] Best val_acc: {best_val_acc:.2f}%")
    print(f"[✓] Model saved: {CURE_MODEL_PATH}")
    return best_val_acc


# ============================================================
# 评估
# ============================================================
def evaluate_method(model, device, folder, enhancement_fn, name):
    """对一个 challenge folder 跑一个 method，返回 acc + macro_f1"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])
    ds = CureTSRDataset(folder, enhancement_fn=enhancement_fn, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(y.numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return {"acc": acc, "macro_f1": macro_f1, "n": len(ds)}


def fixed_clahe_fn(img):
    return apply_clahe(img, clip_limit=3.0)


def fixed_gamma_fn(img):
    return apply_gamma(img, gamma=0.5)


def fixed_stretch_fn(img):
    """新增的 fixed stretch baseline（响应 Comment 1 的对比完整性）"""
    img_f = img.astype(np.float32) / 255.0
    out_f = np.clip((img_f - 0.5) * 1.5 + 0.5, 0, 1)
    return (out_f * 255).astype(np.uint8)


def va_adaptive_fn(img):
    """与 paper 一致的 VA-Adaptive"""
    return adaptive_enhance(img)


def evaluate_all():
    print("\nStep 2: 完整定量评估...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CureClassifier(GTSRB_MODEL_PATH, n_classes=N_CURE_CLASSES, freeze_backbone=True)
    model.load_state_dict(torch.load(CURE_MODEL_PATH, map_location=device))
    model = model.to(device)
    
    methods = {
        "baseline": None,
        "fixed_clahe": fixed_clahe_fn,
        "fixed_gamma": fixed_gamma_fn,
        "fixed_stretch": fixed_stretch_fn,
        "va_adaptive": va_adaptive_fn,
    }
    
    # 评估 ChallengeFree（reference）
    print("\nReference (clean):")
    cf_results = {}
    cf_folder = CURE_TSR_DIR / "Real_Test" / "ChallengeFree"
    for method_name, fn in methods.items():
        r = evaluate_method(model, device, cf_folder, fn, method_name)
        cf_results[method_name] = r
        print(f"  {method_name:15s} acc={r['acc']:.2f}%  macro_f1={r['macro_f1']:.2f}%")
    
    # 评估每个 challenge × severity
    all_results = {"clean": cf_results, "challenges": {}}
    
    for challenge in CHALLENGES_AND_SEVERITY:
        print(f"\n{challenge}:")
        all_results["challenges"][challenge] = {}
        for severity in SEVERITY_LEVELS:
            folder = CURE_TSR_DIR / "Real_Test" / f"{challenge}-{severity}"
            if not folder.exists():
                continue
            
            results_per_severity = {}
            for method_name, fn in methods.items():
                r = evaluate_method(model, device, folder, fn, method_name)
                results_per_severity[method_name] = r
            
            all_results["challenges"][challenge][severity] = results_per_severity
            
            # 打印简明结果
            line = f"  Severity {severity}: "
            for method_name in methods:
                acc = results_per_severity[method_name]["acc"]
                line += f"{method_name[:8]}={acc:.1f}  "
            print(line)
    
    # 写 JSON
    out_json = OUTPUT_DIR / "cure_tsr_main_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] 完整结果：{out_json}")
    
    # 写 CSV
    out_csv = OUTPUT_DIR / "cure_tsr_main_results.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["challenge", "severity", "method", "acc", "macro_f1", "n"])
        for method_name in methods:
            r = cf_results[method_name]
            writer.writerow(["clean", 0, method_name, r["acc"], r["macro_f1"], r["n"]])
        for challenge, by_severity in all_results["challenges"].items():
            for severity, by_method in by_severity.items():
                for method_name, r in by_method.items():
                    writer.writerow([challenge, severity, method_name, 
                                   r["acc"], r["macro_f1"], r["n"]])
    print(f"[✓] CSV 表：{out_csv}")
    
    return all_results


# ============================================================
# 强度曲线（响应 Comment 5 "performance curves under different intensities"）
# ============================================================
def plot_intensity_curves(results):
    print("\nStep 3: 绘制强度曲线...")
    
    methods = ["baseline", "fixed_clahe", "fixed_gamma", "fixed_stretch", "va_adaptive"]
    colors = ['#666', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax, challenge in zip(axes, CHALLENGES_AND_SEVERITY):
        if challenge not in results["challenges"]:
            continue
        
        for method, color in zip(methods, colors):
            xs, ys = [], []
            for severity in SEVERITY_LEVELS:
                if severity in results["challenges"][challenge]:
                    xs.append(severity)
                    ys.append(results["challenges"][challenge][severity][method]["acc"])
            
            ax.plot(xs, ys, '-o', color=color, label=method, linewidth=1.5, markersize=5)
        
        ax.set_xlabel("Severity level")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(challenge)
        ax.grid(alpha=0.3)
        ax.set_xticks(SEVERITY_LEVELS)
        if challenge == CHALLENGES_AND_SEVERITY[0]:
            ax.legend(loc='lower left', fontsize=8)
    
    plt.tight_layout()
    out_png = OUTPUT_DIR / "cure_tsr_intensity_curves.png"
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[✓] 强度曲线：{out_png}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if not CURE_MODEL_PATH.exists():
        train_cure_classifier()
    else:
        print(f"[i] 已找到训练好的模型：{CURE_MODEL_PATH}")
        print(f"    如需重训，删除此文件重跑")
    
    results = evaluate_all()
    plot_intensity_curves(results)
    
    print("\n" + "=" * 70)
    print("  CURE-TSR 评估完成")
    print("=" * 70)
    print("  输出文件：")
    print(f"    - {OUTPUT_DIR}/cure_tsr_main_results.json")
    print(f"    - {OUTPUT_DIR}/cure_tsr_main_results.csv")
    print(f"    - {OUTPUT_DIR}/cure_tsr_intensity_curves.png")
    print("\n  下一步：")
    print("    把 va_adaptive 的 acc + macro_f1 整理成 paper Table 3")
    print("    （Section IV.F: Cross-Dataset Real-World Evaluation）")
