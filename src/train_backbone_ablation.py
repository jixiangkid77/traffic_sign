"""
Day 4-C: Backbone Ablation Training。

训练 2 个轻量骨干（GTSRB 32x32，10 epoch each）：
  - ShuffleNetV2 (x0.5)  — 约 340K 参数
  - MobileNetV2  (w=0.5) — 约 700K 参数

两者都改了第一层 conv stride（2→1），适配 32x32 小输入。

使用：
  python src/train_backbone_ablation.py

时间：CPU 上约 70-90 分钟（两个模型串行训练）
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import GTSRB
from torchvision.models import shufflenet_v2_x0_5, mobilenet_v2
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'


def build_shufflenet_v2_x0_5(num_classes=43):
    """ShuffleNetV2 x0.5，适配 32x32 输入。"""
    model = shufflenet_v2_x0_5(weights=None, num_classes=num_classes)
    # 第一层 conv 从 stride 2 改 stride 1（保留更多空间信息）
    model.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    # 删掉首个 maxpool 的下采样（stride 2→1）
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    return model


def build_mobilenet_v2_w0_5(num_classes=43):
    """MobileNetV2 width_mult=0.5，适配 32x32 输入。"""
    model = mobilenet_v2(weights=None, num_classes=num_classes, width_mult=0.5)
    # 第一层 conv stride 2→1
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        first_conv.in_channels, first_conv.out_channels,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    return model


def get_train_labels(train_set):
    if hasattr(train_set, '_samples'):
        return [label for _, label in train_set._samples]
    elif hasattr(train_set, 'samples'):
        return [label for _, label in train_set.samples]
    return [train_set[i][1] for i in range(len(train_set))]


def evaluate_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def train_one_backbone(name, builder, save_path,
                       num_epochs=10, batch_size=256, lr_max=2e-3):
    """训一个骨干模型，保存到 save_path。"""
    print()
    print("=" * 60)
    print(f"训练 {name}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(
            degrees=8, translate=(0.05, 0.05),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
    ])

    print("加载 GTSRB ...")
    train_set = GTSRB(root=str(DATA_ROOT), split='train',
                       transform=train_tf, download=False)
    test_set = GTSRB(root=str(DATA_ROOT), split='test',
                      transform=test_tf, download=False)
    print(f"  训练: {len(train_set)} | 测试: {len(test_set)}")

    labels = get_train_labels(train_set)
    labels_t = torch.tensor(labels)
    class_counts = torch.bincount(labels_t, minlength=43).float()
    sample_weights = (1.0 / class_counts)[labels_t]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    model = builder(num_classes=43).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}  ({n_params/1e3:.1f} K)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max,
        steps_per_epoch=len(train_loader), epochs=num_epochs,
        pct_start=0.2,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    history = []

    t0_total = time.time()
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss, n_batch = 0.0, 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1:2d}/{num_epochs}", ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n_batch += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        train_loss /= n_batch
        test_acc = evaluate_clean(model, test_loader, device)
        epoch_time = time.time() - t0
        improved = test_acc > best_acc
        marker = "  ← best" if improved else ""
        print(f"    loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
              f"time={epoch_time:.1f}s{marker}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'test_acc': float(test_acc),
            'time_s': float(epoch_time),
        })

        if improved:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    total_time = time.time() - t0_total
    print(f"\n  ✓ 完成 ({total_time/60:.1f} 分钟)，"
          f"clean test acc = {best_acc*100:.2f}%")
    print(f"  模型保存: {save_path}")

    return best_acc, history


if __name__ == '__main__':
    torch.manual_seed(42)

    print("=" * 60)
    print("Backbone Ablation Training")
    print("两个骨干各 10 epoch，CPU 上预计 70-90 分钟")
    print("=" * 60)

    t0 = time.time()
    results = {}

    acc1, h1 = train_one_backbone(
        'ShuffleNetV2 (x0.5)',
        build_shufflenet_v2_x0_5,
        MODELS_DIR / 'shufflenet_baseline.pth',
        num_epochs=10,
    )
    results['shufflenet_v2_x0_5'] = {
        'best_acc': float(acc1), 'history': h1,
    }

    acc2, h2 = train_one_backbone(
        'MobileNetV2 (w=0.5)',
        build_mobilenet_v2_w0_5,
        MODELS_DIR / 'mobilenet_baseline.pth',
        num_epochs=10,
    )
    results['mobilenet_v2_w0_5'] = {
        'best_acc': float(acc2), 'history': h2,
    }

    with open(RESULTS_DIR / 'backbone_training.json', 'w') as f:
        json.dump(results, f, indent=2)

    total = (time.time() - t0) / 60
    print()
    print("=" * 60)
    print(f"✓ 全部完成（总耗时 {total:.1f} 分钟）")
    print("=" * 60)
    print(f"  ShuffleNetV2 (x0.5):  {acc1*100:.2f}%")
    print(f"  MobileNetV2  (w=0.5): {acc2*100:.2f}%")
    print(f"  CompactCNN (参考):    93.92%")
    print()
    print("下一步：python src/evaluate_backbone_ablation.py")
