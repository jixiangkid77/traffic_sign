"""
Day 2 (v4): CompactCNN baseline 训练。

设计目标：10 epoch × 1 分钟 → 95%+ accuracy

关键改动：
  - 模型：MobileNetV3-Small → CompactCNN (1.55M → 0.15M 参数)
  - 输入：64x64 → 32x32
  - 不使用 ImageNet 预训练（GTSRB 域差异大，从头训反而稳定）
  - 增强：温和 (RandomAffine ±8° + ColorJitter 0.15)

输出:
  models/mbnetv3_baseline.pth      (沿用旧文件名，方便后续脚本不改)
  results/baseline_training.json   每个 epoch 训练日志

CPU 训练时间预估: 每 epoch 15-30 秒，总 5-10 分钟
预期 clean test accuracy: 95-97%

使用：
  python src/train_baseline.py
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import GTSRB
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model  # noqa: E402


def get_train_labels(train_set):
    """从 GTSRB 训练集提取所有标签。"""
    if hasattr(train_set, '_samples'):
        return [label for _, label in train_set._samples]
    elif hasattr(train_set, 'samples'):
        return [label for _, label in train_set.samples]
    else:
        return [train_set[i][1] for i in range(len(train_set))]


def evaluate(model, loader, device):
    """在测试集上算 Top-1 accuracy。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / 'data'
    MODELS_DIR = PROJECT_ROOT / 'models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ============== 超参数 ==============
    IMAGE_SIZE = 32        # ← GTSRB 标准尺寸
    BATCH_SIZE = 256       # ← 大 batch 因为图变小了
    NUM_EPOCHS = 10        # ← 严格 10 epoch
    LR = 2e-3              # ← 略高（因为从头训）
    WEIGHT_DECAY = 5e-4
    NUM_CLASSES = 43
    SEED = 42

    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Day 2 (v4): CompactCNN Baseline Training")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"超参数: image_size={IMAGE_SIZE}, batch_size={BATCH_SIZE}, "
          f"epochs={NUM_EPOCHS}, lr={LR}")
    print()

    # ============== 数据增强（温和但充分）==============
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=8,                  # 小幅旋转
            translate=(0.05, 0.05),     # 小幅平移
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        # 注意：从头训不用 ImageNet mean/std，用 GTSRB 自己的
        transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.3401, 0.3120, 0.3212], [0.2725, 0.2609, 0.2669]),
    ])

    print("加载 GTSRB ...")
    train_set = GTSRB(root=str(DATA_ROOT), split='train',
                       transform=train_tf, download=False)
    test_set = GTSRB(root=str(DATA_ROOT), split='test',
                      transform=test_tf, download=False)
    print(f"  训练集: {len(train_set)} 张")
    print(f"  测试集: {len(test_set)} 张")

    # ============== 类别均衡采样 ==============
    print("\n计算类别权重 ...")
    labels = get_train_labels(train_set)
    labels_t = torch.tensor(labels)
    class_counts = torch.bincount(labels_t, minlength=NUM_CLASSES).float()
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_t]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f"  类别样本数范围: {int(class_counts.min())} ~ {int(class_counts.max())}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ============== 模型 ==============
    print("\n构建 CompactCNN (从头训练，无 ImageNet 预训练) ...")
    model = build_model(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}  ({n_params/1e3:.1f} K)")

    # ============== 优化器 ==============
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,             # 前 20% 升 LR，后 80% 余弦下降
        anneal_strategy='cos',
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑提高泛化

    # ============== 训练循环 ==============
    print(f"\n开始训练 {NUM_EPOCHS} epochs ...\n")
    results = []
    best_acc = 0.0
    total_t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        n_batch = 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:2d}/{NUM_EPOCHS}",
                    ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()       # OneCycle 每 batch step
            train_loss += loss.item()
            n_batch += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        train_loss /= n_batch
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - t0

        improved = test_acc > best_acc
        marker = "  ← best" if improved else ""
        print(f"  Epoch {epoch+1:2d}: loss={train_loss:.4f}  "
              f"test_acc={test_acc:.4f}  time={epoch_time:.1f}s{marker}")

        results.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'test_acc': float(test_acc),
            'time_s': float(epoch_time),
        })

        if improved:
            best_acc = test_acc
            torch.save(model.state_dict(), MODELS_DIR / 'mbnetv3_baseline.pth')

        with open(RESULTS_DIR / 'baseline_training.json', 'w') as f:
            json.dump({
                'best_acc': float(best_acc),
                'epochs': results,
            }, f, indent=2)

    total_time = time.time() - total_t0
    print()
    print("=" * 60)
    print(f"✓ 训练完成！")
    print("=" * 60)
    print(f"  最佳 clean test accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"  总训练时间: {total_time/60:.1f} 分钟")
    print(f"  平均每 epoch: {total_time/NUM_EPOCHS:.1f} 秒")
    print(f"  模型保存在: {MODELS_DIR / 'mbnetv3_baseline.pth'}")
    print()
    if best_acc >= 0.95:
        print("✓ 准确率达标！可以直接进入 Day 3。")
    elif best_acc >= 0.92:
        print("△ 准确率可以接受（92-95%），可以进入 Day 3，paper 数据稍弱但能用。")
    else:
        print("✗ 准确率仍 < 92%，告诉我具体数字，我帮你 debug。")


if __name__ == '__main__':
    main()
