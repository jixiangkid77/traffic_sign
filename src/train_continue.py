"""
Day 2 续训：从 v4 训完的 89.45% checkpoint 继续训 10 epoch。

策略：
  - 加载已有权重（不重新初始化）
  - 备份 v4 模型到 mbnetv3_baseline_v4_10ep.pth（防止覆盖丢失）
  - 用低学习率 (5e-4 → 1e-5 cosine decay) 做精细微调
  - 保留原 model/data/loss 配置不变

预期：89.45% → 93-96%
耗时：约 11 分钟（10 epoch × 65 秒）

使用：
  python src/train_continue.py
"""
import json
import shutil
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
    if hasattr(train_set, '_samples'):
        return [label for _, label in train_set._samples]
    elif hasattr(train_set, 'samples'):
        return [label for _, label in train_set.samples]
    return [train_set[i][1] for i in range(len(train_set))]


def evaluate(model, loader, device):
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

    # ============== 超参数 ==============
    IMAGE_SIZE = 32
    BATCH_SIZE = 256
    NUM_EPOCHS = 10           # 续训再加 10 epoch
    LR_MAX = 5e-4             # ← 比原始 2e-3 低 4 倍（精细微调）
    LR_MIN = 1e-5             # ← cosine 退火到 1e-5
    WEIGHT_DECAY = 5e-4
    NUM_CLASSES = 43
    SEED = 42

    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============== 备份 v4 模型（防止训坏丢失）==============
    model_path = MODELS_DIR / 'mbnetv3_baseline.pth'
    backup_path = MODELS_DIR / 'mbnetv3_baseline_v4_10ep.pth'
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到 {model_path}，请先跑完 train_baseline.py 的 10 epoch"
        )
    if not backup_path.exists():
        shutil.copy(model_path, backup_path)
        print(f"✓ 已备份 v4 (89.45%) 到: {backup_path}")
    else:
        print(f"  v4 备份已存在: {backup_path}")

    print()
    print("=" * 60)
    print("Day 2 续训：从 v4 (89.45%) 继续 10 epoch")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"超参数: LR {LR_MAX} → {LR_MIN} (cosine), epochs={NUM_EPOCHS}")
    print()

    # ============== 数据（和 train_baseline.py 一样）==============
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
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

    # 类别权重采样
    labels = get_train_labels(train_set)
    labels_t = torch.tensor(labels)
    class_counts = torch.bincount(labels_t, minlength=NUM_CLASSES).float()
    sample_weights = (1.0 / class_counts)[labels_t]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ============== 加载已有模型 ==============
    print(f"\n加载 v4 模型: {model_path}")
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}  ({n_params/1e3:.1f} K)")

    # 先测一下加载后的初始 acc，确认没出错
    initial_acc = evaluate(model, test_loader, device)
    print(f"  续训前 test_acc: {initial_acc:.4f}  ({initial_acc*100:.2f}%)")
    print(f"  （应当接近 89.45%）")

    # ============== 优化器（精细微调）==============
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ============== 续训循环 ==============
    print(f"\n开始续训 {NUM_EPOCHS} epochs ...\n")

    # 加载已有训练日志（接着写）
    log_path = RESULTS_DIR / 'baseline_training.json'
    if log_path.exists():
        with open(log_path) as f:
            log_data = json.load(f)
        prev_results = log_data.get('epochs', [])
        prev_best = log_data.get('best_acc', initial_acc)
    else:
        prev_results = []
        prev_best = initial_acc

    best_acc = max(prev_best, initial_acc)
    print(f"历史最佳: {prev_best:.4f}, 当前模型: {initial_acc:.4f}, "
          f"基准: {best_acc:.4f}\n")

    results = list(prev_results)
    base_epoch = len(prev_results)
    total_t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        actual_epoch = base_epoch + epoch + 1

        model.train()
        train_loss = 0.0
        n_batch = 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {actual_epoch:2d} (续 {epoch+1}/{NUM_EPOCHS})",
                    ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        train_loss /= n_batch
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        improved = test_acc > best_acc
        marker = "  ← best" if improved else ""
        print(f"  Epoch {actual_epoch:2d}: loss={train_loss:.4f}  "
              f"test_acc={test_acc:.4f}  lr={cur_lr:.2e}  "
              f"time={epoch_time:.1f}s{marker}")

        results.append({
            'epoch': actual_epoch,
            'train_loss': float(train_loss),
            'test_acc': float(test_acc),
            'time_s': float(epoch_time),
            'phase': 'continue',
        })

        if improved:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)

        with open(log_path, 'w') as f:
            json.dump({
                'best_acc': float(best_acc),
                'epochs': results,
            }, f, indent=2)

    total_time = time.time() - total_t0
    print()
    print("=" * 60)
    print("✓ 续训完成！")
    print("=" * 60)
    print(f"  续训前 acc: {initial_acc*100:.2f}%")
    print(f"  续训后 best acc: {best_acc*100:.2f}%")
    print(f"  提升: {(best_acc - initial_acc)*100:+.2f} pp")
    print(f"  总续训时间: {total_time/60:.1f} 分钟")
    print(f"  最佳模型: {model_path}")
    print(f"  v4 备份:   {backup_path}")
    print()
    if best_acc >= 0.95:
        print("✓ 95% 达标！立刻进 Day 3。")
    elif best_acc >= 0.93:
        print("△ 93-95%，论文数据已很扎实，进 Day 3。")
    elif best_acc >= 0.91:
        print("△ 91-93%，比 v4 有提升，可以进 Day 3。")
    else:
        print("✗ 提升不明显，建议直接接受 v4 (89.45%)，进 Day 3。")


if __name__ == '__main__':
    main()
