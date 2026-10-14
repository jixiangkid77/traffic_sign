r"""
ffa_smoke_test.py: 跑满 6 个测试集之前的廉价真权重确认

只在 clean 测试集的前 N 张上跑修好的 FFA-Net，打印准确率，不跑全量。
判据：修对了的话 clean 准确率应回到 ~88-92%（去雾会轻微改动干净图，但不该崩）。
      如果还是 ~20% 上下，说明没修好，别去跑那 4.5 小时，把输出贴给我。

复用 evaluate_learned_baselines.py 里修好的 FFANetEnhancer 和 evaluate_all.py 的
GTSRBTestDataset / test_tf，所以这里测的就是全量会用的同一套代码。

用法:
  conda activate pcm_sim
  python src\ffa_smoke_test.py
约 5 到 7 分钟。
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from evaluate_all import load_test_labels, test_tf, GTSRBTestDataset, MODELS_DIR  # noqa: E402
from model import build_model  # noqa: E402
from evaluate_learned_baselines import (  # noqa: E402
    FFANetEnhancer, FFA_NET_WEIGHTS, resolve_testsets,
)

N = 2000  # 抽 clean 前 2000 张做快速核对


def main():
    device = torch.device('cpu')
    if not FFA_NET_WEIGHTS.exists():
        print(f"[STOP] 没找到 {FFA_NET_WEIGHTS}")
        sys.exit(1)

    model = build_model(num_classes=43)
    model.load_state_dict(torch.load(MODELS_DIR / 'mbnetv3_baseline.pth', map_location=device))
    model.eval()

    ffa = FFANetEnhancer(FFA_NET_WEIGHTS, device)
    print(f"loaded FFA-Net: {FFA_NET_WEIGHTS.name} "
          f"({sum(p.numel() for p in ffa.net.parameters()):,} params)")

    labels = load_test_labels()
    clean_dir = resolve_testsets()['clean']
    ds = GTSRBTestDataset(clean_dir, labels, ffa, test_tf)
    n = min(N, len(ds))
    loader = DataLoader(Subset(ds, list(range(n))), batch_size=128,
                        shuffle=False, num_workers=0)

    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    acc = correct / total * 100.0

    print("=" * 52)
    print(f"FFA-Net  clean subset  (N={total})  accuracy = {acc:.2f}%")
    print("=" * 52)
    if acc >= 80:
        print("-> 修好了（接近 baseline 的 93.92%）。可以放心跑满 6 个测试集。")
    elif acc <= 40:
        print("-> 还是坏的。别跑全量，把这段输出贴给我。")
    else:
        print("-> 介于中间，把这段输出贴我看一下再决定。")


if __name__ == '__main__':
    main()
