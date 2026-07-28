r"""
allinone_smoke_test.py: 全量前的廉价真权重确认 (只依赖 CURE, 不需要 data/ 合成集)

两步:
  1. 小图尺寸自检: 用 CURE 尺寸跨度 (含 3x7) 的随机图过封装, 确认不报错、尺寸还原。
  2. 真实 clean 自检: 在 CURE-TSR ChallengeFree (真实无退化标志) 上, 同跑 baseline 和 PromptIR,
     PromptIR 准确率应与 baseline 接近 (都约 80%)。若 PromptIR 远低于 baseline,
     说明封装/权重没对上, 别去全量, 把这段输出 + 上面 missing/unexpected 贴回。

为什么不用 GTSRB clean
  那需要 data/ (合成集) 里的 clean GTSRB; 你这次可能只传 CURE, 所以改用 CURE ChallengeFree,
  只依赖 datasets/CURE-TSR。判据从"绝对 90%"改成"和 baseline 的差距", 自校准、更稳。

用法 (Colab):
  %cd /content/traffic_sign
  !python src/allinone_smoke_test.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from evaluate_cure_tsr_external import (   # noqa: E402
    CureTSRMappedDataset, DEFAULT_CURE_TO_GTSRB, CURE_TSR_DIR,
    GTSRB_MEAN, GTSRB_STD, load_gtsrb_model, evaluate, fn_baseline,
)
from allinone_enhancer import AllInOneEnhancer  # noqa: E402

WEIGHTS = "/content/promptir_allinone.ckpt"     # 改成你的 ckpt 实际路径


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    enh = AllInOneEnhancer(WEIGHTS, device)

    # ---- 1. 小图尺寸自检 (无需数据) ----
    print("\n[1] 小图尺寸自检")
    for h, w in [(3, 7), (14, 14), (28, 28), (60, 90), (120, 200)]:
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)      # BGR uint8
        out = enh(img)
        assert out.shape == img.shape, f"尺寸不还原: in {img.shape} out {out.shape}"
        print(f"    {str((h, w)):>12} -> OK")
    print("    小图全过")

    # ---- 2. CURE ChallengeFree 真实 clean 自检 ----
    print("\n[2] CURE ChallengeFree 真实 clean 自检")
    model = load_gtsrb_model(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])

    def cf_acc(fn):
        ds = CureTSRMappedDataset(
            root=CURE_TSR_DIR, mapping=DEFAULT_CURE_TO_GTSRB,
            challenge_type=0, enhancement_fn=fn, transform=transform,
        )
        return evaluate(model, device, ds)["acc"], len(ds)

    base_acc, n = cf_acc(fn_baseline)
    pir_acc, _ = cf_acc(enh)

    print("=" * 60)
    print(f"ChallengeFree (n={n})   baseline={base_acc:.2f}%   PromptIR={pir_acc:.2f}%")
    print("=" * 60)
    gap = base_acc - pir_acc
    if pir_acc >= base_acc - 8:
        print(f"-> 通过 (PromptIR 比 baseline 低 {gap:.1f}pp, 在合理范围)。可全量跑 CURE。")
    elif pir_acc <= base_acc - 25:
        print(f"-> 没对上 (低了 {gap:.1f}pp)。别全量, 把本段 + missing/unexpected 贴回。")
    else:
        print(f"-> 偏低 {gap:.1f}pp, 把本段输出贴回再决定。")


if __name__ == "__main__":
    main()
