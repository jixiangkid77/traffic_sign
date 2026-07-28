r"""
dump_cure_predictions.py: 导出 CURE-TSR 逐图预测 (bootstrap_ci.py 的前置, 实验 2)

为什么需要
  你跑过的 evaluate_cure_tsr_external.py 只存了每个 (challenge, severity, method)
  的聚合 acc / macro-F1, 没存逐图预测。自助置信区间需要逐图的对错和预测标签。
  本脚本复用同一套 harness (同一冻结 CompactCNN、同一 CureTSRMappedDataset、
  同一 transform、同一 METHODS 含学习型基线), 重跑一遍并把逐图预测落盘。
  模型确定性 (eval / CPU), 所以这次预测与你正文结果一致。

输出
  outputs_cure_tsr\cure_tsr_per_image_predictions.csv
  列: split, challenge, severity, method, idx, true_label, pred_label, correct

成本
  与当初那次 7 方法跑 CURE 差不多 (FFA 最慢, 约 95 到 120 分钟)。只要关键三方对比,
  把 ONLY_METHODS 改成 ["baseline", "va_adaptive", "ffa_net"] 可省时间。建议挂后台或过夜。

配对前提
  CureTSRMappedDataset 的图序只由 (challenge, severity) 决定, 与 enhancement_fn 无关,
  所以同一 cell 内各方法的 idx 一一对应。bootstrap_ci.py 还会再做一次 true_label
  跨方法一致性检查兜底。

用法
  conda activate pcm_sim
  python src\dump_cure_predictions.py
"""
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

# 复用 CURE 评测脚本的全部部件 (与正文完全一致)
from evaluate_cure_tsr_external import (   # noqa: E402
    CureTSRMappedDataset, DEFAULT_CURE_TO_GTSRB, CURE_TSR_DIR,
    EVAL_CHALLENGES, EVAL_SEVERITIES, CHALLENGE_TYPES,
    GTSRB_MEAN, GTSRB_STD, METHODS, load_gtsrb_model, OUTPUT_DIR,
)
# 学习型基线: 与 evaluate_cure_tsr_external.main() 里一样按需加入
from evaluate_learned_baselines import (   # noqa: E402
    ZeroDCEEnhancer, FFANetEnhancer, ZERO_DCE_WEIGHTS, FFA_NET_WEIGHTS,
)

ONLY_METHODS = None   # None = 全部; 或 ["baseline", "va_adaptive", "ffa_net"] 只跑关键三方
OUT_CSV = OUTPUT_DIR / "cure_tsr_per_image_predictions.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_gtsrb_model(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])

    if ZERO_DCE_WEIGHTS.exists() and "zero_dce" not in METHODS:
        METHODS["zero_dce"] = ZeroDCEEnhancer(ZERO_DCE_WEIGHTS, device)
        print(f"  + zero_dce ({ZERO_DCE_WEIGHTS.name})")
    if FFA_NET_WEIGHTS.exists() and "ffa_net" not in METHODS:
        METHODS["ffa_net"] = FFANetEnhancer(FFA_NET_WEIGHTS, device)
        print(f"  + ffa_net ({FFA_NET_WEIGHTS.name})")

    method_names = (list(METHODS.keys()) if ONLY_METHODS is None
                    else [m for m in METHODS if m in ONLY_METHODS])
    print(f"dumping methods: {method_names}")

    # (split, challenge_name, challenge_id, severity); severity=0 仅占位给 ChallengeFree
    jobs = [("challengefree", "ChallengeFree", 0, 0)]
    for ch_id in EVAL_CHALLENGES:
        for sev in EVAL_SEVERITIES:
            jobs.append(("challenge", CHALLENGE_TYPES[ch_id], ch_id, sev))

    t0 = time.time()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "challenge", "severity", "method",
                    "idx", "true_label", "pred_label", "correct"])
        for split, ch_name, ch_id, sev in jobs:
            for m in method_names:
                fn = METHODS[m]
                if split == "challenge":
                    ds = CureTSRMappedDataset(
                        root=CURE_TSR_DIR, mapping=DEFAULT_CURE_TO_GTSRB,
                        challenge_type=ch_id, severity=sev,
                        enhancement_fn=fn, transform=transform,
                    )
                else:
                    ds = CureTSRMappedDataset(
                        root=CURE_TSR_DIR, mapping=DEFAULT_CURE_TO_GTSRB,
                        challenge_type=0, enhancement_fn=fn, transform=transform,
                    )
                if len(ds) == 0:
                    continue
                loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
                idx = 0
                with torch.no_grad():
                    for x, y in loader:
                        pred = model(x.to(device)).argmax(1).cpu().numpy()
                        y = y.numpy()
                        for t_, p_ in zip(y, pred):
                            w.writerow([split, ch_name, sev, m, idx,
                                        int(t_), int(p_), int(t_ == p_)])
                            idx += 1
                print(f"  {split:13s} {ch_name:13s} sev{sev} {m:12s} n={idx}")

    print(f"\n  total {(time.time() - t0) / 60:.1f} min")
    print(f"  -> {OUT_CSV}")
    print("\n下一步: python src\\bootstrap_ci.py")


if __name__ == "__main__":
    main()
