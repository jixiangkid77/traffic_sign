r"""
evaluate_learned_baselines.py: 学习型增强基线对比（期刊扩展, Zero-DCE + FFA-Net）

目的
  论文 Related Work 引用了 Zero-DCE [13][14]、URetinex-Net [15]、FFA-Net [12]，
  并断言学习型方法用在 regime 之外会伤下游性能，但正文实验只比了 fixed 三件。
  本脚本把学习型增强当"统一施加"的预处理基线，和 fixed / VA-Adaptive 同位比较：
    - Zero-DCE: 低光增强专精
    - FFA-Net : 去雾专精

设计原则（和论文主实验完全对齐，保证可比）
  复用 evaluate_all.py 已验证的 harness：同一个冻结 CompactCNN、同一套 test_tf
  (Resize32 + GTSRB Normalize)、同一个 GTSRBTestDataset、同一个 evaluate()。
  唯一新增的是学习型增强函数，槽位和 fixed / VA 一致：原图尺寸增强，BGR uint8 进出。

正确性锚点
  脚本默认同时重跑 baseline 和 va_adaptive，它们的 5 退化集平均应复现论文 Table I
  的 63.05 / 68.84。对得上说明 harness 没漂，学习型那几行才可信。

哪些方法会跑
  baseline 和 va_adaptive 总是跑（锚点）。zero_dce / ffa_net 各自只在权重存在时才跑，
  所以你可以只放一个权重先跑一个，也可以两个都放一次跑齐。

权重准备
  Zero-DCE:
    1. git clone https://github.com/Li-Chongyi/Zero-DCE
    2. copy Zero-DCE\Zero-DCE_code\snapshots\Epoch99.pth
            -> D:\Project\traffic_sign\models\zero_dce_Epoch99.pth
  FFA-Net:
    1. git clone https://github.com/zhilin007/FFA-Net
    2. FFA-Net 的预训练权重不在 git 仓库里，在其 README 给出的 Google Drive
       链接的 trained_models\ 下。下载 ots_train_ffa_3_19.pk（室外去雾模型），复制到
            D:\Project\traffic_sign\models\ffa_net_ots.pk

运行
  conda activate pcm_sim
  python src\evaluate_learned_baselines.py

注意
  FFA-Net 比 Zero-DCE 重很多，CPU 上跑 6 个测试集会慢（可能 1 到 2 小时以上），
  建议挂后台或过夜。Zero-DCE 约几十分钟。

输出（results\）
  learned_baseline_results.csv / .json    和 main_results.csv 同结构
  learned_baseline_run_config.json        复现配置
  learned_baseline_log.txt                执行日志
"""
import sys
import csv
import json
import time
import platform
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---- 复用主实验已验证的 harness ----
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from evaluate_all import (  # noqa: E402
    load_test_labels, evaluate, test_tf,
    DATA_ROOT, MODELS_DIR, RESULTS_DIR,
)
from model import build_model              # noqa: E402
from enhance import no_enhance, adaptive_enhance  # noqa: E402

NUM_CLASSES = 43
GTSRB_MODEL_PATH = MODELS_DIR / 'mbnetv3_baseline.pth'   # 实为 CompactCNN（旧文件名）
ZERO_DCE_WEIGHTS = MODELS_DIR / 'zero_dce_Epoch99.pth'
FFA_NET_WEIGHTS  = MODELS_DIR / 'ffa_net_ots.pk'


def _strip_module(state):
    """去掉 DataParallel 保存的 'module.' 前缀。"""
    return {k.replace('module.', '', 1): v for k, v in state.items()}


# ============================================================
# Zero-DCE 网络（逐字采用官方 Li-Chongyi/Zero-DCE 仓库 model.py）
# ============================================================
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image_1, enhance_image, torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)


# ============================================================
# FFA-Net 网络（逐字采用官方 zhilin007/FFA-Net 仓库 net/models/FFA.py）
# ============================================================
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.pa(x)


class CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.ca(self.avg_pool(x))


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super().__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()])
        self.palayer = PALayer(self.dim)
        post_precess = [conv(self.dim, self.dim, kernel_size),
                        conv(self.dim, 3, kernel_size)]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


# ============================================================
# 把两个学习型网络包成和 enhance.py 同契约：BGR uint8 进、BGR uint8 出。
# 增强在原图尺寸上施加（全卷积，无尺寸约束），与 fixed / VA 一致。
# ============================================================
class _LearnedEnhancer:
    def __init__(self, net, device):
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self._run(t)
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)        # RGB uint8
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)          # 还原 BGR 契约

    def _run(self, t):
        raise NotImplementedError


class ZeroDCEEnhancer(_LearnedEnhancer):
    def __init__(self, weights_path, device):
        net = enhance_net_nopool()
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(_strip_module(state))
        super().__init__(net, device)

    def _run(self, t):
        _, enhanced, _ = self.net(t)
        return enhanced


class FFANetEnhancer(_LearnedEnhancer):
    # FFA-Net 官方训练与推理都对输入做这个归一化（data_utils.py / test.py）。
    # 推理必须一致，否则网络收到分布外输入会输出乱码（上一版 clean 掉到 21% 就是漏了这步）。
    _MEAN = torch.tensor([0.64, 0.60, 0.58]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.14, 0.15, 0.152]).view(1, 3, 1, 1)

    def __init__(self, weights_path, device):
        net = FFA(gps=3, blocks=19)
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        net.load_state_dict(_strip_module(state))
        super().__init__(net, device)
        self._mean = self._MEAN.to(device)
        self._std = self._STD.to(device)

    def _run(self, t):
        t = (t - self._mean) / self._std       # FFA-Net 要求的输入归一化
        return self.net(t)


def resolve_testsets():
    clean_candidates = [
        DATA_ROOT / 'gtsrb' / 'GTSRB' / 'Final_Test' / 'Images',
        DATA_ROOT / 'gtsrb' / 'Final_Test' / 'Images',
    ]
    clean_dir = next((p for p in clean_candidates if p.exists()), None)
    if clean_dir is None:
        for p in DATA_ROOT.rglob('Final_Test'):
            if (p / 'Images').exists():
                clean_dir = p / 'Images'
                break
    if clean_dir is None:
        raise FileNotFoundError("找不到 clean 测试集目录")
    return {
        'clean':       clean_dir,
        'lowlight':    DATA_ROOT / 'gtsrb_lowlight',
        'foggy':       DATA_ROOT / 'gtsrb_foggy',
        'lowcontrast': DATA_ROOT / 'gtsrb_lowcontrast',
        'noisy':       DATA_ROOT / 'gtsrb_noisy',
        'mixed':       DATA_ROOT / 'gtsrb_mixed',
    }


DEGRADED = ['lowlight', 'foggy', 'lowcontrast', 'noisy', 'mixed']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 70)
    log("Learned-Enhancement Baseline Comparison (Zero-DCE + FFA-Net)")
    log("=" * 70)
    log(f"device: {device}")

    if not GTSRB_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 GTSRB 模型: {GTSRB_MODEL_PATH}")

    # 冻结分类器
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(GTSRB_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    log(f"classifier: {GTSRB_MODEL_PATH.name} "
        f"({sum(p.numel() for p in model.parameters()):,} params)")

    # 阈值
    with open(RESULTS_DIR / 'thresholds.json') as f:
        th = json.load(f)
    T1, T2 = th['T1_brightness_low'], th['T2_contrast_low']
    T3, T4 = th['T3_edge_low'], th['T4_brightness_high']
    log(f"thresholds: T1={T1:.4f} T2={T2:.4f} T3={T3:.4f} T4={T4:.4f}")

    # 学习型增强器：各自只在权重存在时加入
    learned = {}
    if ZERO_DCE_WEIGHTS.exists():
        enc = ZeroDCEEnhancer(ZERO_DCE_WEIGHTS, device)
        learned['zero_dce'] = enc
        log(f"loaded Zero-DCE: {ZERO_DCE_WEIGHTS.name} "
            f"({sum(p.numel() for p in enc.net.parameters()):,} params)")
    else:
        log(f"[skip] 未找到 {ZERO_DCE_WEIGHTS.name}, 不跑 zero_dce")
    if FFA_NET_WEIGHTS.exists():
        enc = FFANetEnhancer(FFA_NET_WEIGHTS, device)
        learned['ffa_net'] = enc
        log(f"loaded FFA-Net: {FFA_NET_WEIGHTS.name} "
            f"({sum(p.numel() for p in enc.net.parameters()):,} params)")
    else:
        log(f"[skip] 未找到 {FFA_NET_WEIGHTS.name}, 不跑 ffa_net")

    if not learned:
        log("\n[STOP] 没有任何学习型权重。请至少准备一个:")
        log(f"  Zero-DCE -> {ZERO_DCE_WEIGHTS}")
        log(f"  FFA-Net  -> {FFA_NET_WEIGHTS}")
        sys.exit(1)

    labels = load_test_labels()
    testsets = resolve_testsets()
    log(f"test labels: {len(labels)}")

    METHODS = {
        'baseline':    no_enhance,
        'va_adaptive': lambda img: adaptive_enhance(img, T1, T2, T3, T4),
    }
    METHODS.update(learned)

    log("\n" + "=" * 70)
    results = {}
    total_t0 = time.time()
    for m_name, fn in METHODS.items():
        results[m_name] = {}
        for t_name, t_dir in testsets.items():
            r = evaluate(model, t_dir, labels, fn, device)
            results[m_name][t_name] = r
            log(f"  {m_name:12s} × {t_name:12s}  "
                f"acc={r['top1_acc']*100:6.2f}%  f1={r['macro_f1']*100:6.2f}%  "
                f"({r['time_per_img_ms']:.2f} ms/img)")
            with open(RESULTS_DIR / 'learned_baseline_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        log("")
    total_min = (time.time() - total_t0) / 60.0

    def deg_avg(m, key):
        return sum(results[m][t][key] for t in DEGRADED) / len(DEGRADED) * 100

    log("=" * 70)
    log("  Degraded-average (5 sets)   acc / macro-F1")
    log("=" * 70)
    for m in METHODS:
        log(f"  {m:12s}  acc={deg_avg(m,'top1_acc'):6.2f}%   "
            f"f1={deg_avg(m,'macro_f1'):6.2f}%   "
            f"clean_acc={results[m]['clean']['top1_acc']*100:6.2f}%")

    log("\n  [correctness anchor vs paper Table I]")
    anchor = {'baseline': 63.05, 'va_adaptive': 68.84}
    ok_all = True
    for m, exp in anchor.items():
        got = deg_avg(m, 'top1_acc')
        ok = abs(got - exp) <= 0.05
        ok_all = ok_all and ok
        log(f"    {m:12s} deg-avg got={got:.2f}  paper={exp}  "
            f"{'OK' if ok else 'MISMATCH -> harness/data drift, learned rows not trustworthy'}")
    if ok_all:
        log("    -> harness reproduces paper; learned rows are trustworthy.")

    csv_path = RESULTS_DIR / 'learned_baseline_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'testset', 'top1_acc', 'macro_f1',
                    'n_samples', 'total_time_s', 'time_per_img_ms'])
        for m in METHODS:
            for t in testsets:
                r = results[m][t]
                w.writerow([m, t, r['top1_acc'], r['macro_f1'],
                            r['n_samples'], r['total_time_s'], r['time_per_img_ms']])

    run_config = {
        'script': 'evaluate_learned_baselines.py',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'device': str(device),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'opencv': cv2.__version__,
        'classifier_ckpt': str(GTSRB_MODEL_PATH),
        'zero_dce_weights': str(ZERO_DCE_WEIGHTS) if 'zero_dce' in learned else None,
        'ffa_net_weights': str(FFA_NET_WEIGHTS) if 'ffa_net' in learned else None,
        'gtsrb_normalize_mean': [0.3401, 0.3120, 0.3212],
        'gtsrb_normalize_std': [0.2725, 0.2609, 0.2669],
        'thresholds': {'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4},
        'methods': list(METHODS.keys()),
        'testsets': list(testsets.keys()),
        'pipeline': 'cv2.imread(BGR) -> enhance(orig size) -> BGR2RGB -> PIL -> Resize32 -> GTSRB Normalize -> frozen CompactCNN',
        'total_minutes': round(total_min, 2),
    }
    with open(RESULTS_DIR / 'learned_baseline_run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
    with open(RESULTS_DIR / 'learned_baseline_log.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines))

    log(f"\n  total: {total_min:.1f} min")
    log(f"  csv  -> {csv_path}")
    log(f"  json -> {RESULTS_DIR / 'learned_baseline_results.json'}")


if __name__ == '__main__':
    main()
