r"""
allinone_enhancer.py: PromptIR (all-in-one 修复) 封装, 接现有 _LearnedEnhancer 契约

已对仓库 va1shn9v/PromptIR 核实:
  - 模型类: from net.model import PromptIR; all-in-one = PromptIR(decoder=True)
  - 输入: [0,1] RGB 直接喂, 不做 mean/std 归一化 (与 FFA 不同, 别加归一化)
  - forward(inp_img) 返回单个张量 (复原图), 不是 tuple
  - 纯 PyTorch, 无可变形卷积/自定义 CUDA 算子, CPU/GPU 都能跑
  - 官方权重是 Lightning ckpt (键名形如 net.*), 故剥掉 'net.' 前缀再装进 bare PromptIR

封装契约 (与 evaluate_learned_baselines._LearnedEnhancer 一致):
  BGR uint8 进, 原图尺寸增强, BGR uint8 出。_run 只负责 RGB float[0,1] (1,3,H,W) 的网络部分。

CURE 小图处理:
  Restormer 骨干 3 次 /2 下采样, 要求边长是 8 的倍数。CURE 有小到 3x7 的图,
  故 pad 到 8 的倍数且不小于 64 (保证 latent 不退化), 用 replicate
  (reflect 在 pad >= 边长时会报错), 跑完裁回原尺寸。裁回后只剩真实像素,
  padding 出来的边在送分类器前的 resize 之前就被丢掉。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from evaluate_learned_baselines import _LearnedEnhancer      # 同一 BGR uint8 契约  # noqa: E402

PROMPTIR_REPO = "/content/PromptIR"                          # clone 出来的仓库路径
sys.path.insert(0, PROMPTIR_REPO)
from net.model import PromptIR                               # 已核实: 类在 net/model.py  # noqa: E402


def _strip_prefixes(state):
    """去掉 Lightning / DataParallel 包装前缀, 还原 bare PromptIR 的 key。"""
    out = {}
    for k, v in state.items():
        for pre in ("model.net.", "net.", "module.", "model."):
            if k.startswith(pre):
                k = k[len(pre):]
                break
        out[k] = v
    return out


class AllInOneEnhancer(_LearnedEnhancer):
    def __init__(self, weights_path, device, min_size=64, multiple=8):
        net = PromptIR(decoder=True)                         # 已核实: all-in-one 用 decoder=True
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]                       # Lightning ckpt
        elif isinstance(ckpt, dict) and "params" in ckpt:
            state = ckpt["params"]
        else:
            state = ckpt
        state = _strip_prefixes(state)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[AllInOne] load_state_dict | missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 5 or len(unexpected) > 5:
            print("[AllInOne] 警告: 缺失/多余 key 偏多, 权重可能没对上。")
            print("           若烟雾测试 clean 也低, 改用 Lightning 加载 (见文件末尾注释)。")
        self._min = min_size
        self._mul = multiple
        super().__init__(net, device)                       # 设 self.net = net.to(device).eval()

    def _run(self, t):                                       # t: RGB float[0,1], (1,3,H,W), 不归一化
        _, _, h, w = t.shape
        H = max(self._min, ((h + self._mul - 1) // self._mul) * self._mul)
        W = max(self._min, ((w + self._mul - 1) // self._mul) * self._mul)
        t = F.pad(t, (0, W - w, 0, H - h), mode="replicate")
        out = self.net(t)                                   # 已核实: 返回单张量
        return out[..., :h, :w]                             # 裁回原尺寸


# ---- 备用: 若上面 strict=False 加载 missing/unexpected 很多, 用官方 Lightning 方式 ----
# 在 Colab 先 !pip -q install lightning, 然后:
#
# import lightning.pytorch as pl
# import torch.nn as nn
#
# class PromptIRModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.net = PromptIR(decoder=True)
#         self.loss_fn = nn.L1Loss()
#     def forward(self, x):
#         return self.net(x)
#
# 再把 AllInOneEnhancer.__init__ 里建 net 那几行换成:
#     net = PromptIRModel.load_from_checkpoint(weights_path, map_location=device).net
