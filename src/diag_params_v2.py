r"""
diag_params_v2.py  权重文件参数诊断, 修正 v1 的两个盲区:
  1. v1 只扫 .pth / .pt / .ckpt, 漏掉了 ffa_net_ots.pk 这类扩展名。
     v2 扫 models\ 下所有常见权重扩展名, 含 .pk / .pkl / .safetensors。
  2. v1 只看顶层 tensor。v2 递归解包常见嵌套键
     (state_dict / model / params / params_ema / model_state / net),
     并把参数拆成 learnable 与 buffer 两栏
     (buffer = 键名含 running_mean / running_var / num_batches_tracked)。

USAGE
    python diag_params_v2.py
"""

import sys
from pathlib import Path

import torch

MODELS = Path(r"D:\Project\traffic_sign\models")
SUFFIXES = {".pth", ".pt", ".ckpt", ".pk", ".pkl", ".bin", ".safetensors"}
NEST_KEYS = ("state_dict", "model", "params", "params_ema",
             "model_state", "net", "g")
BUFFER_MARKS = ("running_mean", "running_var", "num_batches_tracked")


def unwrap(ck):
    if not isinstance(ck, dict):
        return None, "not a dict"
    if any(hasattr(v, "numel") for v in ck.values()):
        return ck, "top-level"
    for key in NEST_KEYS:
        v = ck.get(key)
        if isinstance(v, dict) and any(hasattr(t, "numel")
                                       for t in v.values()):
            return v, f"nested['{key}']"
    return None, f"no tensors under top level or {NEST_KEYS}"


def split_count(sd):
    learn, buf = 0, 0
    for k, v in sd.items():
        if not hasattr(v, "numel"):
            continue
        if any(m in k for m in BUFFER_MARKS):
            buf += v.numel()
        else:
            learn += v.numel()
    return learn, buf


def load_any(p):
    if p.suffix == ".safetensors":
        import safetensors.torch as sf
        return sf.load_file(str(p))
    return torch.load(str(p), map_location="cpu")


def main():
    files = sorted(x for x in MODELS.iterdir()
                   if x.is_file() and x.suffix.lower() in SUFFIXES)
    if not files:
        sys.exit(f"no weight files under {MODELS}")
    print(f"{'file':32s} {'where':18s} {'learnable':>12s} "
          f"{'buffers':>9s} {'total':>12s}")
    print("-" * 90)
    for p in files:
        try:
            ck = load_any(p)
        except Exception as ex:
            print(f"{p.name:32s} LOAD FAIL: {type(ex).__name__}: {ex}")
            continue
        sd, where = unwrap(ck)
        if sd is None:
            keys = list(ck.keys())[:8] if isinstance(ck, dict) else "n/a"
            print(f"{p.name:32s} UNRESOLVED ({where}); top keys: {keys}")
            continue
        learn, buf = split_count(sd)
        print(f"{p.name:32s} {where:18s} {learn:12,d} "
              f"{buf:9,d} {learn + buf:12,d}")


if __name__ == "__main__":
    main()
