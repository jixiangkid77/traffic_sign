# diag_params.py  诊断各权重文件的顶层结构与参数计数
import torch
from pathlib import Path

MODELS = Path(r"D:\Project\traffic_sign\models")

def count_tensors(d):
    return sum(v.numel() for v in d.values() if torch.is_tensor(v))

for p in sorted(MODELS.iterdir()):
    if p.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
        continue
    try:
        ck = torch.load(str(p), map_location="cpu")
    except Exception as e:
        print(f"{p.name}: LOAD FAIL: {e}")
        continue
    print(f"\n===== {p.name} =====")
    if torch.is_tensor(ck):
        print("  raw tensor:", tuple(ck.shape))
        continue
    if not isinstance(ck, dict):
        print("  type:", type(ck).__name__)
        continue
    keys = list(ck.keys())
    print(f"  top-level keys ({len(keys)}): {keys[:10]}{' ...' if len(keys) > 10 else ''}")
    print(f"  direct tensor count: {count_tensors(ck):,}")
    for nest in ("state_dict", "model", "params", "net", "g",
                 "model_state_dict", "params_ema", "ema", "network", "weight"):
        if nest in ck and isinstance(ck[nest], dict):
            print(f"  nested['{nest}'] tensor count: {count_tensors(ck[nest]):,}")