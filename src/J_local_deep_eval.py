r"""
J_local_deep_eval.py -- LOCAL deep-restoration baselines on CURE-TSR
                        (enhancement + classification fused, CPU-friendly)

WHAT IT DOES
  For each of the 35,152 mapped CURE-TSR images (same filter as
  F_master_sweep_cache.py), on the LOCAL machine:
     read (BGR, original ~28x28) -> deep restoration model @ original size
     -> cv2.resize(32) -> BGR2RGB -> ToTensor -> Normalize(GTSRB mean/std)
     -> frozen CompactCNN -> record prediction.
  No Google Drive, no tar/MD5, no intermediate PNGs: one CSV of per-image
  predictions per model, keyed by FILENAME so it joins directly onto
  cure_master_cache.csv (which holds the four classical branches).

MODELS  (choose with --model)
  adair   : AdaIR (ICLR'25), all-in-one restoration. Needs the official
            5-degradation checkpoint placed locally (see --adair-weight).
  cidnet  : HVI-CIDNet (CVPR'25), low-light. Weights auto-downloaded from the
            official Hugging Face repo Fediory/HVI-CIDNet-Generalization, or
            supply a local file with --cidnet-weight.

CALIBRATION FIRST (recommended before committing to a full run)
     python J_local_deep_eval.py --model adair  --limit 100
  processes 100 images, prints the measured img/s and the EXTRAPOLATED time
  for the full 35,152, then stops. Run it once per model to get a real ETA
  on your CPU before launching the full sweep.

FULL RUN + RESUME
     python J_local_deep_eval.py --model cidnet
     python J_local_deep_eval.py --model adair            # the slow one
     python J_local_deep_eval.py --model adair --resume    # continue if stopped
  Rows are flushed continuously; --resume skips filenames already in the CSV.

PIPELINE PARITY
  Statistics/classification path is bit-identical to F_master_sweep_cache.py
  and the published harness (enhance at original size, then resize to 32).
  The deep model is applied to the ORIGINAL-size image, padded to a multiple
  of 8 with reflect padding, run in fp32, clamped to [0,1], then unpadded --
  matching the official CIDNet eval_hf.py and the AdaIR test path.

OUTPUT
  outputs_revision\deep_<model>_cure.csv        (filename-keyed per-image)
  outputs_revision\deep_<model>_cure.run_config.json
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as FP
from torchvision import transforms

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC_DIR = PROJECT_ROOT / "src"
THIRD_PARTY = PROJECT_ROOT / "third_party"
sys.path.insert(0, str(SRC_DIR))

from revision_utils import (                       # noqa: E402 (LIVE module)
    load_gtsrb_compactcnn, GTSRB_MEAN, GTSRB_STD,
)

CURE_TSR_DIR_DEFAULT = PROJECT_ROOT / "datasets" / "CURE-TSR"
OUT_DIR = PROJECT_ROOT / "outputs_revision"

INPUT_SIZE = 32
PAD_FACTOR = 8

FILENAME_PATTERN = re.compile(
    r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.(bmp|png|jpg|jpeg)$", re.IGNORECASE)
IMAGE_EXTS = ("*.bmp", "*.png", "*.jpg", "*.jpeg")

CHALLENGE_TYPES = {
    0: "ChallengeFree", 4: "Darkening", 8: "Noise",
    9: "Rain", 11: "Snow", 12: "Haze",
}
CURE_TO_GTSRB = {3: 9, 6: 14, 11: 12, 12: 17, 13: 13}
EVAL_CHALLENGES = [4, 8, 9, 11, 12]
EVAL_SEVERITIES = [1, 2, 3, 4, 5]

CSV_FIELDS = ["filename", "cure_sign", "gtsrb_true", "ch_id", "ch_name",
              "sev", "pred", "prob", "correct"]


# ------------------------------------------------------------------
# pure helpers (unit-tested; no torch needed)
# ------------------------------------------------------------------
def pad_amount(h, w, factor):
    ph = (factor - h % factor) % factor
    pw = (factor - w % factor) % factor
    return ph, pw


def iter_image_files(root):
    for ext in IMAGE_EXTS:
        yield from root.rglob(ext)


def collect_samples(cure_root):
    wanted_ch = set([0] + EVAL_CHALLENGES)
    samples = []
    n_files = 0
    for fpath in sorted(iter_image_files(cure_root),
                        key=lambda p: (p.name, str(p))):
        n_files += 1
        m = FILENAME_PATTERN.match(fpath.name)
        if not m:
            continue
        seq, sign, ch, sev = (int(m.group(1)), int(m.group(2)),
                              int(m.group(3)), int(m.group(4)))
        if seq != 1 or sign not in CURE_TO_GTSRB or ch not in wanted_ch:
            continue
        if ch != 0 and sev not in EVAL_SEVERITIES:
            continue
        samples.append({"path": fpath, "filename": fpath.name,
                        "sign": sign, "ch": ch, "sev": sev})
    return samples, n_files


def same_size_batches(metas, imgs, batch):
    """Yield (metas_chunk, imgs_chunk) where all imgs in a chunk share (H,W)
    and the chunk size is <= batch."""
    cur_m, cur_i, cur_hw = [], [], None
    for m, im in zip(metas, imgs):
        hw = (im.shape[0], im.shape[1])
        if cur_hw is None:
            cur_hw = hw
        if hw != cur_hw or len(cur_i) >= batch:
            yield cur_m, cur_i
            cur_m, cur_i, cur_hw = [], [], hw
        cur_m.append(m)
        cur_i.append(im)
    if cur_i:
        yield cur_m, cur_i


# ------------------------------------------------------------------
# model loading
# ------------------------------------------------------------------
def ensure_repo(url, dest):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] git clone {url}\n         -> {dest}")
    rc = os.system(f'git clone -q "{url}" "{dest}"')
    if rc != 0 or not dest.exists():
        sys.exit(f"[FATAL] git clone failed for {url}\n"
                 f"        Manually clone it into {dest} and rerun.")
    return dest


def _require(mod, hint):
    try:
        __import__(mod)
    except Exception:
        sys.exit(f"[FATAL] missing python package '{mod}'.\n"
                 f"        Install with:  {hint}")


def load_adair(device, weight_path):
    _require("einops", "pip install einops")
    repo = ensure_repo("https://github.com/c-yn/AdaIR",
                       THIRD_PARTY / "AdaIR")
    sys.path.insert(0, str(repo))
    from net.model import AdaIR
    wp = Path(weight_path)
    if not wp.exists():
        sys.exit(
            "[FATAL] AdaIR checkpoint not found at:\n"
            f"        {wp}\n"
            "        Download the FIVE-degradation all-in-one checkpoint "
            "(test mode 6) from the official README pretrained folder\n"
            "        https://drive.google.com/drive/folders/"
            "1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH\n"
            "        and put it there (or pass --adair-weight <path>).")
    net = AdaIR(decoder=True)
    ck = torch.load(str(wp), map_location="cpu")
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[4:] if k.startswith("net.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[adair] loaded {wp.name}  "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[adair][warn] first missing keys: {missing[:5]}")
        print("[adair][warn] if many keys are missing the checkpoint is the "
              "wrong file; stop and report.")
    net = net.to(device).eval()
    return net, "adair"


def load_cidnet(device, weight_path):
    _require("einops", "pip install einops")
    repo = ensure_repo("https://github.com/Fediory/HVI-CIDNet",
                       THIRD_PARTY / "HVI-CIDNet")
    sys.path.insert(0, str(repo))
    from net.CIDNet import CIDNet
    net = CIDNet()
    if weight_path:
        wp = Path(weight_path)
        if not wp.exists():
            sys.exit(f"[FATAL] --cidnet-weight not found: {wp}")
        if wp.suffix == ".safetensors":
            _require("safetensors", "pip install safetensors")
            import safetensors.torch as sf
            sd = sf.load_file(str(wp))
        else:
            ck = torch.load(str(wp), map_location="cpu")
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        net.load_state_dict(sd, strict=False)
        print(f"[cidnet] loaded local {wp.name}")
    else:
        _require("safetensors", "pip install safetensors")
        _require("huggingface_hub", "pip install huggingface_hub")
        import safetensors.torch as sf
        from huggingface_hub import hf_hub_download
        try:
            wf = hf_hub_download(repo_id="Fediory/HVI-CIDNet-Generalization",
                                 filename="model.safetensors")
        except Exception as ex:
            sys.exit("[FATAL] could not download CIDNet weights from Hugging "
                     f"Face ({ex}).\n        Provide a local file with "
                     "--cidnet-weight <path to .safetensors/.pth>.")
        sd = sf.load_file(wf)
        net.load_state_dict(sd, strict=False)
        print("[cidnet] loaded HF generalization weights")
    net = net.to(device).eval()
    # official eval_hf.py inference settings
    net.trans.alpha_s = 1.0
    net.trans.alpha = 1.0
    net.trans.gated = True
    net.trans.gated2 = True
    return net, "cidnet"


# ------------------------------------------------------------------
# enhancement (batch of equal-size images) + classification
# ------------------------------------------------------------------
def enhance_batch(net, imgs_bgr, device):
    """imgs_bgr: list of equal-size BGR uint8 arrays.
    Returns list of enhanced BGR uint8 arrays at the same size."""
    h, w = imgs_bgr[0].shape[:2]
    ph, pw = pad_amount(h, w, PAD_FACTOR)
    batch = []
    for im in imgs_bgr:
        t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) \
            .float().permute(2, 0, 1) / 255.0
        batch.append(t)
    x = torch.stack(batch, 0)
    if ph or pw:
        try:
            x = FP.pad(x, (0, pw, 0, ph), mode="reflect")
        except Exception:
            x = FP.pad(x, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        y = net(x.to(device))
    y = torch.clamp(y, 0, 1)[:, :, :h, :w].cpu()
    outs = []
    for k in range(y.shape[0]):
        arr = (y[k].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        outs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return outs


def classify_batch(model, enh_bgr, tfm, device):
    tens = []
    for im in enh_bgr:
        r = cv2.resize(im, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        tens.append(tfm(rgb))
    x = torch.stack(tens, 0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        top_p, top_i = probs.max(dim=1)
    return top_i.cpu().numpy(), top_p.cpu().numpy()


def build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])


# ------------------------------------------------------------------
def dump_samples(net, samples, device, model_name):
    """Seconds-long visual spot check: save side-by-side original|enhanced
    pairs for four representative images. Touches NO csv."""
    targets = [(12, 5, "Haze_sev5"), (4, 5, "Darkening_sev5"),
               (9, 5, "Rain_sev5"), (0, 0, "ChallengeFree")]
    out_dir = OUT_DIR / f"samples_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch, sev, tag in targets:
        pick = next((s for s in samples
                     if s["ch"] == ch and (ch == 0 or s["sev"] == sev)), None)
        if pick is None:
            print(f"[dump] no image found for {tag}")
            continue
        img = cv2.imread(str(pick["path"]))
        if img is None:
            print(f"[dump] unreadable: {pick['path']}")
            continue
        enh = enhance_batch(net, [img], device)[0]
        pair = np.concatenate([img, enh], axis=1)
        big = cv2.resize(pair, (pair.shape[1] * 6, pair.shape[0] * 6),
                         interpolation=cv2.INTER_NEAREST)
        p1 = out_dir / f"{tag}_{Path(pick['filename']).stem}_pair.png"
        cv2.imwrite(str(p1), big)
        print(f"[dump] {tag}: left=original right=enhanced -> {p1}")
    print("[dump] done. No CSV was touched.")


def run(args):
    device = "cpu"
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cure_root = Path(args.cure_root)
    out_csv = OUT_DIR / f"deep_{args.model}_cure.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples, n_files = collect_samples(cure_root)
    total_target = len(samples)
    print(f"[data] scanned {n_files} files, {total_target} usable "
          f"(target 35152)")
    if total_target == 0:
        sys.exit(f"[FATAL] no CURE images under {cure_root} "
                 "(point --cure-root at the parent CURE-TSR folder).")

    done = set()
    mode = "w"
    if args.resume and out_csv.exists():
        with open(out_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["filename"])
        mode = "a"
        print(f"[resume] {len(done)} rows already present")

    todo = [s for s in samples if s["filename"] not in done]
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
        print(f"[calib] limiting to {len(todo)} images for calibration")

    # load models
    if args.model == "adair":
        net, mname = load_adair(device, args.adair_weight)
    else:
        net, mname = load_cidnet(device, args.cidnet_weight)
    if args.dump_samples:
        dump_samples(net, samples, device, args.model)
        return

    model = load_gtsrb_compactcnn(device)
    tfm = build_transform()

    fout = open(out_csv, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()

    # read images in macro-batches, then split into equal-size sub-batches
    t0 = time.time()
    n_done = 0
    read_buf_m, read_buf_i = [], []

    def process(metas, imgs):
        nonlocal n_done
        for cm, ci in same_size_batches(metas, imgs, args.batch):
            enh = enhance_batch(net, ci, device)
            preds, probs = classify_batch(model, enh, tfm, device)
            for k, m in enumerate(cm):
                p = int(preds[k])
                t = CURE_TO_GTSRB[m["sign"]]
                writer.writerow({
                    "filename": m["filename"], "cure_sign": m["sign"],
                    "gtsrb_true": t, "ch_id": m["ch"],
                    "ch_name": CHALLENGE_TYPES[m["ch"]], "sev": m["sev"],
                    "pred": p, "prob": round(float(probs[k]), 6),
                    "correct": int(p == t),
                })
                n_done += 1
        fout.flush()

    MACRO = max(args.batch * 8, 256)
    for s in todo:
        img = cv2.imread(str(s["path"]))
        if img is None:
            print(f"[warn] unreadable, skipped: {s['path']}")
            continue
        read_buf_m.append(s)
        read_buf_i.append(img)
        if len(read_buf_i) >= MACRO:
            process(read_buf_m, read_buf_i)
            read_buf_m, read_buf_i = [], []
            dt = time.time() - t0
            rate = n_done / dt if dt > 0 else 0.0
            remain_full = total_target - len(done) - n_done
            eta_full = remain_full / rate / 60 if rate > 0 else float("nan")
            print(f"[sweep] {n_done} done  {rate:.2f} img/s  "
                  f"full-run ETA {eta_full:.1f} min "
                  f"({eta_full/60:.1f} h) for remaining {remain_full}")
    if read_buf_i:
        process(read_buf_m, read_buf_i)
    fout.close()

    dt = time.time() - t0
    rate = n_done / dt if dt > 0 else 0.0
    print(f"\n[done] processed {n_done} images in {dt/60:.1f} min "
          f"({rate:.2f} img/s)")

    if args.limit:
        remain_full = total_target - len(done) - n_done
        eta_full = remain_full / rate / 60 if rate > 0 else float("nan")
        print("\n=== CALIBRATION RESULT ===")
        print(f"  measured rate      : {rate:.2f} img/s")
        print(f"  images still to do : {remain_full}")
        print(f"  estimated full run : {eta_full:.1f} min "
              f"({eta_full/60:.2f} h)")
        print("  (re-run without --limit to do the full sweep; --resume "
              "continues from here)")
        return

    # config + quick self-summary (NOT an anchor: new method)
    cfg = {
        "script": "J_local_deep_eval.py", "model": args.model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cure_root": str(cure_root),
        "pad_factor": PAD_FACTOR,
        "pipeline": "imread->deep_enhance@orig(pad/8)->resize32->BGR2RGB"
                    "->ToTensor->Normalize->CompactCNN",
        "adair_weight": args.adair_weight if args.model == "adair" else None,
        "cidnet_weight": (args.cidnet_weight or "HF:Fediory/"
                          "HVI-CIDNet-Generalization")
        if args.model == "cidnet" else None,
        "n_rows_this_run": n_done,
        "torch": torch.__version__, "opencv": cv2.__version__,
    }
    with open(out_csv.with_suffix(".run_config.json"), "w",
              encoding="utf-8") as f:
        import json
        json.dump(cfg, f, indent=2)

    summarize(out_csv, args.model)


def summarize(out_csv, model_name):
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8", newline="")))
    for r in rows:
        r["ch_id"] = int(r["ch_id"])
        r["sev"] = int(r["sev"])
        r["correct"] = int(r["correct"])
    deg = [r for r in rows if r["ch_id"] != 0]
    cf = [r for r in rows if r["ch_id"] == 0]
    if not deg:
        print("[summary] no degraded rows yet.")
        return
    cells = defaultdict(lambda: [0, 0])
    for r in deg:
        cells[(r["ch_id"], r["sev"])][1] += 1
        cells[(r["ch_id"], r["sev"])][0] += r["correct"]
    deg_avg = sum(100.0 * c / n for c, n in cells.values()) / len(cells)
    print(f"\n=== {model_name} on CURE (this run) ===")
    print(f"  rows: degraded={len(deg)}  challengefree={len(cf)}")
    print(f"  degraded-average accuracy (cell-averaged): {deg_avg:.2f}")
    if cf:
        cf_acc = 100.0 * sum(r["correct"] for r in cf) / len(cf)
        print(f"  ChallengeFree accuracy: {cf_acc:.2f}")
    print("  per-challenge (mean over severities):")
    for ch in EVAL_CHALLENGES:
        per = defaultdict(lambda: [0, 0])
        for r in deg:
            if r["ch_id"] == ch:
                per[r["sev"]][1] += 1
                per[r["sev"]][0] += r["correct"]
        if per:
            v = sum(100.0 * c / n for c, n in per.values()) / len(per)
            print(f"    {CHALLENGE_TYPES[ch]:12s} {v:6.2f}")
    print("\n[note] this is one method's own numbers; comparison vs baseline "
          "57.32/52.56 etc. happens at the merge step. Send me this output "
          "plus deep_" + model_name + "_cure.csv (zipped) when both models "
          "are done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["adair", "cidnet"])
    ap.add_argument("--cure-root", default=str(CURE_TSR_DIR_DEFAULT))
    ap.add_argument("--adair-weight",
                    default=str(PROJECT_ROOT / "models" / "adair_5task.ckpt"))
    ap.add_argument("--cidnet-weight", default="",
                    help="optional local CIDNet weight; default = HF download")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0,
                    help=">0 to pin torch CPU threads")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only N images and print a full-run ETA")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dump-samples", action="store_true",
                    help="save 4 original|enhanced sample pairs and exit")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
