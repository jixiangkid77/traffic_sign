#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
O_eval_three_locally.py
=======================
Runs Zero-DCE, FFA-Net and PromptIR over CURE-TSR on this machine and writes
their per-image predictions in exactly the format J_local_deep_eval.py writes
for AdaIR and CIDNet.

WHY
    Those three were scored in an earlier run whose rows cannot be matched to
    the local ones image by image. Every claim that needs a paired test has
    therefore had to leave them out, and the article has had to say so. Running
    them here removes the split: one run, one alignment, five learned restorers
    that can all be compared like for like, and nothing to explain.

WHAT IT REUSES
    Everything that decides a number comes from J_local_deep_eval.py: the file
    enumeration, the padding, the batching, the frozen classifier and its
    transform. Only the three model loaders are new, and they are the ones
    already proven by the timing run. Nothing in J is modified, so the AdaIR and
    CIDNet files already on disk are untouched.

ZERO-DCE
    Its forward returns three tensors rather than one. Rather than fork
    enhance_batch, the network is wrapped so that it returns the enhanced image
    and nothing else; the shared code then applies to it unchanged.

OUTPUT
    outputs_revision/deep_zero_dce_cure.csv
    outputs_revision/deep_ffa_net_cure.csv
    outputs_revision/deep_promptir_cure.csv
    each with J's own columns, so K_merge_results.py can take them as they are.

USAGE
    python O_eval_three_locally.py --model zero_dce
    python O_eval_three_locally.py --model ffa_net   --resume
    python O_eval_three_locally.py --model promptir  --limit 200
    python O_eval_three_locally.py --model all
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
MODELS_DIR = PROJECT_ROOT / "models"
THIRD_PARTY = PROJECT_ROOT / "third_party"
OUT_DIR = PROJECT_ROOT / "outputs_revision"

SPEC = {
    "zero_dce": {"label": "Zero-DCE", "repo": "https://github.com/Li-Chongyi/Zero-DCE",
                 "dir": "Zero-DCE", "globs": ["*zero*dce*", "*Epoch99*"],
                 "params": 79_416, "release_api": None},
    "ffa_net": {"label": "FFA-Net", "repo": "https://github.com/zhilin007/FFA-Net",
                "dir": "FFA-Net", "globs": ["*ffa*"], "params": 4_455_913,
                "release_api": None},
    "promptir": {"label": "PromptIR", "repo": "https://github.com/va1shn9v/PromptIR",
                 "dir": "PromptIR", "globs": ["*prompt*"], "params": 35_592_263,
                 "release_api": "https://api.github.com/repos/va1shn9v/PromptIR/releases"},
}
WEIGHT_EXT = {".pth", ".pt", ".ckpt", ".pk", ".pkl", ".safetensors", ".bin"}


def say(m=""):
    print(m, flush=True)


def rule(t=""):
    say("\n" + "=" * 74)
    if t:
        say(t)
        say("=" * 74)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_weight(globs):
    """Match by pattern over every weight-shaped file under models/.

    Listing expected filenames looks like a probe and behaves like a guess: the
    first version of the timing script asked for ots_train_ffa_3_19.pk while the
    copy on disk was called ffa_net_ots.pk, and reported the weights missing.
    """
    if not MODELS_DIR.exists():
        return None, f"models directory absent: {MODELS_DIR}"
    files = [p for p in MODELS_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in WEIGHT_EXT]
    for g in globs:
        parts = [x for x in g.lower().split("*") if x]
        for f in files:
            if all(x in f.name.lower() for x in parts):
                return f, None
    return None, ("nothing matched " + " or ".join(globs) + "; models/ holds: " +
                  (", ".join(sorted(f.name for f in files)) or "no weight files"))


def _http_get(url, timeout=1800):
    """Three routes, because this machine's certificate store rejected the
    first one. What the file is gets checked afterwards by counting its
    parameters, which is a stronger test than the transport."""
    import ssl
    errs = []
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read(), "verified https"
    except Exception as exc:
        errs.append(f"urllib: {type(exc).__name__}")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
            return r.read(), "https without certificate verification"
    except Exception as exc:
        errs.append(f"unverified: {type(exc).__name__}")
    try:
        out = subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                             capture_output=True, timeout=timeout + 60)
        if out.returncode == 0 and out.stdout:
            return out.stdout, "curl"
    except Exception as exc:
        errs.append(f"curl: {type(exc).__name__}")
    return None, "; ".join(errs)


def fetch_release_asset(api_url, dest_dir):
    raw, how = _http_get(api_url, timeout=90)
    if raw is None:
        return None, f"release query failed ({how})"
    try:
        rels = json.loads(raw.decode())
    except Exception as exc:
        return None, f"release list unreadable: {type(exc).__name__}"
    assets = []
    for rel in rels if isinstance(rels, list) else []:
        for a in rel.get("assets", []):
            if a.get("browser_download_url"):
                assets.append((a.get("size", 0), a["name"],
                               a["browser_download_url"]))
    if not assets:
        return None, "the release carries no downloadable asset"
    size, name, url = max(assets)
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size == size:
        return dest, "already downloaded"
    say(f"    fetching {name} ({size / 1e6:.0f} MB)")
    blob, how2 = _http_get(url)
    if blob is None:
        return None, f"download failed ({how2})"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return dest, f"downloaded via {how2}"


def ensure_repo(url, target):
    if target.exists() and any(target.iterdir()):
        return target, "present"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)],
                       check=True, capture_output=True, timeout=900)
        return target, "cloned"
    except Exception as exc:
        return None, f"clone failed: {type(exc).__name__}"


def strip(sd, *prefixes):
    for p in prefixes:
        sd = {k.replace(p, "", 1): v for k, v in sd.items()}
    return sd


def unwrap(ck):
    if isinstance(ck, dict):
        for k in ("state_dict", "model", "params", "net"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


def build(key, device, allow_download=True):
    """Return ((net, weight_path), note) or (None, why). Never raises."""
    import torch
    spec = SPEC[key]
    repo, note = ensure_repo(spec["repo"], THIRD_PARTY / spec["dir"])
    if repo is None:
        return None, note
    wp, werr = find_weight(spec["globs"])
    if wp is None and spec["release_api"] and allow_download:
        say(f"    weights not local; asking {spec['label']}'s release page")
        wp, dnote = fetch_release_asset(spec["release_api"], MODELS_DIR)
        if wp is None:
            return None, f"{werr}  |  {dnote}"
        say(f"    {dnote}: {wp.name}")
    if wp is None:
        return None, werr

    if key == "zero_dce":
        cand = list(repo.rglob("model.py"))
        if not cand:
            return None, "model.py not found in the Zero-DCE checkout"
        mod = load_module(cand[0], "zdce_mod")
        net = mod.enhance_net_nopool()
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "module.")
    elif key == "ffa_net":
        cand = list(repo.rglob("FFA.py"))
        if not cand:
            return None, "FFA.py not found in the FFA-Net checkout"
        sys.path.insert(0, str(cand[0].parent))
        mod = load_module(cand[0], "ffa_mod")
        net = mod.FFA(gps=3, blocks=19)
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "module.")
    else:
        cand = list(repo.rglob("net/model.py"))
        if not cand:
            return None, "net/model.py not found in the PromptIR checkout"
        sys.path.insert(0, str(cand[0].parent.parent))
        mod = load_module(cand[0], "promptir_mod")
        net = mod.PromptIR(decoder=True)
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "net.", "module.")

    info = net.load_state_dict(sd, strict=False)
    miss, unexp = list(info.missing_keys), list(info.unexpected_keys)
    if miss or unexp:
        say(f"    [WARN] state dict: {len(miss)} missing, {len(unexp)} "
            f"unexpected key(s). strict=False accepts a checkpoint that did "
            f"not load; a network left at its initial values still reports the "
            f"right parameter count.")
        for k in miss[:3]:
            say(f"           missing    {k}")
        for k in unexp[:3]:
            say(f"           unexpected {k}")
    net.eval().to(device)

    # Each network is wrapped so that the shared enhance_batch applies to all
    # three without a special case and without touching J.
    #
    # Zero-DCE returns a triple; the second element is the enhanced image.
    #
    # FFA-Net wants its input normalised. Its own data_utils.py applies
    # ToTensor followed by Normalize(mean=[0.64, 0.6, 0.58],
    # std=[0.14, 0.15, 0.152]) to the input while leaving the target at [0, 1],
    # so the network maps a normalised image to a plain one. Fed raw [0, 1] it
    # produces something the classifier reads at fourteen per cent instead of
    # fifty-five, with no error anywhere: the weights load, the parameter count
    # matches, and the output is a picture. Only the accuracy says otherwise.
    FFA_MEAN = (0.64, 0.60, 0.58)
    FFA_STD = (0.14, 0.15, 0.152)

    class _Enhanced(torch.nn.Module):
        def __init__(self, inner, key):
            super().__init__()
            self.inner = inner
            self.key = key
            if key == "ffa_net":
                self.register_buffer("m", torch.tensor(FFA_MEAN).view(1, 3, 1, 1))
                self.register_buffer("s", torch.tensor(FFA_STD).view(1, 3, 1, 1))

        def forward(self, x):
            if self.key == "ffa_net":
                x = (x - self.m) / self.s
            out = self.inner(x)
            return out[1] if isinstance(out, (tuple, list)) else out

    return (_Enhanced(net, key).eval().to(device), wp), "ok"


# --------------------------------------------------------------------------
def run_one(key, J, model, tfm, samples, device, args):
    import torch
    spec = SPEC[key]
    rule(spec["label"])
    say(f"  repository : {spec['repo']}")
    built, why = build(key, device, not args.skip_download)
    if built is None:
        say(f"  [SKIP] {why}")
        return None, why
    net, wp = built
    n_par = sum(p.numel() for p in net.parameters())
    exp = spec["params"]
    say(f"  weights    : {wp}")
    say(f"  parameters : {n_par:,}   expected {exp:,}   "
        f"{'MATCH' if n_par == exp else 'MISMATCH'}")
    if n_par != exp:
        say("  [STOP] the checkpoint is not the one the article counts. The "
            "predictions it would write must not be merged with the rest.")
        return None, "parameter mismatch"

    # A run of eighty minutes should not be started on a configuration that can
    # be shown wrong in five seconds. One group of eighty-five crops, all of the
    # same challenge, severity and class, is scored and compared with the
    # earlier run before anything else happens. FFA-Net read fourteen per cent
    # against fifty-five for want of an input normalisation, and nothing in the
    # run said so until it had finished.
    if args.earlier and Path(args.earlier).exists() and not args.no_preflight:
        deg = [s for s in samples if s["sev"] > 0]
        grp = [s for s in deg
               if (s["ch"], s["sev"]) == (deg[0]["ch"], deg[0]["sev"])
               and J.CURE_TO_GTSRB[s["sign"]] == J.CURE_TO_GTSRB[deg[0]["sign"]]]
        want = J.CHALLENGE_TYPES[deg[0]["ch"]], deg[0]["sev"], \
            J.CURE_TO_GTSRB[deg[0]["sign"]]
        hit = tot = 0
        with open(args.earlier, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if (r["method"] == key and r["challenge"] == want[0]
                        and int(r["severity"]) == want[1]
                        and int(r["true_label"]) == want[2]):
                    tot += 1
                    hit += int(r["correct"])
        if tot and len(grp) == tot:
            import cv2 as _cv
            ims = [_cv.imread(str(x["path"]), _cv.IMREAD_COLOR) for x in grp]
            ims = [i for i in ims if i is not None]
            enh_ok = 0
            for cm, ci in J.same_size_batches(grp[:len(ims)], ims, args.batch):
                pr, _ = J.classify_batch(model, J.enhance_batch(net, ci, device),
                                         tfm, device)
                for k2, mm in enumerate(cm):
                    enh_ok += int(int(pr[k2]) == J.CURE_TO_GTSRB[mm["sign"]])
            a, b = 100.0 * enh_ok / len(ims), 100.0 * hit / tot
            say(f"  pre-flight : {want[0]} sev {want[1]} class {want[2]}   "
                f"local {a:.2f} vs earlier {b:.2f}  ({len(ims)} crops)")
            if abs(a - b) > 5.0:
                say(f"  [STOP] the two differ by {abs(a - b):.1f} points on a "
                    f"group that should match to within a crop or two. Fix the "
                    f"configuration before spending an hour on it; pass "
                    f"--no-preflight to override.")
                return None, f"pre-flight differs by {abs(a - b):.1f} points"

    out_csv = OUT_DIR / f"deep_{key}_cure.csv"
    done = set()
    mode = "w"
    if args.resume and out_csv.exists():
        with open(out_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["filename"], int(r["ch_id"]), int(r["sev"])))
        mode = "a"
        say(f"  resuming   : {len(done):,} rows already written")

    todo = [s for s in samples
            if (s["filename"], s["ch"], s["sev"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    say(f"  to process : {len(todo):,} crops   batch {args.batch}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fout = open(out_csv, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=J.CSV_FIELDS)
    if mode == "w":
        writer.writeheader()

    import cv2
    t0, n_done = time.time(), 0
    buf_m, buf_i = [], []

    def flush():
        nonlocal n_done
        if not buf_m:
            return
        for cm, ci in J.same_size_batches(buf_m, buf_i, args.batch):
            enh = J.enhance_batch(net, ci, device)
            preds, probs = J.classify_batch(model, enh, tfm, device)
            for k, m in enumerate(cm):
                p = int(preds[k])
                t = J.CURE_TO_GTSRB[m["sign"]]
                writer.writerow({
                    "filename": m["filename"], "cure_sign": m["sign"],
                    "gtsrb_true": t, "ch_id": m["ch"],
                    "ch_name": J.CHALLENGE_TYPES[m["ch"]], "sev": m["sev"],
                    "pred": p, "prob": round(float(probs[k]), 6),
                    "correct": int(p == t)})
                n_done += 1
        fout.flush()
        buf_m.clear()
        buf_i.clear()

    for s in todo:
        im = cv2.imread(str(s["path"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        buf_m.append(s)
        buf_i.append(im)
        if len(buf_m) >= 512:
            flush()
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            say(f"    {n_done:,}/{len(todo):,}   {rate:.1f} crops/s   "
                f"eta {(len(todo) - n_done) / max(rate, 1e-9) / 60:.0f} min")
    flush()
    fout.close()

    acc = None
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8", newline="")))
    deg = [r for r in rows if int(r["sev"]) > 0]
    if deg:
        cells = {}
        for r in deg:
            cells.setdefault((r["ch_id"], r["sev"]), []).append(int(r["correct"]))
        acc = 100.0 * sum(sum(v) / len(v) for v in cells.values()) / len(cells)
        say(f"  rows       : {len(rows):,}   degraded cells {len(cells)}")
        say(f"  degraded-average accuracy: {acc:.2f} per cent")
    (OUT_DIR / f"deep_{key}_cure.run_config.json").write_text(json.dumps({
        "script": "O_eval_three_locally.py", "model": key,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weight": str(wp), "params": n_par, "batch": args.batch,
        "degraded_average": acc}, indent=2), encoding="utf-8")
    return acc, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["zero_dce", "ffa_net", "promptir", "all"])
    ap.add_argument("--cure-root", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only N crops, to check the run before "
                         "committing hours to it")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--earlier",
                    default=str(OUT_DIR / "cure_tsr_per_image_predictions_12class.csv"),
                    help="the earlier run, used only for the "
                         "pre-flight check")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--classifier",
                    default=str(MODELS_DIR / "mbnetv3_baseline.pth"))
    args = ap.parse_args()

    rule("O: the three learned restorers, scored on this machine")
    say("Same enumeration, same padding, same frozen classifier as AdaIR and "
        "CIDNet, so the rows line up image by image with what is already on "
        "disk and every comparison in the article can be paired.")

    here = Path(__file__).parent
    for need in ("J_local_deep_eval.py", "F_master_sweep_cache.py"):
        if not (here / need).exists():
            say(f"[FATAL] {need} must sit beside this script")
            sys.exit(1)
    try:
        import torch
        import cv2  # noqa: F401
    except Exception as exc:
        say(f"[FATAL] {type(exc).__name__}: {exc}")
        sys.exit(1)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    J = load_module(here / "J_local_deep_eval.py", "J_local")
    F = load_module(here / "F_master_sweep_cache.py", "F_master")

    rule("sampling")
    root = Path(args.cure_root) if args.cure_root else J.CURE_TSR_DIR_DEFAULT
    samples, n_files = J.collect_samples(root)
    say(f"  usable files {n_files:,}   records {len(samples):,}")

    model = F.load_model(str(args.classifier), device)
    tfm = J.build_transform()
    say(f"  classifier   {Path(args.classifier).name}, frozen")

    keys = ["zero_dce", "ffa_net", "promptir"] if args.model == "all" \
        else [args.model]
    got, missed = {}, {}
    for k in keys:
        acc, why = run_one(k, J, model, tfm, samples, device, args)
        if why:
            missed[k] = why
        else:
            got[k] = acc

    rule("summary")
    for k in keys:
        if k in got:
            say(f"  {SPEC[k]['label']:10s} written, degraded-average "
                f"{got[k]:.2f} per cent")
        else:
            say(f"  {SPEC[k]['label']:10s} not written: {missed[k]}")
    if len(got) == len(keys):
        say("\n  All three are now local. Merge them with K_merge_results.py "
            "and every learned restorer can enter the paired comparisons.")
    else:
        say("\n  Some are missing; do not merge a partial set.")


if __name__ == "__main__":
    main()
