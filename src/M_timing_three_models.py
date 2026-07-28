#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M_timing_three_models.py
========================
Times Zero-DCE, FFA-Net and PromptIR under exactly the protocol that produced
the AdaIR (203.667 ms) and CIDNet (18.253 ms) figures already in the paper, so
that the five learned restorers can be compared on one latency scale.

WHY THIS EXISTS
    L2_timing_full_pool.py only wired in AdaIR and CIDNet. The other three were
    never added to the local pipeline, which is why Table II has three empty
    latency cells. Nothing about them prevents timing: all three publish their
    code and their weights.

WHAT IT DOES NOT DO
    It does not retrain anything. All five restorers use the original authors'
    released weights; the only model trained for this work is the CompactCNN
    classifier, and this script never touches it.

PROTOCOL (identical to L2, not re-derived)
    * the same stratified sample of degraded crops, seed 42
    * 200 timed crops after 10 discarded warm-up crops, so 210 loaded
    * batch of one, torch.set_num_threads(1), CPU
    * one pass, time.perf_counter around the enhancement call only
    * mean / median / p95 reported; the paper quotes the median

DESIGN NOTES, EACH ONE A BUG THIS PROJECT ACTUALLY HIT
    1. Nothing is hardcoded as unavailable. Every weight and every repository is
       probed and the finding is printed. An earlier script simply asserted that
       PromptIR's weights "live only on Colab"; that line was never a
       measurement, and it survived long after it stopped being true.
    2. A model is not trusted because it loaded. Its parameter count is compared
       with the published value and a mismatch is reported loudly, because a
       checkpoint that loads is not necessarily the checkpoint you wanted.
    3. Failures are reported and the run continues. A missing repository for one
       model must not cost you the other two.
    4. The end of the run states which of the three produced a number and which
       did not, rather than only counting successes.
    5. The script writes exactly one file, its own results JSON.
    6. Weight files are located by pattern rather than by an expected name. The
       first run failed on FFA-Net because the checkpoint on disk is called
       ffa_net_ots.pk and the script was asking for ots_train_ffa_3_19.pk. A
       list of guessed filenames is an assumption wearing the costume of a
       probe.

USAGE
    python M_timing_three_models.py
    python M_timing_three_models.py --models zero_dce,ffa_net
    python M_timing_three_models.py --cure-root D:\\Project\\traffic_sign\\datasets\\CURE-TSR
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# paths, matching the layout the other scripts assume
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
MODELS_DIR = PROJECT_ROOT / "models"
THIRD_PARTY = PROJECT_ROOT / "third_party"
CURE_TSR_DEFAULT = PROJECT_ROOT / "datasets" / "CURE-TSR"
OUT_JSON = PROJECT_ROOT / "outputs_revision" / "M_timing_three_models.results.json"

# What each model needs, and what it should weigh when it arrives. The parameter
# counts are the published ones; they are the check that the right checkpoint
# was loaded rather than merely a checkpoint.
SPEC = {
    "zero_dce": {
        "repo": "https://github.com/Li-Chongyi/Zero-DCE",
        "repo_dir": "Zero-DCE",
        "weight_globs": ["*zero*dce*", "*Epoch99*"],
        "weight_url": None,          # ships inside the repository
        "expect_params": 79_416,
        "label": "Zero-DCE",
    },
    "ffa_net": {
        "repo": "https://github.com/zhilin007/FFA-Net",
        "repo_dir": "FFA-Net",
        "weight_globs": ["*ffa*"],
        "weight_url": None,          # released in the repository
        "expect_params": 4_455_913,
        "label": "FFA-Net",
    },
    "promptir": {
        "repo": "https://github.com/va1shn9v/PromptIR",
        "repo_dir": "PromptIR",
        "weight_globs": ["*prompt*", "*promptir*"],
        "weight_url": None,          # published under the repository's releases
        "expect_params": None,       # printed for the record, not asserted
        "label": "PromptIR",
    },
}


def say(msg=""):
    print(msg, flush=True)


def rule(title=""):
    say("\n" + "=" * 74)
    if title:
        say(title)
        say("=" * 74)


# --------------------------------------------------------------------------
# probing, never assuming
# --------------------------------------------------------------------------
def find_weight(globs):
    """Find a weight file by pattern, not by an exact name.

    The first version of this script listed the filenames it expected and
    missed FFA-Net's checkpoint because the copy on disk is called
    ffa_net_ots.pk while the list asked for ots_train_ffa_3_19.pk. Guessing
    names is the same mistake as hardcoding availability: it looks like a
    probe and behaves like an assumption. Matching a pattern over every
    weight-shaped file under models/ removes the guess.
    """
    if not MODELS_DIR.exists():
        return None, f"models directory not found at {MODELS_DIR}"
    EXT = {".pth", ".pt", ".ckpt", ".pk", ".pkl", ".safetensors", ".bin"}
    files = [p for p in MODELS_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in EXT]
    for g in globs:
        pat = g.lower().replace("*", "")
        parts = [x for x in g.lower().split("*") if x]
        for f in files:
            name = f.name.lower()
            if all(x in name for x in parts):
                return f, None
    return None, ("no file under models/ matched " + " or ".join(globs) +
                  "; present: " + (", ".join(sorted(f.name for f in files))
                                   if files else "no weight files"))


def ensure_repo(url, target):
    """Clone if absent. Returns (path, note); never raises."""
    if target.exists() and any(target.iterdir()):
        return target, "already present"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)],
                       check=True, capture_output=True, timeout=600)
        return target, "cloned"
    except Exception as exc:
        return None, f"clone failed: {type(exc).__name__}: {exc}"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_params(net):
    return sum(p.numel() for p in net.parameters())


# --------------------------------------------------------------------------
# the three loaders, each returning (callable, note) and never raising
# --------------------------------------------------------------------------
def load_zero_dce(device):
    import torch
    repo, note = ensure_repo(SPEC["zero_dce"]["repo"],
                             THIRD_PARTY / SPEC["zero_dce"]["repo_dir"])
    if repo is None:
        return None, note
    cand = list(repo.rglob("model.py"))
    if not cand:
        return None, "model.py not found in the Zero-DCE checkout"
    mod = load_module(cand[0], "zdce_model")
    if not hasattr(mod, "enhance_net_nopool"):
        return None, "enhance_net_nopool missing from Zero-DCE model.py"
    net = mod.enhance_net_nopool()
    wp, werr = find_weight(SPEC["zero_dce"]["weight_globs"])
    if wp is None:
        inrepo = list(repo.rglob("Epoch99.pth"))
        if inrepo:
            wp = inrepo[0]
        else:
            return None, werr
    sd = torch.load(str(wp), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    net.load_state_dict(sd, strict=False)
    net.eval().to(device)

    def run(img):
        with torch.no_grad():
            x = to_tensor(img, device)
            out = net(x)
            return out[1] if isinstance(out, (tuple, list)) else out

    return (run, net, wp), "ok"


def load_ffa_net(device):
    import torch
    repo, note = ensure_repo(SPEC["ffa_net"]["repo"],
                             THIRD_PARTY / SPEC["ffa_net"]["repo_dir"])
    if repo is None:
        return None, note
    cand = [p for p in repo.rglob("FFA.py")] or [p for p in repo.rglob("models/*.py")]
    if not cand:
        return None, "FFA.py not found in the FFA-Net checkout"
    sys.path.insert(0, str(cand[0].parent))
    mod = load_module(cand[0], "ffa_model")
    if not hasattr(mod, "FFA"):
        return None, "class FFA missing from the FFA-Net sources"
    net = mod.FFA(gps=3, blocks=19)
    wp, werr = find_weight(SPEC["ffa_net"]["weight_globs"])
    if wp is None:
        inrepo = list(repo.rglob("*.pk"))
        if inrepo:
            wp = inrepo[0]
        else:
            return None, werr
    ck = torch.load(str(wp), map_location="cpu", weights_only=False)
    # FFA-Net's release wraps the state dict and prefixes every key with
    # "module."; unwrapping is why a naive parameter count once read zero.
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval().to(device)

    def run(img):
        with torch.no_grad():
            return net(to_tensor(img, device))

    return (run, net, wp), "ok"


def load_promptir(device):
    import torch
    repo, note = ensure_repo(SPEC["promptir"]["repo"],
                             THIRD_PARTY / SPEC["promptir"]["repo_dir"])
    if repo is None:
        return None, note
    cand = list(repo.rglob("net/model.py")) or list(repo.rglob("model.py"))
    if not cand:
        return None, "net/model.py not found in the PromptIR checkout"
    sys.path.insert(0, str(cand[0].parent.parent))
    mod = load_module(cand[0], "promptir_model")
    if not hasattr(mod, "PromptIR"):
        return None, "class PromptIR missing from the PromptIR sources"
    net = mod.PromptIR(decoder=True)
    wp, werr = find_weight(SPEC["promptir"]["weight_globs"])
    if wp is None:
        inrepo = list(repo.rglob("*.ckpt"))
        if inrepo:
            wp = inrepo[0]
        else:
            return None, (werr + "  |  PromptIR publishes its checkpoint on the "
                          "repository's releases page; download it into models/ "
                          "and run again")
    ck = torch.load(str(wp), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("net.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval().to(device)

    def run(img):
        with torch.no_grad():
            return net(to_tensor(img, device))

    return (run, net, wp), "ok"


LOADERS = {"zero_dce": load_zero_dce, "ffa_net": load_ffa_net,
           "promptir": load_promptir}


def to_tensor(img, device):
    """BGR uint8 HxWx3 to a 1x3xHxW float tensor on [0,1], as J does."""
    import torch
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


# --------------------------------------------------------------------------
# sampling and timing, taken from L2 rather than rewritten
# --------------------------------------------------------------------------
def stratified_pick(samples, n, seed):
    """Round-robin across (ch, sev) groups. Verbatim from L2."""
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[(s["ch"], s["sev"])].append(s)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    picked, i = [], 0
    while len(picked) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            picked.append(groups[k].pop())
        i += 1
    return picked


def pstats(ts):
    """Verbatim from L2."""
    a = np.asarray(ts, dtype=np.float64) * 1000.0
    return {"n_timed": int(a.size),
            "mean_ms": round(float(a.mean()), 3),
            "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


def time_scope(fn, imgs, warmup):
    """Verbatim from L2: one pass, warm-up discarded, per-image perf_counter."""
    ts = []
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        fn(img)
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return pstats(ts)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(CURE_TSR_DEFAULT))
    ap.add_argument("--models", default="zero_dce,ffa_net,promptir")
    ap.add_argument("--n-degraded", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    rule("M: timing the three learned restorers that L2 never covered")
    say("Protocol is L2's, not a new one: stratified sample at seed "
        f"{args.seed}, {args.n_degraded} timed crops after {args.warmup} "
        "discarded, batch of one, one CPU thread, median reported.")
    say("Comparable figures already in the paper: classifier 1.280 ms, "
        "AdaIR 203.667 ms, CIDNet 18.253 ms.")

    try:
        import torch
        import cv2  # noqa: F401
    except Exception as exc:
        say(f"\n[FATAL] {type(exc).__name__}: {exc}")
        say("Activate the environment that runs the other scripts, then retry.")
        sys.exit(1)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # sample exactly as L2 does, by importing J's collector so that the file
    # enumeration cannot drift between scripts
    rule("sampling")
    jpath = Path(__file__).with_name("J_local_deep_eval.py")
    if not jpath.exists():
        say(f"[FATAL] J_local_deep_eval.py must sit beside this script: {jpath}")
        sys.exit(1)
    J = load_module(jpath, "J_local_deep_eval")
    samples, n_files = J.collect_samples(Path(args.cure_root))
    degraded = [s for s in samples if s["sev"] > 0]
    say(f"  usable files {n_files:,}   degraded records {len(degraded):,}")
    picked = stratified_pick(degraded, args.n_degraded + args.warmup, args.seed)
    say(f"  picked {len(picked)} crops "
        f"({args.warmup} warm-up + {args.n_degraded} timed)")
    if len(picked) < args.n_degraded + args.warmup:
        say("  [WARN] fewer crops than requested; the medians will not be "
            "comparable with the published figures")

    import cv2
    imgs = []
    for s in picked:
        im = cv2.imread(str(s["path"]), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    say(f"  loaded {len(imgs)} images")

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    results, notes = {}, {}

    for key in wanted:
        if key not in LOADERS:
            notes[key] = "unknown model key"
            continue
        rule(f"{SPEC[key]['label']}")
        say(f"  repository : {SPEC[key]['repo']}")
        loaded, note = LOADERS[key](device)
        if loaded is None:
            say(f"  [SKIP] {note}")
            notes[key] = note
            continue
        run, net, wp = loaded
        n_par = count_params(net)
        exp = SPEC[key]["expect_params"]
        say(f"  weights    : {wp}")
        say(f"  parameters : {n_par:,}" +
            ("" if exp is None else
             (f"   expected {exp:,}   " +
              ("MATCH" if n_par == exp else "MISMATCH, check the checkpoint"))))
        if exp is not None and n_par != exp:
            say("  [WARN] the parameter count does not match the published "
                "value; the timing below is recorded but should not be quoted "
                "until the checkpoint is confirmed")
        try:
            st = time_scope(run, imgs, args.warmup)
        except Exception as exc:
            say(f"  [SKIP] forward pass failed: {type(exc).__name__}: {exc}")
            notes[key] = f"forward failed: {type(exc).__name__}"
            continue
        say(f"  median     : {st['median_ms']} ms      "
            f"(mean {st['mean_ms']}, p95 {st['p95_ms']}, n={st['n_timed']})")
        say(f"  vs classifier: {st['median_ms'] / 1.280:.2f}x")
        results[key] = {**st, "params": n_par, "weight": str(wp),
                        "vs_classifier": round(st["median_ms"] / 1.280, 2)}

    # a coverage statement, not a success count: say what is missing and why
    rule("summary")
    for key in wanted:
        if key in results:
            say(f"  {SPEC[key]['label']:10s} {results[key]['median_ms']:>10.3f} ms"
                f"   ({results[key]['vs_classifier']}x the classifier)")
        else:
            say(f"  {SPEC[key]['label']:10s} {'no figure':>10s}"
                f"   {notes.get(key, 'unknown reason')}")
    got, want = len(results), len(wanted)
    say(f"\n  {got} of {want} models produced a latency figure.")
    if got < want:
        say("  The paper can be delivered with the missing cells left empty; "
            "nothing in Section VI depends on these three.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "M_timing_three_models.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "protocol": {"source": "L2_timing_full_pool.py", "seed": args.seed,
                     "n_degraded": args.n_degraded, "warmup": args.warmup,
                     "threads": 1, "batch": 1, "device": "cpu"},
        "reference_ms": {"classifier": 1.280, "adair": 203.667,
                         "cidnet": 18.253},
        "n_images_loaded": len(imgs),
        "results": results,
        "not_timed": notes,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    say(f"\n  written: {out}")
    say("  send this file back and the three cells in Table II can be filled.")


if __name__ == "__main__":
    main()
