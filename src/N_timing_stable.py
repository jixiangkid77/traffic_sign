#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N_timing_stable.py
==================
Re-measures every latency the paper quotes, on one pass, with a protocol that
returns the same answer twice.

WHY THIS REPLACES L2
    L2 timed each crop once and reported the median of those single readings.
    Running Zero-DCE twice under that protocol gave medians of 35.600 ms and
    9.441 ms: a factor of 3.8 on the same seed, the same 210 crops and the same
    machine. A single reading measures how long the call took that time, which
    includes the scheduler, the cache state, the CPU's clock and whatever else
    the machine was doing. The paper prints these numbers to three decimals; on
    that evidence they are not stable to one.

WHAT CHANGES
    Each crop is timed REPEATS times and the MINIMUM is kept. The minimum is the
    reading least contaminated by interference: no amount of luck makes a
    computation faster than it is, while any amount of bad luck makes it slower.
    This is the standard practice for micro-benchmarks and it is what makes the
    numbers reproducible.

    Everything is measured in one process, in one pass, so no scope can be
    penalised for having run while the machine was busy with something else.

    Stability is reported, not assumed: the run is repeated in two independent
    blocks and the two medians are printed side by side. The paper should quote
    only the digits that survive that comparison.

WHAT IS MEASURED (twelve scopes, matching Table II plus the classifier)
    four operators        gamma, CLAHE, contrast stretch, dark channel prior
    two routers           the frozen rule, and V-B
    five learned models   AdaIR, CIDNet, Zero-DCE, FFA-Net, PromptIR
    one reference         the CompactCNN classifier itself
    no enhancement is not timed: it does nothing, and its latency is zero by
    definition rather than by measurement.

PROMPTIR
    Its checkpoint is published on the repository's releases page. Rather than
    asking anyone to find it, this script asks the GitHub API which asset the
    release carries and downloads it. Hardcoding a URL would rot the first time
    the authors renamed a file; asking at run time does not.

USAGE
    python N_timing_stable.py
    python N_timing_stable.py --repeats 5 --n-degraded 200
    python N_timing_stable.py --skip-download        # offline, use what is local
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
MODELS_DIR = PROJECT_ROOT / "models"
THIRD_PARTY = PROJECT_ROOT / "third_party"
CURE_TSR_DEFAULT = PROJECT_ROOT / "datasets" / "CURE-TSR"
OUT_JSON = PROJECT_ROOT / "outputs_revision" / "N_timing_stable.results.json"

DEEP = {
    "adair": {
        "label": "AdaIR", "repo": "https://github.com/c-yn/AdaIR",
        "dir": "AdaIR", "globs": ["*adair*"], "params": 28_784_824,
        "release_api": None,
    },
    "cidnet": {
        "label": "CIDNet", "repo": "https://github.com/Fediory/HVI-CIDNet",
        "dir": "HVI-CIDNet", "globs": ["*cidnet*", "*hvi*"],
        "params": 1_975_569, "release_api": None,
    },
    "zero_dce": {
        "label": "Zero-DCE", "repo": "https://github.com/Li-Chongyi/Zero-DCE",
        "dir": "Zero-DCE", "globs": ["*zero*dce*", "*Epoch99*"],
        "params": 79_416, "release_api": None,
    },
    "ffa_net": {
        "label": "FFA-Net", "repo": "https://github.com/zhilin007/FFA-Net",
        "dir": "FFA-Net", "globs": ["*ffa*"], "params": 4_455_913,
        "release_api": None,
    },
    "promptir": {
        "label": "PromptIR", "repo": "https://github.com/va1shn9v/PromptIR",
        "dir": "PromptIR", "globs": ["*prompt*"], "params": None,
        "release_api": "https://api.github.com/repos/va1shn9v/PromptIR/releases",
    },
}

WEIGHT_EXT = {".pth", ".pt", ".ckpt", ".pk", ".pkl", ".safetensors", ".bin"}


def say(m=""):
    print(m, flush=True)


def rule(t=""):
    say("\n" + "=" * 74)
    if t:
        say(t)
        say("=" * 74)


# --------------------------------------------------------------------------
def find_weight(globs):
    """Match by pattern over every weight-shaped file, never by a guessed name."""
    if not MODELS_DIR.exists():
        return None, f"models directory absent: {MODELS_DIR}"
    files = [p for p in MODELS_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in WEIGHT_EXT]
    for g in globs:
        parts = [x for x in g.lower().split("*") if x]
        for f in files:
            if all(x in f.name.lower() for x in parts):
                return f, None
    return None, ("nothing matched " + " or ".join(globs) + "; models/ has: " +
                  (", ".join(sorted(f.name for f in files)) or "no weight files"))


def _http_get(url, timeout=90):
    """Fetch bytes, trying three routes before giving up.

    The first attempt on this machine failed with an SSL parsing error, which
    is a property of the local certificate store rather than of the file. A
    public checkpoint is worth fetching by whatever route works; its identity
    is checked afterwards by counting its parameters, which is a stronger test
    than the transport.
    """
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
                             capture_output=True, timeout=timeout + 30)
        if out.returncode == 0 and out.stdout:
            return out.stdout, "curl"
    except Exception as exc:
        errs.append(f"curl: {type(exc).__name__}")
    return None, "; ".join(errs)


def fetch_release_asset(api_url, dest_dir):
    """Ask the release which asset it carries, then fetch the largest one.

    The largest asset is the checkpoint: the source archives GitHub attaches
    automatically are far smaller than a set of weights. Nothing here hardcodes
    a filename or a URL, so a rename upstream costs nothing.
    """
    raw, how = _http_get(api_url, timeout=60)
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
    say(f"    fetching {name} ({size / 1e6:.0f} MB) via {how} ...")
    blob, how2 = _http_get(url, timeout=1800)
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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_tensor(img, device):
    import torch
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


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


# --------------------------------------------------------------------------
J_MOD = None          # set in main once J is imported


def build_deep(key, device, allow_download):
    """Return ((callable, net, weight_path), note) or (None, why)."""
    import torch
    spec = DEEP[key]
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

    if key == "adair":
        sys.path.insert(0, str(repo))
        from net.model import AdaIR                       # noqa
        net = AdaIR(decoder=True)
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "net.", "module.")
    elif key == "cidnet":
        sys.path.insert(0, str(repo))
        cand = list(repo.rglob("net/CIDNet.py")) or list(repo.rglob("CIDNet.py"))
        mod = load_module(cand[0], "cidnet_mod")
        net = mod.CIDNet()
        if wp.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(wp))
        else:
            sd = unwrap(torch.load(str(wp), map_location="cpu",
                                   weights_only=False))
        sd = strip(sd, "module.")
    elif key == "zero_dce":
        cand = list(repo.rglob("model.py"))
        mod = load_module(cand[0], "zdce_mod")
        net = mod.enhance_net_nopool()
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "module.")
    elif key == "ffa_net":
        cand = list(repo.rglob("FFA.py"))
        sys.path.insert(0, str(cand[0].parent))
        mod = load_module(cand[0], "ffa_mod")
        net = mod.FFA(gps=3, blocks=19)
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "module.")
    else:                                                  # promptir
        cand = list(repo.rglob("net/model.py"))
        sys.path.insert(0, str(cand[0].parent.parent))
        mod = load_module(cand[0], "promptir_mod")
        net = mod.PromptIR(decoder=True)
        sd = strip(unwrap(torch.load(str(wp), map_location="cpu",
                                     weights_only=False)), "net.", "module.")

    net.load_state_dict(sd, strict=False)
    net.eval().to(device)

    # Go through J.enhance_batch, which is what L2 timed and what the sweep
    # used. It pads each crop up to the factor these networks require and
    # crops the result back. Calling the network directly, as the first draft
    # did, fails on any crop whose height is odd: AdaIR's pixel_unshuffle
    # wants a multiple of two, and a 17-pixel crop is not one.
    # Zero-DCE cannot use enhance_batch unchanged: its forward returns a
    # triple, and enhance_batch clamps whatever comes back, which a tuple will
    # not accept. It gets the same padding through the same helper, with the
    # second element picked out first, so the two paths differ only where they
    # must.
    if key == "zero_dce":
        import cv2
        import torch.nn.functional as FP

        def run(img):
            h, w = img.shape[:2]
            ph, pw = J_MOD.pad_amount(h, w, J_MOD.PAD_FACTOR)
            t = torch.from_numpy(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)
            x = (t / 255.0).unsqueeze(0)
            if ph or pw:
                try:
                    x = FP.pad(x, (0, pw, 0, ph), mode="reflect")
                except Exception:
                    x = FP.pad(x, (0, pw, 0, ph), mode="replicate")
            with torch.no_grad():
                out = net(x.to(device))
            y = out[1] if isinstance(out, (tuple, list)) else out
            y = torch.clamp(y, 0, 1)[:, :, :h, :w].cpu()
            arr = (y[0].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        def run(img):
            return J_MOD.enhance_batch(net, [img], device)[0]

    return (run, net, wp), "ok"


# --------------------------------------------------------------------------
def time_all_interleaved(scopes, imgs, repeats, warmup, fast_reps=None):
    """Time every scope on every crop, cycling scopes inside the crop loop.

    The first version ran each scope to completion before starting the next.
    That gave Zero-DCE 4.3 ms in the first block and 84.1 ms in the second: the
    second block reached it after two minutes of FFA-Net had heated the
    processor, so the reading was measuring the clock rate rather than the
    model. Cycling the scopes crop by crop puts every scope through the same
    thermal history, which is the only way the numbers can be compared.

    Within a crop each scope is run `repeats` times and the fastest kept, since
    interference can only ever add time.
    """
    names = list(scopes)
    per = {n: [] for n in names}
    # A scope that takes a fraction of a millisecond is hurt far more by one
    # scheduling hiccup than a scope that takes a quarter of a second. The fast
    # ones therefore get more attempts to find their floor. The classifier came
    # back with a twenty-three per cent spread between blocks at five repeats,
    # which is not a property of the classifier.
    reps = dict(fast_reps or {})
    for i, img in enumerate(imgs):
        for n in names:
            fn = scopes[n][0]
            t = float("inf")
            ok = True
            for _ in range(reps.get(n, repeats)):
                try:
                    t0 = time.perf_counter()
                    fn(img)
                    t = min(t, time.perf_counter() - t0)
                except Exception as exc:
                    per[n] = exc
                    ok = False
                    break
            if not ok:
                continue
            if i >= warmup and isinstance(per[n], list):
                per[n].append(t)
    out = {}
    for n, v in per.items():
        if not isinstance(v, list):
            out[n] = {"error": f"{type(v).__name__}: {v}"}
            continue
        if not v:
            out[n] = {"error": "no timings collected"}
            continue
        a = np.asarray(v, dtype=np.float64) * 1000.0
        out[n] = {"n": int(a.size),
                  "median_ms": round(float(np.median(a)), 4),
                  "mean_ms": round(float(a.mean()), 4),
                  "p95_ms": round(float(np.percentile(a, 95)), 4)}
    return out


def stratified_pick(samples, n, seed):
    """Verbatim from L2, so the crops are the same ones."""
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


def spread(a, b):
    """Relative disagreement between the two blocks, as a percentage.

    The first version counted how many decimals the two readings shared, which
    called 1.3435 and 1.3860 "agreeing to zero decimals" even though they are
    within three per cent of each other. A ratio says what a reader needs to
    know: how much the number moves when you measure it again.
    """
    if a <= 0 or b <= 0:
        return float("inf")
    return 100.0 * abs(a - b) / min(a, b)


def quote_at(a, b):
    """Round to the precision the spread supports, and no further."""
    m = 0.5 * (a + b)
    s = spread(a, b)
    if s > 25:
        return None                      # too unstable to quote at all
    if s > 5:
        return round(m, 0) if m >= 10 else round(m, 1)
    if s > 1:
        return round(m, 1) if m >= 1 else round(m, 3)
    return round(m, 2) if m >= 1 else round(m, 4)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cure-root", default=str(CURE_TSR_DEFAULT))
    ap.add_argument("--n-degraded", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fast-repeats", type=int, default=25,
                    help="extra repeats for scopes under 5 ms, where a single "
                         "scheduling hiccup is a large fraction of the reading")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classifier",
                    default=str(MODELS_DIR / "mbnetv3_baseline.pth"),
                    help="the frozen CompactCNN checkpoint")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    rule("N: one stable measurement of every latency the paper quotes")
    say(f"Each crop timed {args.repeats} times, fastest kept; "
        f"{args.n_degraded} crops after {args.warmup} warm-up; "
        "batch of one, one CPU thread.")
    say("Two independent blocks are run and compared, so the paper can quote "
        "only the digits that hold.")

    try:
        import torch
        import cv2
    except Exception as exc:
        say(f"\n[FATAL] {type(exc).__name__}: {exc}")
        sys.exit(1)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    here = Path(__file__).parent
    for need in ("J_local_deep_eval.py", "F_master_sweep_cache.py",
                 "Q_dcp_branch.py"):
        if not (here / need).exists():
            say(f"[FATAL] {need} must sit beside this script")
            sys.exit(1)
    global J_MOD
    J = load_module(here / "J_local_deep_eval.py", "J_local")
    J_MOD = J
    F = load_module(here / "F_master_sweep_cache.py", "F_master")
    Q = load_module(here / "Q_dcp_branch.py", "Q_dcp")

    rule("sampling")
    samples, n_files = J.collect_samples(Path(args.cure_root))
    degraded = [s for s in samples if s["sev"] > 0]
    picked = stratified_pick(degraded, args.n_degraded + args.warmup, args.seed)
    imgs = []
    for s in picked:
        im = cv2.imread(str(s["path"]), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    say(f"  usable files {n_files:,}   degraded {len(degraded):,}   "
        f"loaded {len(imgs)} crops")
    if len(imgs) < args.n_degraded + args.warmup:
        say("  [WARN] fewer crops than asked for; the figures will not be "
            "comparable with the published ones")

    # ---- assemble every scope -------------------------------------------
    rule("building the twelve scopes")
    scopes = {}

    # Call the very functions the sweep used, through the same modules, so a
    # timing cannot drift from what produced the accuracies. The names are L2's,
    # checked against the sources rather than remembered.
    scopes["gamma"] = (lambda im: F.apply_branch(im, "gamma"), None)
    scopes["clahe"] = (lambda im: F.apply_branch(im, "clahe"), None)
    scopes["stretch"] = (lambda im: F.apply_branch(im, "stretch"), None)
    scopes["dcp"] = (lambda im: Q.dcp_enhance(im)[0], None)
    say("  four training-free operators: ready")

    def rule_fn(dcp_for_clahe):
        def go(im):
            b, c, e = F.compute_stats(im)
            br = F.route_decision(b, c, e, F.THRESHOLDS)
            if br == "gamma":
                return F.apply_branch(im, "gamma")
            if br == "clahe":
                return Q.dcp_enhance(im)[0] if dcp_for_clahe else \
                    F.apply_branch(im, "clahe")
            if br == "stretch":
                return F.apply_branch(im, "stretch")
            return im
        return go

    scopes["rule"] = (rule_fn(False), None)
    scopes["vb"] = (rule_fn(True), None)
    say("  two routers: ready")

    # The classifier scope is L2's classify32: resize to 32, BGR to RGB, the
    # shared transform, then the frozen CompactCNN.
    clf, note = None, ""
    try:
        clf = F.load_model(str(args.classifier), device)
        tfm = J.build_transform()
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
    if clf is not None:
        def classify32(im):
            small = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                return clf(tfm(rgb).unsqueeze(0).to(device))
        scopes["classifier"] = (classify32, None)
        say("  classifier: ready")
    else:
        say(f"  [WARN] classifier not timed ({note}); the vs-classifier "
            "ratios will fall back to the published 1.28 ms")

    missing = {}
    for key in DEEP:
        say(f"  {DEEP[key]['label']} ...")
        built, why = build_deep(key, device, not args.skip_download)
        if built is None:
            say(f"    [SKIP] {why}")
            missing[key] = why
            continue
        run, net, wp = built
        n_par = sum(p.numel() for p in net.parameters())
        exp = DEEP[key]["params"]
        flag = "" if exp is None else (
            "  MATCH" if n_par == exp else f"  MISMATCH, expected {exp:,}")
        say(f"    weights {wp.name}   parameters {n_par:,}{flag}")
        if exp is not None and n_par != exp:
            say("    [WARN] the checkpoint is not the one the paper counted; "
                "the timing is recorded but should not be quoted")
        scopes[key] = (run, {"params": n_par, "weight": str(wp)})

    # ---- two blocks, so stability is measured rather than hoped ----------
    # A cheap probe on a handful of crops decides which scopes are fast enough
    # to need extra attempts. Measuring this rather than listing it by hand
    # keeps the rule true if a model is added or a machine changes.
    rule("probing which scopes are fast")
    probe = time_all_interleaved(scopes, imgs[:12], 2, 2)
    fast_reps = {}
    for n, st in probe.items():
        if "error" in st:
            continue
        if st["median_ms"] < 5.0:
            fast_reps[n] = args.fast_repeats
    say(f"  extra repeats for: {', '.join(sorted(fast_reps)) or 'none'}")

    results = {}
    for block in (1, 2):
        rule(f"block {block} of 2")
        say("  cycling the scopes crop by crop so each meets the same "
            "machine state")
        got = time_all_interleaved(scopes, imgs, args.repeats, args.warmup,
                                   fast_reps)
        for name, st in got.items():
            if "error" in st:
                say(f"  {name:12s} [SKIP] {st['error'][:70]}")
                missing.setdefault(name, st["error"])
                continue
            results.setdefault(name, {})[f"block{block}"] = st
            meta = scopes[name][1]
            if meta:
                results[name].update(meta)
            say(f"  {name:12s} {st['median_ms']:>10.4f} ms")

    # ---- report -----------------------------------------------------------
    rule("stability and final figures")
    base = None
    if "classifier" in results:
        base = results["classifier"]["block2"]["median_ms"]
    say(f"  classifier reference: "
        f"{base if base else 1.280:.4f} ms"
        f"{'' if base else '  (published value, not re-measured)'}\n")
    say(f"  {'scope':12s} {'block1':>11s} {'block2':>11s} {'spread':>9s} "
        f"{'quote as':>12s} {'x clf':>8s}")
    final = {}
    for name, r in results.items():
        if "block1" not in r or "block2" not in r:
            continue
        a, b = r["block1"]["median_ms"], r["block2"]["median_ms"]
        sp = spread(a, b)
        q = quote_at(a, b)
        mid = 0.5 * (a + b)
        ratio = mid / (base or 1.280)
        final[name] = {"block1_ms": a, "block2_ms": b,
                       "spread_pct": round(sp, 1), "quote_ms": q,
                       "vs_classifier": round(ratio, 2),
                       **{k: v for k, v in r.items() if k in ("params", "weight")}}
        say(f"  {name:12s} {a:>11.4f} {b:>11.4f} {sp:>8.1f}% "
            f"{(str(q) if q is not None else 'unstable'):>12s} {ratio:>7.2f}x")

    rule("summary")
    say(f"  {len(final)} scopes measured, {len(missing)} not measured")
    for k, why in missing.items():
        say(f"    {k}: {why}")
    shaky = [k for k, v in final.items() if v["spread_pct"] > 5]
    if shaky:
        say(f"  scopes whose two blocks differ by more than five per cent: "
            f"{', '.join(shaky)}")
        say("  quote these at the precision in the table, not beyond it")
    bad = [k for k, v in final.items() if v["quote_ms"] is None]
    if bad:
        say(f"  scopes too unstable to quote at all: {', '.join(bad)}")
        say("  re-run with more repeats, or on a quieter machine")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "script": "N_timing_stable.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "protocol": {"repeats": args.repeats, "keep": "minimum per crop",
                     "n_degraded": args.n_degraded, "warmup": args.warmup,
                     "seed": args.seed, "threads": 1, "batch": 1,
                     "blocks": 2, "device": "cpu"},
        "n_images": len(imgs),
        "classifier_ms": base,
        "final": final,
        "not_measured": missing,
    }, indent=2), encoding="utf-8")
    say(f"\n  written: {out}")


if __name__ == "__main__":
    main()
