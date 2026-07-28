# -*- coding: utf-8 -*-
"""
RERUN_dcp.py
Re-runs, in the right order, every script whose output was computed with the
DCP variant that skipped its refinement step. Q_dcp_branch has already been
rerun with the guided filter built in; this driver takes it from there.

WHY A DRIVER AND NOT A LIST OF COMMANDS
  The scripts have a dependency order that is not obvious and easy to get wrong:

      Q_dcp_branch  -> dcp_cure.csv                         (DONE, the new cache)
        |-> Z_injection_definitive   -> Z_injection_per_image.csv
        |     |-> ZA_deep_vs_injection -> ZA_deep_vs_injection.json
        |-> T_noise_robustness       -> T_noise_per_image.csv
        |     |-> U_noise_selector     -> U_noise_selector.csv/json
        |     |-> V_robust_routing     -> V_robust_routing.csv/json
        |-> M_multiseed_cascade      -> M_multiseed_cascade.json
      K_merge_results               -> merged rebuild + nine-method table
      L2_timing_full_pool           -> L2 timing with the new DCP cost

  U and V do NOT read dcp_cure.csv; they read T_noise_per_image.csv, so T has to
  finish before they start. Running them out of order silently uses stale noise
  predictions. This driver enforces the order.

WHAT IT GUARANTEES
  1. It refuses to start unless dcp_cure.csv on disk is the REFINED one. It reads
     Q_dcp.run_config.json and checks guided_filter is present and HAS_GUIDED-era
     wording is gone. If the old cache is still there, it stops and says so, so we
     never rebuild figures on the broken numbers again.
  2. Each step is skipped if its output already exists AND is newer than
     dcp_cure.csv, so the driver is resumable: rerun it after an interruption and
     it continues where it stopped. Use --force to redo everything.
  3. After each step it checks the expected output file was written and is
     non-empty. A step that fails stops the driver; nothing downstream runs on a
     missing input.
  4. It prints a one-line result per step and a final summary, and writes
     RERUN_dcp.log with the full transcript.

WHAT IT DELIBERATELY DOES NOT DO
  It does not touch merged_per_image.csv's deep-model columns (AdaIR, CIDNet,
  the rule operators). Those never used DCP and stay valid; K_merge_results is
  invoked with --skip-rescan so it re-joins rather than re-runs the deep models.
  If you WANT a full rescan, pass --full-merge.

RUN
  conda activate pcm_sim
  cd D:\\Project\\traffic_sign\\src
  python RERUN_dcp.py                 # do the whole chain, skipping done steps
  python RERUN_dcp.py --dry-run       # print the plan, run nothing
  python RERUN_dcp.py --force         # redo every step from scratch
  python RERUN_dcp.py --only T,U,V    # run just these, respecting dependencies
  python RERUN_dcp.py --full-merge    # let K_merge_results rescan deep models too

  Long steps: T_noise_robustness is the slow one (~65 min because it runs AdaIR
  under noise). Everything else is minutes. Total, first clean run, roughly 80
  minutes; a resumed run only does what is left.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC = PROJECT_ROOT / "src"
OUT = PROJECT_ROOT / "outputs_revision"
DCP_CACHE = OUT / "dcp_cure.csv"
RUN_CFG = OUT / "Q_dcp.run_config.json"

LOG = []


def log(msg=""):
    line = f"[{datetime.now():%H:%M:%S}] {msg}" if msg else ""
    print(line)
    LOG.append(line)


def flush_log():
    (SRC / "RERUN_dcp.log").write_text("\n".join(LOG), encoding="utf-8")


# key -> (script, [extra args], expected_output, note)
STEPS = {
    "Z": ("Z_injection_definitive.py", ["--challenges", "all"],
          OUT / "Z_injection_per_image.csv",
          "operator-injection sweep, ALL 12 challenges (Fig.3 needs 12, not the "
          "4-challenge default). --challenges all is REQUIRED here."),
    "ZA": ("ZA_deep_vs_injection.py", [],
           OUT / "ZA_deep_vs_injection.json",
           "deep-vs-injection comparison; reads Z + merged + dcp"),
    # T has its own overwrite guard: it refuses to run if T_noise_per_image.csv
    # already exists unless --fresh or --resume is passed. The driver supplies the
    # right one at call time (see run_step): --fresh on a clean run so the stale
    # file is rebuilt, --resume if a partial run is being continued. Without this,
    # T aborts immediately and the whole chain stops, which is exactly what
    # happened on the first attempt.
    "T": ("T_noise_robustness.py", ["--fresh"],
          OUT / "T_noise_per_image.csv",
          "per-operator behaviour under sensor noise; runs AdaIR, ~65 min"),
    "U": ("U_noise_selector.py", [],
          OUT / "U_noise_selector.csv",
          "noise-injection selector; reads T_noise_per_image.csv"),
    "V": ("V_robust_routing.py", [],
          OUT / "V_robust_routing.csv",
          "robust-routing variant; reads T_noise_per_image.csv"),
    "M": ("M_multiseed_cascade.py", [],
          OUT / "M_multiseed_cascade.json",
          "multi-seed cascade on the DCP-augmented pool"),
    "K": ("K_merge_results.py", ["--skip-rescan"],
          OUT / "merged_per_image.csv",
          "re-join + nine-method table (deep models NOT re-run)"),
    "L2": ("L2_timing_full_pool.py", [],
           OUT / "L2_timing_full_pool.results.json",
           "unified-clock timing with the new DCP cost"),
}
# dependency order (topological). U and V require T; ZA requires Z.
ORDER = ["Z", "ZA", "T", "U", "V", "M", "K", "L2"]
DEPENDS = {"ZA": ["Z"], "U": ["T"], "V": ["T"]}


def check_cache_is_refined():
    """Refuse to proceed on the old unrefined cache."""
    if not DCP_CACHE.exists():
        return False, f"{DCP_CACHE} does not exist. Run Q_dcp_branch.py --fresh first."
    if not RUN_CFG.exists():
        return False, (f"{RUN_CFG} not found, so the cache provenance cannot be "
                       f"confirmed. Re-run Q_dcp_branch.py --fresh to regenerate "
                       f"both.")
    try:
        cfg = json.loads(RUN_CFG.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"could not parse {RUN_CFG}: {e}"
    gf = cfg.get("guided_filter", "")
    if "boxFilter" not in gf and "guided" not in gf.lower():
        return False, (f"Q_dcp.run_config.json says guided_filter = {gf!r}. This "
                       f"looks like the OLD cache without the refinement. Re-run "
                       f"Q_dcp_branch.py --fresh with the rewritten script before "
                       f"rebuilding anything.")
    if cfg.get("guided_filter_used") is False:
        return False, ("Q_dcp.run_config.json has guided_filter_used=false, the "
                       "broken cache. Re-run Q_dcp_branch.py --fresh.")
    n = cfg.get("images_this_run", 0)
    return True, (f"cache is the refined one: guided_filter = {gf!r}, "
                  f"{n} images, opencv {cfg.get('opencv','?')}")


def needs_run(key, force):
    _, _, out, _ = STEPS[key]
    if force:
        return True, "forced"
    if not out.exists():
        return True, "output missing"
    if out.stat().st_size == 0:
        return True, "output is empty"
    if out.stat().st_mtime < DCP_CACHE.stat().st_mtime:
        return True, "output older than the new dcp_cure.csv"
    return False, "up to date"


def run_step(key, dry):
    script, extra, out, note = STEPS[key]
    path = SRC / script
    if not path.exists():
        log(f"  ✗ {script} not found in {SRC}. Cannot continue.")
        return False
    # Steps that carry --fresh (only T) always rebuild from scratch. We do NOT
    # auto-switch to --resume: a T_noise_per_image.csv left over from the OLD
    # broken cache looks the same on disk as a half-finished new one, and
    # resuming would keep the stale rows. A clean 65-minute rebuild is cheap
    # next to silently mixing old and new predictions. If T itself is
    # interrupted, the operator can rerun with --only T after adding --resume
    # by hand, having confirmed the partial file is from THIS cache.
    cmd = [sys.executable, str(path), *extra]
    log(f"  {key}: {script} {' '.join(extra)}")
    log(f"     {note}")
    if dry:
        log(f"     (dry-run, not executed)  expected -> {out.name}")
        return True
    size_before = out.stat().st_size if out.exists() else -1
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(SRC))
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        log(f"     ✗ exited with code {r.returncode} after {dt:.1f} min. "
            f"Stopping so nothing downstream runs on a missing input.")
        return False
    if not out.exists() or out.stat().st_size == 0:
        log(f"     ✗ finished but {out.name} is missing or empty. Stopping.")
        return False
    grew = "" if size_before < 0 else f" ({size_before} -> {out.stat().st_size} bytes)"
    log(f"     ✓ done in {dt:.1f} min -> {out.name}{grew}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and run nothing")
    ap.add_argument("--force", action="store_true",
                    help="redo every step even if its output looks current")
    ap.add_argument("--only", default="",
                    help="comma-separated keys to run, e.g. T,U,V "
                         "(dependencies are pulled in automatically)")
    ap.add_argument("--full-merge", action="store_true",
                    help="let K_merge_results rescan the deep models too "
                         "(default re-joins with --skip-rescan)")
    args = ap.parse_args()

    log("=" * 68)
    log("RERUN_dcp.py  -  ordered rebuild after the DCP refinement fix")
    log("=" * 68)

    ok, msg = check_cache_is_refined()
    log(f"cache check: {msg}")
    if not ok:
        log("ABORT.")
        flush_log()
        sys.exit(1)

    if args.full_merge:
        STEPS["K"] = ("K_merge_results.py", [], OUT / "merged_per_image.csv",
                      "FULL rebuild: rescan deep models + merge + nine-method table")
        log("note: --full-merge set, K_merge_results will rescan the deep models "
            "(much slower).")

    # decide which steps to run
    if args.only:
        want = [k.strip().upper() for k in args.only.split(",") if k.strip()]
        for k in want:
            if k not in STEPS:
                log(f"unknown step {k!r}. Valid: {', '.join(ORDER)}")
                flush_log()
                sys.exit(1)
        # pull in dependencies
        need = set(want)
        changed = True
        while changed:
            changed = False
            for k in list(need):
                for dep in DEPENDS.get(k, []):
                    if dep not in need:
                        need.add(dep)
                        changed = True
        plan = [k for k in ORDER if k in need]
        log(f"running only {want} (with dependencies: {plan})")
    else:
        plan = list(ORDER)

    log("")
    log("PLAN:")
    for k in plan:
        run, why = needs_run(k, args.force)
        mark = "RUN " if run else "skip"
        log(f"  [{mark}] {k:3s} {STEPS[k][0]:28s} ({why})")
    log("")

    if args.dry_run:
        log("dry-run: nothing executed.")
        flush_log()
        return

    done, skipped, failed = [], [], None
    for k in plan:
        run, why = needs_run(k, args.force)
        if not run:
            log(f"  {k}: skip ({why})")
            skipped.append(k)
            continue
        if not run_step(k, dry=False):
            failed = k
            break
        done.append(k)

    log("")
    log("=" * 68)
    log("SUMMARY")
    log("=" * 68)
    log(f"  ran:     {', '.join(done) or '(none)'}")
    log(f"  skipped: {', '.join(skipped) or '(none)'}")
    if failed:
        log(f"  FAILED at: {failed}. Fix the error above, then rerun "
            f"RERUN_dcp.py; it resumes from here.")
    else:
        log("  all steps complete. The six DCP-dependent artifacts and the "
            "nine-method merge and timing are now on the refined cache.")
        log("  next: rebuild the figures (they read these files), then the paper.")
    flush_log()
    log(f"  full transcript -> {SRC / 'RERUN_dcp.log'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
