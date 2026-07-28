r"""
make_qualitative_figures.py  (v2)
=================================
为 JEI VA-Adaptive 论文生成定性图，全部使用 REAL CURE-TSR 真实退化图像。

v2 相对 v1 的改动:
  1. 分支感知选图: showcase 每行可指定 want=分支名(gamma/CLAHE/stretch/pass-through)，
     脚本在该 (challenge,severity) 桶里挑一张真正路由到该分支的图，从而让四个分支
     在图里都展示到，而不是清一色 pass-through。
  2. prefer_sign: 优先选某一类标志(默认 STOP)，让各行主体一致、视觉连贯;找不到则退而求其次。
  3. 每个桶打印路由分布 tally，方便你看清该退化下 VA 实际怎么路由。
  4. Figure 2(honest)只保留 VA 确实不占优的案例: Rain(误路由变差) + Noise(无专用分支)。
     不再把"VA 大胜"的 Darkening 误放进 honest 图。

设计不变的要点:
  - 复用 evaluate_cure_tsr_external.py 里已审计的 fn_baseline / fn_fixed_clahe /
    fn_fixed_gamma / fn_fixed_stretch / fn_va_adaptive，保证图里的增强和正式实验逐像素一致。
  - VA 分支用"VA 输出与哪个固定分支输出逐像素相等"反推，不依赖阈值常量或 route_decision。
  - 纯 CPU，本地 conda pcm_sim 直接跑;不需要 GPU、分类器、PromptIR/FFA。

注意(写图注时别越界): 本图展示的是"增强后的图像"，说明的是路由机制与固定法的失败模式，
  不是识别准确率(那是定量表和 bootstrap CI 的事)。图注请写 "routing mechanism / failure
  modes of fixed methods"，不要写 "improves recognition"。

运行(本地 Windows):
  conda activate pcm_sim
  python src\make_qualitative_figures.py

产物(写到 PROJECT_ROOT/outputs_qualitative/):
  fig_routing_showcase.png / .svg   VA 把每张真实退化图路由到合理分支(四分支各展示一次)
  fig_honest_cases.png   / .svg     VA 不占优的诚实案例(Rain 误路由 / Noise 无分支)
  run_config.json                   本次配置(可复现)
  execution_log.txt                 逐条记录: 桶的路由分布、选了哪张图、VA 触发了哪个分支
"""

import os
import sys
import json
import time
import datetime

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_cure_tsr_external as E   # noqa: E402

# ==================================================================
# 配置区
# ==================================================================

CURE_DIR = E.CURE_TSR_DIR
_PROJECT_ROOT = getattr(E, "PROJECT_ROOT", CURE_DIR.parent.parent)
OUT_DIR = _PROJECT_ROOT / "outputs_qualitative"

# 颜色假设开关: cv2.imread 给 BGR；若 fn_* 输入输出是 BGR(最常见)，保持 True，显示转 RGB。
# 出图后若红标志显示成蓝，改 False。(你之前那次 True 是对的。)
FN_OUTPUT_IS_BGR = True

# 优先展示哪一类标志(让各行主体一致)。CURE sign 6 = STOP(-> GTSRB 14)。找不到则用任意映射类。
PREFER_SIGN = 6

# 为找特定分支，每个桶最多扫多少张(排序后取前 N，决定性可复现)。越大越全但越慢。
SCAN_CAP = 800

# 挑战 id: 0=ChallengeFree, 4=Darkening, 8=Noise, 9=Rain, 11=Snow, 12=Haze
# severity: 挑战图 1..5；ChallengeFree 用 0。
# want: 想展示的分支名(gamma / CLAHE / stretch / pass-through)；不写则展示该图实际分支。

SHOWCASE = [   # 图1: 尽量让四个分支各出现一次
    dict(ch=4,  sev=5, want="gamma"),         # Darkening 重度 -> gamma 救暗图(且固定 stretch 会变全黑)
    dict(ch=12, sev=3, want="CLAHE"),         # Haze 中度     -> CLAHE 提局部对比
    dict(ch=9,  sev=2, want="stretch"),       # Rain 轻度     -> stretch(较稀有, 找不到会告警回退)
    dict(ch=0,  sev=0, want="pass-through"),  # ChallengeFree -> pass-through(干净图不动)
]

HONEST = [     # 图2: 只放 VA 确实不占优的
    dict(ch=9, sev=5),   # Rain 重度:  聚合 VA 20.7 < baseline 25.4，误路由把雨纹放大
    dict(ch=8, sev=5),   # Noise 重度: 无专用噪声分支，VA 走 pass-through，无法处理传感器噪声
]

METHODS = [
    ("Degraded input", E.fn_baseline),     # baseline = 不增强 = 退化输入本身
    ("Fixed CLAHE",    E.fn_fixed_clahe),
    ("Fixed gamma",    E.fn_fixed_gamma),
    ("Fixed stretch",  E.fn_fixed_stretch),
    ("VA-Adaptive",    E.fn_va_adaptive),
]

# 合法分支名(用于校验 want 拼写；写错会导致该行一直回退)
KNOWN_BRANCHES = {"gamma", "CLAHE", "stretch", "pass-through"}

DPI = 300

# ==================================================================
# 工具函数
# ==================================================================

def log(msg, fh=None):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if fh is not None:
        fh.write(line + "\n")


def build_index(cure_dir, fh):
    """扫 .bmp，按 (challenge, severity) 索引"真实(seq=01)+映射类"的 (sign, path)。
    CURE-TSR 文件名: seq_sign_challenge_level_id.bmp -> group(1..4)=seq,sign,challenge,level。
    """
    idx = {}
    n_seen = n_kept = 0
    for p in cure_dir.rglob("*.bmp"):
        m = E.FILENAME_PATTERN.search(p.name)
        if not m:
            continue
        n_seen += 1
        seq, sign, ch, sev = (int(m.group(1)), int(m.group(2)),
                              int(m.group(3)), int(m.group(4)))
        if seq != 1:
            continue
        if sign not in E.DEFAULT_CURE_TO_GTSRB:
            continue
        idx.setdefault((ch, sev), []).append((sign, p))
        n_kept += 1
    for k in idx:
        idx[k].sort(key=lambda sp: sp[1].name)   # 桶内按文件名排序，保证决定性
    log(f"index: scanned {n_seen} bmp, kept {n_kept} real+mapped, "
        f"{len(idx)} (challenge,severity) buckets", fh)
    if n_kept == 0:
        raise FileNotFoundError(
            f"在 {cure_dir} 下没扫到真实+映射的 .bmp。检查 CURE_DIR 是否指向 CURE-TSR 真实子集。")
    return idx


def detect_va_branch(bgr):
    """VA 输出与哪个固定分支输出逐像素相等就是哪个分支。不依赖阈值/route_decision。"""
    va = E.fn_va_adaptive(bgr.copy())
    for name, fn in (("pass-through", E.fn_baseline), ("gamma", E.fn_fixed_gamma),
                     ("CLAHE", E.fn_fixed_clahe), ("stretch", E.fn_fixed_stretch)):
        if np.array_equal(va, fn(bgr.copy())):
            return name
    return "unknown"


def find_in_bucket(idx, ch, sev, want=None, prefer_sign=PREFER_SIGN, cap=SCAN_CAP, fh=None):
    """在 (ch,sev) 桶里挑一张图。返回 (path, branch, tally)。
    want=None  : 不限分支，优先 prefer_sign，否则第一张。
    want=分支名: 扫描(<=cap)找路由到该分支的图，优先 prefer_sign；同时统计 tally。
                找不到 -> 告警并回退到 prefer_sign / 第一张。
    """
    items = idx.get((ch, sev), [])
    if not items:
        return None, None, {}

    if want is None:
        path = next((p for s, p in items if s == prefer_sign), items[0][1])
        return path, detect_va_branch(cv2.imread(str(path))), {}

    tally, first_any, first_pref = {}, None, None
    for s, p in items[:cap]:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        br = detect_va_branch(bgr)
        tally[br] = tally.get(br, 0) + 1
        if br == want:
            if first_any is None:
                first_any = p
            if s == prefer_sign and first_pref is None:
                first_pref = p
    chosen = first_pref or first_any
    if chosen is None:
        chosen = next((p for s, p in items if s == prefer_sign), items[0][1])
        if fh:
            log(f"  [warn] (ch={ch},sev={sev}) 扫{min(len(items), cap)}张未见 '{want}' 分支; "
                f"tally={tally}; 回退 {chosen.name}", fh)
        return chosen, detect_va_branch(cv2.imread(str(chosen))), tally
    return chosen, want, tally


def to_rgb_for_display(out):
    """统一成可显示 RGB uint8，兼容 uint8 / float[0,1] / float[0,255]。"""
    a = np.asarray(out)
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if FN_OUTPUT_IS_BGR else a
    return a


def render_figure(conditions, title, out_stem, idx, fh):
    resolved = []
    for cond in conditions:
        path, branch, tally = find_in_bucket(
            idx, cond["ch"], cond["sev"], want=cond.get("want"), fh=fh)
        if path is None:
            log(f"  [skip] 没有 challenge={cond['ch']} severity={cond['sev']} 的真实+映射图", fh)
            continue
        ch_name = E.CHALLENGE_TYPES.get(cond["ch"], str(cond["ch"])) \
            if hasattr(E, "CHALLENGE_TYPES") else str(cond["ch"])
        want_note = f" want={cond['want']}" if cond.get("want") else ""
        tally_note = f" | bucket tally={tally}" if tally else ""
        log(f"  {ch_name} sev{cond['sev']}{want_note} | {path.name} | VA->{branch}{tally_note}", fh)
        resolved.append((cond, path, branch, ch_name))

    if not resolved:
        log(f"  [warn] {title}: 没有任何条件命中，跳过此图", fh)
        return None

    nrows, ncols = len(resolved), len(METHODS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.7, nrows * 1.85),
                             squeeze=False)

    for i, (cond, path, branch, ch_name) in enumerate(resolved):
        bgr = cv2.imread(str(path))
        row_label = ch_name if cond["ch"] == 0 else f"{ch_name}\nseverity {cond['sev']}"
        for j, (mname, fn) in enumerate(METHODS):
            rgb = to_rgb_for_display(fn(bgr.copy()))      # .copy(): 防 fn 原地改输入
            ax = axes[i][j]
            ax.imshow(rgb, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.6"); sp.set_linewidth(0.6)
            if i == 0:
                ax.set_title(mname, fontsize=9)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=28)
            if mname == "VA-Adaptive":
                ax.set_xlabel(f"-> {branch}", fontsize=8, color="#c0392b")

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = OUT_DIR / f"{out_stem}.png"
    svg = OUT_DIR / f"{out_stem}.svg"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {png.name} / {svg.name}  ({nrows} rows)", fh)
    return str(png)


# ==================================================================
# 主流程
# ==================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(OUT_DIR / "execution_log.txt", "w", encoding="utf-8")
    t0 = time.time()
    log(f"CURE_DIR = {CURE_DIR}", logf)
    log(f"OUT_DIR  = {OUT_DIR}", logf)
    log(f"FN_OUTPUT_IS_BGR={FN_OUTPUT_IS_BGR}  PREFER_SIGN={PREFER_SIGN}  SCAN_CAP={SCAN_CAP}", logf)

    idx = build_index(CURE_DIR, logf)

    # 校验 want 拼写: 写错(如小写 'clahe')会导致该行每次回退而非报错
    for c in SHOWCASE + HONEST:
        w = c.get("want")
        if w is not None and w not in KNOWN_BRANCHES:
            log(f"[warn] want='{w}' 不是合法分支名 {sorted(KNOWN_BRANCHES)}; 该行将一直回退", logf)

    produced = []
    log("=== Figure 1: routing showcase ===", logf)
    f1 = render_figure(SHOWCASE,
                       "VA-Adaptive routing on real CURE-TSR degradations",
                       "fig_routing_showcase", idx, logf)
    if f1:
        produced.append(f1)
    log("=== Figure 2: honest (does-not-improve) cases ===", logf)
    f2 = render_figure(HONEST,
                       "Cases where VA-Adaptive does not improve over baseline (honest disclosure)",
                       "fig_honest_cases", idx, logf)
    if f2:
        produced.append(f2)

    cfg = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "cure_dir": str(CURE_DIR),
        "out_dir": str(OUT_DIR),
        "fn_output_is_bgr": FN_OUTPUT_IS_BGR,
        "prefer_sign": PREFER_SIGN,
        "scan_cap": SCAN_CAP,
        "methods": [m for m, _ in METHODS],
        "showcase_conditions": SHOWCASE,
        "honest_conditions": HONEST,
        "branch_detection": "pixel-equality vs fixed branches (no threshold/route_decision dependency)",
        "note": "figures show enhanced images (routing mechanism / fixed-method failures), NOT recognition accuracy",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    log(f"done in {time.time() - t0:.1f}s; produced {len(produced)} figures", logf)
    logf.close()


if __name__ == "__main__":
    main()
