"""
inspect_cure_tsr.py — CURE-TSR 数据检查 + 类别映射可行性分析（修正版）

【这是 CURE-TSR 工作流的第 1 步，必须先跑这个】

修正项（vs 旧版）：
  ✓ Filename pattern 改为 5 段：sequenceType_signType_challengeType_challengeLevel_Index
    例：01_06_04_03_00012.bmp = real / stop / darkening / level 3 / instance 12
  ✓ Challenge type 用数字编号 (00-12) 与 README 对齐
  ✓ 递归扫描所有子目录，不依赖特定文件夹组织
  ✓ 默认只统计 sequenceType=01 (real data)

输出：
  outputs_cure_tsr/
    inspection_summary.csv        — 类别 × challenge 矩阵（仅 real data）
    challenge_severity_matrix.csv — challenge × severity 矩阵
    mapping_proposal.json         — GTSRB 映射 + 置信度
    sample_thumbnails.png         — 每类抽样，目测验证映射
    decision_recommendation.txt   — 路径 A/B/C 自动建议

运行：
  conda activate pcm_sim
  python src/inspect_cure_tsr.py
"""

import os
import re
import json
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
CURE_TSR_DIR = PROJECT_ROOT / "datasets" / "CURE-TSR"
OUTPUT_DIR = PROJECT_ROOT / "outputs_cure_tsr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CURE-TSR 文件名格式（5 段，对照 README）：
# sequenceType_signType_challengeType_challengeLevel_Index.bmp
FILENAME_PATTERN = re.compile(
    r'(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{5})\.(bmp|png|jpg|jpeg)',
    re.IGNORECASE
)

SEQUENCE_TYPES = {1: "Real", 2: "Unreal"}

CURE_CLASSES = {
    1:  "speed_limit",     2: "goods_vehicles", 3: "no_overtaking",
    4:  "no_stopping",     5: "no_parking",     6: "stop",
    7:  "bicycle",         8: "hump",           9: "no_left",
    10: "no_right",       11: "priority_to",   12: "no_entry",
    13: "yield",          14: "parking",
}

CHALLENGE_TYPES = {
    0:  "ChallengeFree",   1: "Decolorization",  2: "LensBlur",
    3:  "CodecError",      4: "Darkening",       5: "DirtyLens",
    6:  "Exposure",        7: "GaussianBlur",    8: "Noise",
    9:  "Rain",           10: "Shadow",         11: "Snow",
    12: "Haze",
}

# 你 paper 5 种合成退化对应的 CURE 真实退化：
PAPER_RELEVANT_CHALLENGES = [4, 8, 9, 11, 12]  # Darkening, Noise, Rain, Snow, Haze

GTSRB_CLASSES = {
    0: "speed_20", 1: "speed_30", 2: "speed_50", 3: "speed_60",
    4: "speed_70", 5: "speed_80", 6: "speed_80_end", 7: "speed_100",
    8: "speed_120", 9: "no_overtaking", 10: "no_overtaking_trucks",
    11: "right_of_way", 12: "priority_road", 13: "yield",
    14: "stop", 15: "no_vehicles", 16: "no_trucks", 17: "no_entry",
    18: "general_warning", 19: "curve_left", 20: "curve_right",
    21: "double_curve", 22: "bumpy_road", 23: "slippery",
    24: "narrow_right", 25: "road_work", 26: "traffic_signal",
    27: "pedestrian", 28: "children", 29: "bicycle_crossing",
    30: "snow", 31: "animals", 32: "end_all_restrictions",
    33: "right_ahead", 34: "left_ahead", 35: "ahead_only",
    36: "ahead_right", 37: "ahead_left", 38: "keep_right",
    39: "keep_left", 40: "roundabout", 41: "end_overtaking",
    42: "end_overtaking_trucks",
}

# 我提议的 CURE → GTSRB 映射（带置信度，目测后可调整）
PROPOSED_MAPPING = {
    1:  {"gtsrb_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8], "confidence": "low",
         "note": "CURE conflates 8+ GTSRB speed limits"},
    2:  {"gtsrb_classes": [16], "confidence": "medium",
         "note": "CURE goods_vehicles likely = GTSRB no_trucks; visual confirmation needed"},
    3:  {"gtsrb_classes": [9], "confidence": "high",
         "note": "Both = no overtaking"},
    4:  {"gtsrb_classes": [], "confidence": "none",
         "note": "GTSRB has no no_stopping"},
    5:  {"gtsrb_classes": [], "confidence": "none",
         "note": "GTSRB has no no_parking"},
    6:  {"gtsrb_classes": [14], "confidence": "high",
         "note": "Both = stop sign"},
    7:  {"gtsrb_classes": [29], "confidence": "low",
         "note": "GTSRB 29 is bicycle_crossing warning, semantic mismatch"},
    8:  {"gtsrb_classes": [22], "confidence": "low",
         "note": "CURE hump vs GTSRB bumpy_road, similar but distinct"},
    9:  {"gtsrb_classes": [], "confidence": "none",
         "note": "GTSRB has no no_left"},
    10: {"gtsrb_classes": [], "confidence": "none",
         "note": "GTSRB has no no_right"},
    11: {"gtsrb_classes": [12], "confidence": "high",
         "note": "Both = priority road"},
    12: {"gtsrb_classes": [17], "confidence": "high",
         "note": "Both = no entry"},
    13: {"gtsrb_classes": [13], "confidence": "high",
         "note": "Both = yield"},
    14: {"gtsrb_classes": [], "confidence": "none",
         "note": "GTSRB has no parking sign"},
}


# ============================================================
def scan_all_files():
    print("=" * 70)
    print("  Step 1: 递归扫描 CURE-TSR")
    print("=" * 70)

    if not CURE_TSR_DIR.exists():
        print(f"[!] {CURE_TSR_DIR} 不存在 — 数据未下载")
        return None

    all_files = []
    for ext in ('*.bmp', '*.BMP', '*.png', '*.PNG', '*.jpg', '*.JPG'):
        all_files.extend(CURE_TSR_DIR.rglob(ext))

    print(f"  Found {len(all_files)} image files (recursive)")
    if not all_files:
        return None

    parsed, bad = [], []
    for fpath in all_files:
        m = FILENAME_PATTERN.match(fpath.name)
        if m:
            parsed.append({
                "path": fpath,
                "sequence_type": int(m.group(1)),
                "sign_type": int(m.group(2)),
                "challenge_type": int(m.group(3)),
                "challenge_level": int(m.group(4)),
                "index": int(m.group(5)),
                "filename": fpath.name,
            })
        else:
            bad.append(str(fpath.relative_to(CURE_TSR_DIR)))

    print(f"  Parseable: {len(parsed)}")
    print(f"  Unparseable: {len(bad)}")
    if bad[:3]:
        print(f"  Bad examples: {bad[:3]}")

    print(f"\n  Top-level directory contents:")
    for item in sorted(os.listdir(CURE_TSR_DIR)):
        item_path = CURE_TSR_DIR / item
        if item_path.is_dir():
            n = sum(1 for _ in item_path.rglob('*.bmp'))
            print(f"    {item}/  ({n} bmp files)")
        else:
            print(f"    {item}")

    return parsed, bad


def metadata_check(parsed):
    print("\n" + "=" * 70)
    print("  Step 2: 元数据范围验证")
    print("=" * 70)

    seq_ids = sorted(set(p["sequence_type"] for p in parsed))
    print(f"  Sequence types: {seq_ids}")
    for sid in seq_ids:
        n = sum(1 for p in parsed if p["sequence_type"] == sid)
        name = SEQUENCE_TYPES.get(sid, "UNEXPECTED")
        print(f"    {sid:02d} ({name:8s}): {n:>8d}")

    sign_ids = sorted(set(p["sign_type"] for p in parsed))
    print(f"\n  Sign types: {sign_ids}")
    expected = set(range(1, 15))
    if set(sign_ids) == expected:
        print(f"    ✓ All 14 expected sign types present")
    else:
        missing = expected - set(sign_ids)
        extra = set(sign_ids) - expected
        if missing: print(f"    ⚠ Missing: {sorted(missing)}")
        if extra:   print(f"    ⚠ Unexpected: {sorted(extra)}")

    chal_ids = sorted(set(p["challenge_type"] for p in parsed))
    print(f"\n  Challenge types:")
    for cid in chal_ids:
        name = CHALLENGE_TYPES.get(cid, "UNKNOWN")
        n = sum(1 for p in parsed if p["challenge_type"] == cid)
        marker = "★" if cid in PAPER_RELEVANT_CHALLENGES or cid == 0 else " "
        print(f"   {marker} {cid:02d} ({name:18s}): {n:>8d}")


def build_matrices(parsed):
    print("\n" + "=" * 70)
    print("  Step 3: Real-data 统计矩阵")
    print("=" * 70)

    real = [p for p in parsed if p["sequence_type"] == 1]
    print(f"  Real-data total: {len(real)}")

    # Class × Challenge matrix
    m1 = defaultdict(int)
    for p in real:
        m1[(p["sign_type"], p["challenge_type"])] += 1

    csv1 = OUTPUT_DIR / "inspection_summary.csv"
    with open(csv1, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        cids = sorted(set(p["challenge_type"] for p in real))
        w.writerow(["sign_id", "sign_name"] + 
                   [f"{c:02d}_{CHALLENGE_TYPES[c]}" for c in cids] + ["TOTAL"])
        for sid in sorted(set(p["sign_type"] for p in real)):
            row = [sid, CURE_CLASSES.get(sid, "?")]
            t = 0
            for c in cids:
                n = m1.get((sid, c), 0)
                row.append(n); t += n
            row.append(t)
            w.writerow(row)
    print(f"  [✓] {csv1}")

    # Challenge × Severity
    m2 = defaultdict(int)
    for p in real:
        m2[(p["challenge_type"], p["challenge_level"])] += 1

    csv2 = OUTPUT_DIR / "challenge_severity_matrix.csv"
    with open(csv2, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        sevs = sorted(set(p["challenge_level"] for p in real))
        w.writerow(["challenge_id", "challenge_name"] + 
                   [f"sev_{s:02d}" for s in sevs] + ["TOTAL"])
        for c in sorted(set(p["challenge_type"] for p in real)):
            row = [c, CHALLENGE_TYPES.get(c, "?")]
            t = 0
            for s in sevs:
                n = m2.get((c, s), 0)
                row.append(n); t += n
            row.append(t)
            w.writerow(row)
    print(f"  [✓] {csv2}")

    print(f"\n  Class distribution (real):")
    print(f"  {'cls':<4} {'name':<20} {'total':>10}")
    for sid in sorted(set(p["sign_type"] for p in real)):
        n = sum(1 for p in real if p["sign_type"] == sid)
        print(f"  {sid:<4} {CURE_CLASSES.get(sid, '?'):<20} {n:>10d}")

    return real


def make_thumbnails(real_data):
    print("\n" + "=" * 70)
    print("  Step 4: 抽样图（ChallengeFree real）")
    print("=" * 70)
    random.seed(42)

    by_class = defaultdict(list)
    for p in real_data:
        if p["challenge_type"] == 0:
            by_class[p["sign_type"]].append(p)

    sign_ids = sorted(set(p["sign_type"] for p in real_data))
    n_per = 5
    n_classes = len(sign_ids)

    fig, axes = plt.subplots(n_classes, n_per + 1,
                             figsize=(n_per * 1.5 + 2.8, n_classes * 1.3))
    if n_classes == 1:
        axes = axes[None, :]

    for row, sid in enumerate(sign_ids):
        prop = PROPOSED_MAPPING.get(sid, {})
        gids = prop.get("gtsrb_classes", [])
        conf = prop.get("confidence", "none")

        if gids:
            gstr = (f"GTSRB {gids[0]} ({GTSRB_CLASSES[gids[0]]})" if len(gids) == 1
                    else f"GTSRB {min(gids)}-{max(gids)} ({len(gids)} classes)")
            text = f"CURE {sid}: {CURE_CLASSES[sid]}\n→ {gstr}\nconf: {conf}"
        else:
            text = f"CURE {sid}: {CURE_CLASSES[sid]}\n→ NO MATCH\nconf: none"

        color = {"high": "#2ca02c", "medium": "#ff7f0e",
                 "low": "#d62728", "none": "#666666"}.get(conf, "#000")
        axes[row, 0].text(0.05, 0.5, text, fontsize=7, va='center',
                          color=color, fontweight='bold',
                          transform=axes[row, 0].transAxes)
        axes[row, 0].axis('off')

        candidates = by_class.get(sid, [])
        sampled = random.sample(candidates, min(n_per, len(candidates))) if candidates else []
        for col_idx in range(n_per):
            ax = axes[row, col_idx + 1]
            if col_idx < len(sampled):
                img = cv2.imread(str(sampled[col_idx]["path"]))
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"{img.shape[0]}x{img.shape[1]}", fontsize=6)
            ax.axis('off')

    plt.tight_layout()
    out = OUTPUT_DIR / "sample_thumbnails.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [✓] {out}")
    print(f"  ⚠ 必须用图片浏览器打开目测，发给 Claude 确认置信度")


def write_outputs(real_data):
    print("\n" + "=" * 70)
    print("  Step 5: 决策建议")
    print("=" * 70)

    high_conf = [c for c, info in PROPOSED_MAPPING.items() 
                 if info["confidence"] == "high"]

    samples_per_chal = {}
    for cid in PAPER_RELEVANT_CHALLENGES:
        n = sum(1 for p in real_data
                if p["challenge_type"] == cid and p["sign_type"] in high_conf)
        samples_per_chal[CHALLENGE_TYPES[cid]] = n

    n_cf_real = sum(1 for p in real_data if p["challenge_type"] == 0)

    proposal = {
        "high_confidence_classes": high_conf,
        "high_confidence_class_names": [CURE_CLASSES[c] for c in high_conf],
        "samples_per_paper_challenge_high_conf": samples_per_chal,
        "challengefree_real_total": n_cf_real,
        "detailed_mapping": {
            cls: {"cure_name": CURE_CLASSES[cls], **PROPOSED_MAPPING[cls]}
            for cls in CURE_CLASSES
        },
    }
    out_json = OUTPUT_DIR / "mapping_proposal.json"
    with open(out_json, 'w') as f:
        json.dump(proposal, f, indent=2)
    print(f"  [✓] {out_json}")

    lines = [
        "=" * 70,
        "  CURE-TSR Evaluation 路径决策",
        "=" * 70,
        "",
        f"High-confidence classes: {len(high_conf)}",
        f"  IDs: {high_conf}",
        f"  Names: {[CURE_CLASSES[c] for c in high_conf]}",
        "",
        "Sample counts (high-conf only) on paper-relevant challenges:",
    ]
    for ch, n in samples_per_chal.items():
        lines.append(f"  {ch:18s}: {n:>6d}")
    lines.append(f"\nReference: ChallengeFree real total = {n_cf_real}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("路径 A (推荐): GTSRB-trained model → mapped subset external eval")
    lines.append("-" * 70)

    min_n = min(samples_per_chal.values()) if samples_per_chal else 0
    if len(high_conf) >= 4 and min_n >= 50:
        lines += [
            "  ✓ 可行",
            "  - 不重训任何模型",
            "  - 仅评估 high-conf 子集",
            "  - 最严格的 external validation",
            "  - 工作量: ~30 分钟",
            "  - 下一步: python src/evaluate_cure_tsr_external.py",
        ]
    else:
        if len(high_conf) < 4:
            lines.append(f"  ✗ 不可行: high-conf 类别 {len(high_conf)} < 4")
        if min_n < 50:
            lines.append(f"  ✗ 不可行: 关键 challenge 样本 min={min_n} < 50")

    lines += [
        "",
        "-" * 70,
        "路径 B (fallback): Qualitative-only validation",
        "-" * 70,
        "  ✓ 总是可行",
        "  - 仅做视觉对比 + routing 分布",
        "  - 不报 accuracy 数字",
        "",
        "-" * 70,
        "路径 C: 训练独立 CURE classifier （不推荐）",
        "-" * 70,
        "  ⚠ 削弱 external validation 的说服力",
    ]

    out_txt = OUTPUT_DIR / "decision_recommendation.txt"
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    for line in lines:
        print("  " + line)
    print(f"\n  [✓] {out_txt}")


def main():
    print(f"\nCURE-TSR Inspection (corrected version)")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Dataset: {CURE_TSR_DIR}")
    print(f"Output:  {OUTPUT_DIR}\n")

    res = scan_all_files()
    if res is None:
        return
    parsed, bad = res
    if not parsed:
        return

    metadata_check(parsed)
    real = build_matrices(parsed)
    make_thumbnails(real)
    write_outputs(real)

    print("\n" + "=" * 70)
    print("  下一步：")
    print("=" * 70)
    print(f"  1. 打开 {OUTPUT_DIR}/sample_thumbnails.png 目测验证映射")
    print(f"  2. 把 sample_thumbnails.png 发给 Claude 确认置信度")
    print(f"  3. 看 {OUTPUT_DIR}/decision_recommendation.txt 选 A/B/C")
    print("=" * 70)


if __name__ == "__main__":
    main()
