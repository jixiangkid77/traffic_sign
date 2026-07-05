"""
revision_utils.py — Day 1 共享工具（v5）

修复历史：
  v4: 全部数据加载逻辑就位
  v5: ★ Pipeline 改为 paper-consistent (enhance at original size, then PIL resize)
      - 添加 build_paper_test_transform()
      - compute_image_stats / compute_noise_stat 不再要求预先 resize

【关键 pipeline 变化（v4 → v5）】
  v4 错误：cv2.imread → cv2.resize(32x32) → enhance → cv2 BGR2RGB → torchvision transform
  v5 正确：cv2.imread → enhance (at original size) → cv2 BGR2RGB → PIL → transforms.Resize → ToTensor → Normalize
"""

import os
import sys
import json
import csv as _csv
import platform
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs_revision"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

GTSRB_TRAIN_DIR        = DATA_ROOT / "gtsrb" / "GTSRB" / "Training"
GTSRB_FINAL_TEST_DIR   = DATA_ROOT / "gtsrb" / "GTSRB" / "Final_Test"
GTSRB_TEST_IMAGES_DIR  = GTSRB_FINAL_TEST_DIR / "Images"
GTSRB_TEST_CSV         = DATA_ROOT / "gtsrb" / "GT-final_test.csv"

DEGRADED_TESTSET_DIRS = {
    "lowlight":    DATA_ROOT / "gtsrb_lowlight",
    "foggy":       DATA_ROOT / "gtsrb_foggy",
    "lowcontrast": DATA_ROOT / "gtsrb_lowcontrast",
    "noisy":       DATA_ROOT / "gtsrb_noisy",
    "mixed":       DATA_ROOT / "gtsrb_mixed",
}

GTSRB_MODEL_PATH = MODELS_DIR / "mbnetv3_baseline.pth"

GTSRB_MEAN = [0.3401, 0.3120, 0.3212]
GTSRB_STD  = [0.2725, 0.2609, 0.2669]
INPUT_SIZE = 32

PAPER_THRESHOLDS = {
    "T1": 0.1206,
    "T2": 0.1061,
    "T3": 0.0726,
    "T4": 0.4085,
}

PAPER_CLEAN_BASELINE_ACC = 93.92


# ============================================================
# v5: Paper-consistent transform builder
# ============================================================
def build_paper_test_transform():
    """构建与 evaluate_all.py 完全一致的 transform pipeline
    
    输入：PIL Image (任意尺寸)
    输出：tensor [3, 32, 32] normalized
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),     # PIL bilinear resize
        transforms.ToTensor(),
        transforms.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
    ])


def cv2_to_pil_rgb(img_bgr):
    """BGR ndarray → PIL RGB Image"""
    import cv2
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ============================================================
# 数据路径验证
# ============================================================
def verify_data_paths(check_csv=True, check_train=True, check_clean=True, check_degraded=True):
    errors = []
    if check_csv and not GTSRB_TEST_CSV.exists():
        errors.append(f"GTSRB test CSV not found: {GTSRB_TEST_CSV}")
    if check_train and not GTSRB_TRAIN_DIR.exists():
        errors.append(f"GTSRB Training dir not found: {GTSRB_TRAIN_DIR}")
    if check_clean and not GTSRB_TEST_IMAGES_DIR.exists():
        errors.append(f"GTSRB Final_Test/Images not found: {GTSRB_TEST_IMAGES_DIR}")
    if check_degraded:
        for name, path in DEGRADED_TESTSET_DIRS.items():
            if not path.exists():
                errors.append(f"Degraded testset '{name}' not found: {path}")
    if not GTSRB_MODEL_PATH.exists():
        errors.append(f"Model not found: {GTSRB_MODEL_PATH}")
    if errors:
        msg = "\n".join(f"  ✗ {e}" for e in errors)
        raise FileNotFoundError(f"verify_data_paths FAILED:\n{msg}")
    print("[verify] All required data paths OK")


# ============================================================
# CSV-based label loading
# ============================================================
def load_gtsrb_test_labels():
    if not GTSRB_TEST_CSV.exists():
        raise FileNotFoundError(f"Test label CSV not found: {GTSRB_TEST_CSV}")
    label_map = {}
    with open(GTSRB_TEST_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f, delimiter=";")
        for row in reader:
            filename = row.get("Filename") or row.get("filename") or row.get("FileName")
            class_id = row.get("ClassId") or row.get("class_id") or row.get("ClassID")
            if filename is None or class_id is None:
                raise KeyError(f"CSV columns not recognized: {reader.fieldnames}")
            stem = Path(filename).stem
            label_map[stem] = int(class_id)
    if len(label_map) == 0:
        raise RuntimeError(f"No labels loaded from {GTSRB_TEST_CSV}")
    return label_map


def collect_gtsrb_test_samples(testset_name):
    label_map = load_gtsrb_test_labels()
    if testset_name == "clean":
        root = GTSRB_TEST_IMAGES_DIR
        exts = ("*.ppm", "*.PPM", "*.png", "*.PNG", "*.jpg", "*.JPG")
    else:
        if testset_name not in DEGRADED_TESTSET_DIRS:
            raise KeyError(f"Unknown testset: {testset_name}")
        root = DEGRADED_TESTSET_DIRS[testset_name]
        exts = ("*.png", "*.PNG", "*.ppm", "*.PPM", "*.jpg", "*.JPG")
    if not root.exists():
        raise FileNotFoundError(f"Testset directory not found: {root}")
    samples = []
    seen_stems = set()
    for ext in exts:
        for f in sorted(root.glob(ext)):
            stem = f.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            if stem in label_map:
                samples.append((str(f), label_map[stem]))
    if len(samples) == 0:
        raise RuntimeError(f"No labeled samples found for '{testset_name}' in {root}")
    return samples


def collect_gtsrb_train_samples():
    if not GTSRB_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training dir not found: {GTSRB_TRAIN_DIR}")
    samples = []
    for class_dir in sorted(GTSRB_TRAIN_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        for ext in ("*.ppm", "*.PPM", "*.png", "*.PNG", "*.jpg", "*.JPG"):
            for f in class_dir.glob(ext):
                samples.append((str(f), class_id))
    if len(samples) == 0:
        raise RuntimeError(f"No train samples found in {GTSRB_TRAIN_DIR}")
    return samples


def collect_gtsrb_train_samples_stratified(samples_per_class=50, seed=42):
    import random
    rng = random.Random(seed)
    if not GTSRB_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training dir not found: {GTSRB_TRAIN_DIR}")
    by_class = {}
    for class_dir in sorted(GTSRB_TRAIN_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        files = []
        for ext in ("*.ppm", "*.PPM", "*.png", "*.PNG", "*.jpg", "*.JPG"):
            files.extend(class_dir.glob(ext))
        if files:
            by_class[class_id] = files
    if not by_class:
        raise RuntimeError(f"No class folders found in {GTSRB_TRAIN_DIR}")
    sampled = []
    for class_id in sorted(by_class.keys()):
        files = list(by_class[class_id])
        rng.shuffle(files)
        n = min(samples_per_class, len(files))
        for f in files[:n]:
            sampled.append((str(f), class_id))
    return sampled


# ============================================================
# 环境信息 + run config
# ============================================================
def collect_environment_info():
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu": platform.processor() or "unknown",
        "cwd": os.getcwd(),
    }
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
    except ImportError:
        info["opencv_version"] = "not installed"
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
    return info


def init_run_config(script_name, params=None):
    timestamp = datetime.now()
    config = {
        "script_name": script_name,
        "timestamp": timestamp.isoformat(),
        "timestamp_human": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": collect_environment_info(),
        "params": params or {},
        "model_path": str(GTSRB_MODEL_PATH),
        "outputs_dir": str(OUTPUTS_DIR),
    }
    config_path = OUTPUTS_DIR / f"run_config_{script_name}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str)
    log_path = OUTPUTS_DIR / "execution_log.txt"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{config['timestamp_human']}] STARTED: {script_name}\n")
    print(f"[init] Run config: {config_path}")
    return config


def finalize_run(script_name, summary=None):
    timestamp = datetime.now()
    log_path = OUTPUTS_DIR / "execution_log.txt"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] FINISHED: {script_name}\n")
        if summary:
            for line in summary.split("\n"):
                f.write(f"  {line}\n")
    print(f"[done] {script_name} finished.")


# ============================================================
# 模型加载
# ============================================================
def _safe_torch_load(model_path, device):
    import torch
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError as e:
        if "weights_only" in str(e):
            return torch.load(model_path, map_location=device)
        raise


def load_gtsrb_compactcnn(device="cpu"):
    sys.path.insert(0, str(SRC_DIR))
    import torch
    from model import CompactCNN
    if not GTSRB_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {GTSRB_MODEL_PATH}")
    model = CompactCNN(num_classes=43)
    ckpt = _safe_torch_load(GTSRB_MODEL_PATH, device)
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# 图像统计 (与 enhance.py 一致 — 不要 pre-resize)
# ============================================================
def compute_image_stats(img_bgr):
    """计算 brightness, contrast, edge_strength
    
    ★ v5 注意：必须在原始尺寸上调用，不要先 resize。
    与 enhance.py 的 compute_image_stats 完全一致。
    """
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 128.0
    edges = cv2.Canny(gray, 50, 150)
    edge_strength = float(edges.mean()) / 255.0
    return brightness, contrast, edge_strength


def compute_noise_stat(img_bgr):
    """高频残差能量 ★ v5: 在原始尺寸上调用"""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    smoothed = cv2.medianBlur((gray * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    return float(np.std(gray - smoothed))


def per_class_accuracy(y_true, y_pred, n_classes):
    import numpy as np
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accs = np.zeros(n_classes)
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            accs[c] = (y_pred[mask] == c).mean() * 100
        else:
            accs[c] = float('nan')
    return accs


if __name__ == "__main__":
    print("=" * 60)
    print("revision_utils.py self-test")
    print("=" * 60)
    print(f"GTSRB_TRAIN_DIR exists: {GTSRB_TRAIN_DIR.exists()}")
    print(f"GTSRB_TEST_CSV exists: {GTSRB_TEST_CSV.exists()}")
    for name, path in DEGRADED_TESTSET_DIRS.items():
        print(f"  {name}: {path.exists()}")
    print(f"Model exists: {GTSRB_MODEL_PATH.exists()}")
    verify_data_paths()
    labels = load_gtsrb_test_labels()
    print(f"\nLoaded {len(labels)} labels, sample: '00000' -> {labels.get('00000')}")
    for ts in ["clean", "noisy"]:
        print(f"  {ts}: {len(collect_gtsrb_test_samples(ts))} samples")
    print("\n[OK] All checks passed.")
