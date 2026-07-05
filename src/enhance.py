"""
图像增强方法（论文 Section III 的核心）。

包含：
  - compute_image_stats: 计算单图的 brightness/contrast/edge 三个统计量
  - apply_clahe        : 固定 CLAHE 增强（baseline 1）
  - apply_gamma        : 固定 gamma 校正（baseline 2）
  - adaptive_enhance   : 能见度感知自适应增强（本文方法）
  - no_enhance         : 不预处理（baseline 0）

所有函数输入输出都是 BGR uint8 ndarray（OpenCV 标准）。
"""
import cv2
import numpy as np


def compute_image_stats(img_bgr):
    """计算单张图的 brightness / contrast / edge_strength 统计量。

    Args:
        img_bgr: BGR ndarray (H, W, 3)

    Returns:
        (brightness, contrast, edge): 三个 0-1 之间的 float
        - brightness: 灰度均值 / 255
        - contrast:   灰度标准差 / 128（大致 0-1）
        - edge:       Canny 边缘像素均值 / 255
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 128.0
    edges = cv2.Canny(gray, 50, 150)
    edge_strength = float(edges.mean()) / 255.0
    return brightness, contrast, edge_strength


def no_enhance(img_bgr):
    """不做任何增强（baseline 0：直接送进网络）。"""
    return img_bgr


def apply_clahe(img_bgr, clip_limit=3.0, tile_grid=(8, 8)):
    """固定 CLAHE 增强（baseline 1）。

    在 LAB 空间的 L 通道做 CLAHE，避免破坏色彩。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma(img_bgr, gamma=0.5):
    """固定 Gamma 校正（baseline 2）：gamma < 1 提亮。"""
    img_norm = img_bgr.astype(np.float32) / 255.0
    out = np.power(img_norm, gamma)
    return (out * 255).astype(np.uint8)


def adaptive_enhance(img_bgr, T1, T2, T3, T4):
    """能见度感知自适应增强（本文核心方法）。

    根据图像统计量自动选择增强策略：
      - brightness < T1                 → low_light    → Gamma 校正提亮
      - contrast < T2                   → low_contrast → CLAHE
      - edge < T3 且 brightness > T4    → foggy        → 对比度拉伸
      - 否则                             → normal       → 不处理

    Args:
        img_bgr: BGR ndarray
        T1: brightness 低阈值
        T2: contrast 低阈值
        T3: edge 低阈值
        T4: brightness 高阈值

    Returns:
        增强后的 ndarray，与输入同 shape 同 dtype。
    """
    b, c, e = compute_image_stats(img_bgr)

    if b < T1:
        # low_light: Gamma 校正提亮
        return apply_gamma(img_bgr, gamma=0.5)
    elif c < T2:
        # low_contrast: CLAHE
        return apply_clahe(img_bgr)
    elif e < T3 and b > T4:
        # foggy: 简易去雾（对比度拉伸）
        img_f = img_bgr.astype(np.float32) / 255.0
        out = np.clip((img_f - 0.5) * 1.5 + 0.5, 0, 1) * 255
        return out.astype(np.uint8)
    else:
        # normal: 不处理
        return img_bgr


if __name__ == '__main__':
    test = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    print("增强函数自测：")
    print(f"  compute_image_stats: {compute_image_stats(test)}")
    for name, fn in [
        ('no_enhance',     no_enhance),
        ('apply_clahe',    apply_clahe),
        ('apply_gamma',    apply_gamma),
        ('adaptive_enhance', lambda x: adaptive_enhance(x, 0.3, 0.2, 0.1, 0.5)),
    ]:
        out = fn(test)
        assert out.shape == test.shape and out.dtype == test.dtype
        print(f"  {name:18s} OK  mean={out.mean():.1f}  std={out.std():.1f}")
    print("\n全部通过")
