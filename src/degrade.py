"""
图像退化函数。

包含 4 种单一退化（lowlight / foggy / lowcontrast / noisy）
和 1 种混合退化（mixed），用于生成论文测试集。

参数选取参考公开 benchmark：
- lowlight: 参考 ExDark 数据集
- foggy:    参考 RESIDE 数据集（大气散射模型）
- 其他:    参考 ImageNet-C corruption benchmark

设计原则：
- 所有函数输入输出都是 BGR 格式 uint8 ndarray（OpenCV 标准）
- 退化强度统一在"中等偏强"水平：能明显影响识别但不至于完全无法看清
"""
import cv2
import numpy as np


def degrade_lowlight(img, gamma=3.0):
    """低光照退化：gamma 校正模拟夜间拍摄。

    Args:
        img: BGR 格式 ndarray (H, W, 3)
        gamma: > 1 变暗，越大越暗。默认 3.0 对应中等夜间。

    Returns:
        退化后的 ndarray，同输入 shape 和 dtype。
    """
    img_norm = img.astype(np.float32) / 255.0
    out = np.power(img_norm, gamma)
    return (out * 255).astype(np.uint8)


def degrade_foggy(img, beta=0.6, A=220):
    """雾化退化：基于大气散射模型 I = J*t + A*(1-t)。

    Args:
        img: BGR 格式 ndarray
        beta: 雾的浓度（散射系数），越大越浓。默认 0.6 对应中等雾。
        A: 大气光强度，0-255 之间。默认 220 对应明亮的雾。

    Returns:
        退化后的 ndarray。
    """
    h, w = img.shape[:2]
    # 简化为均匀深度（真实场景应有深度图，但交通标志近距离拍摄可忽略）
    t = np.exp(-beta) * np.ones((h, w, 1), dtype=np.float32)
    img_f = img.astype(np.float32)
    fog = img_f * t + A * (1 - t)
    return np.clip(fog, 0, 255).astype(np.uint8)


def degrade_lowcontrast(img, factor=0.3):
    """低对比度退化：将像素值压缩到中间灰度区。

    Args:
        img: BGR 格式 ndarray
        factor: 0.0~1.0，越小对比度越低。默认 0.3 对应明显对比度下降。

    Returns:
        退化后的 ndarray。
    """
    img_f = img.astype(np.float32)
    mean = np.mean(img_f)
    out = (img_f - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_noisy(img, sigma=25):
    """加性高斯噪声。

    Args:
        img: BGR 格式 ndarray
        sigma: 噪声标准差（在 0-255 灰度范围内）。默认 25 对应明显雪花点。

    Returns:
        退化后的 ndarray。
    """
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_mixed(img, rng=None):
    """混合退化：随机选 2-3 种单一退化叠加，模拟真实场景。

    Args:
        img: BGR 格式 ndarray
        rng: numpy.random.Generator，用于复现性。None 时新建一个。

    Returns:
        退化后的 ndarray。
    """
    if rng is None:
        rng = np.random.default_rng()

    out = img.copy()
    funcs = [
        ('lowlight',    lambda x: degrade_lowlight(x, gamma=rng.uniform(1.8, 2.5))),
        ('foggy',       lambda x: degrade_foggy(x, beta=rng.uniform(0.3, 0.5))),
        ('lowcontrast', lambda x: degrade_lowcontrast(x, factor=rng.uniform(0.4, 0.6))),
        ('noisy',       lambda x: degrade_noisy(x, sigma=rng.uniform(10, 20))),
    ]

# 固定叠加 2 种退化（3 种叠加会摧毁图像，不利于评估）
    n = 2
    chosen_idx = sorted(rng.choice(len(funcs), size=n, replace=False).tolist())

    for idx in chosen_idx:
        _, fn = funcs[idx]
        out = fn(out)

    return out


# 字典形式方便批量调用
DEGRADATIONS = {
    'lowlight':    degrade_lowlight,
    'foggy':       degrade_foggy,
    'lowcontrast': degrade_lowcontrast,
    'noisy':       degrade_noisy,
    'mixed':       degrade_mixed,
}


if __name__ == '__main__':
    # 简单自测：生成一张 100x100 的灰色图，跑 5 种退化看是否报错
    test = np.full((100, 100, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(42)
    for name, fn in DEGRADATIONS.items():
        if name == 'mixed':
            out = fn(test, rng=rng)
        else:
            out = fn(test)
        assert out.shape == test.shape and out.dtype == test.dtype
        print(f"  {name:12s} OK  mean={out.mean():.1f}  std={out.std():.1f}")
    print("\n所有退化函数自测通过。")
