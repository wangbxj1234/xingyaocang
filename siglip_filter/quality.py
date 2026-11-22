"""CPU 上的画面质量。这些数只作记录，低运动本身不构成无关。"""

from __future__ import annotations

from PIL import Image
import numpy as np


def gray_of(image: Image.Image, max_side: int = 256) -> np.ndarray:
    im = image.convert("RGB")
    width, height = im.size
    scale = max_side / float(max(width, height))
    if scale < 1:
        im = im.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.BILINEAR,
        )
    arr = np.asarray(im).astype(np.float32)
    return 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]


def laplacian_var(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var()) if lap.size else 0.0


def mean_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float32).ravel()
    y = right.astype(np.float32).ravel()
    if x.shape != y.shape:
        raise ValueError("ssim inputs must share a shape")
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(x.var())
    var_y = float(y.var())
    cov = float(((x - mu_x) * (y - mu_y)).mean())
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / den)


def measure(image: Image.Image, prev: np.ndarray | None, cfg: dict) -> tuple[dict, np.ndarray]:
    """返回这一帧的质量，以及给下一帧用的灰度图。"""
    gray = gray_of(image)
    luma = float(gray.mean())
    blur = laplacian_var(gray)
    mad = None
    similarity = None
    if prev is not None and prev.shape == gray.shape:
        mad = mean_abs(gray, prev)
        similarity = ssim(gray, prev)
    row = {
        "decode_ok": True,
        "black": bool(luma <= float(cfg["black_mean_max"])),
        "luma": luma,
        "blur": blur,
        "blurry": bool(blur < float(cfg["sharp_var_min"])),
        "mad": mad,
        "ssim": similarity,
        "duplicate": bool(mad is not None and mad <= float(cfg["duplicate_mad_max"])),
        "frozen": bool(
            similarity is not None and similarity >= float(cfg["frozen_ssim_min"])
        ),
        "motion": 0.0 if mad is None else mad / 255.0,
    }
    return row, gray


def failed_measure(error: str) -> dict:
    return {
        "decode_ok": False,
        "decode_error": error,
        "black": False,
        "luma": None,
        "blur": None,
        "blurry": False,
        "mad": None,
        "ssim": None,
        "duplicate": False,
        "frozen": False,
        "motion": 0.0,
    }
