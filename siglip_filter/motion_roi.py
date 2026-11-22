"""工位多边形里的运动。低运动本身不是无关。"""

from __future__ import annotations

import numpy as np

from siglip_filter.hands import work_mask
from siglip_filter.quality import gray_of


class MotionRoi:
    def __init__(self, polygon: list, diff_threshold: float = 12.0, background_alpha: float = 0.02):
        self.polygon = polygon
        self.diff_threshold = float(diff_threshold)
        self.background_alpha = float(background_alpha)
        self._prev = None
        self._background = None
        self._mask = None

    def measure(self, image) -> dict:
        gray = gray_of(image, max_side=10**9)
        height, width = gray.shape
        if self._mask is None or self._mask.shape != gray.shape:
            self._mask = work_mask(width, height, self.polygon)
        mask = self._mask
        count = int(mask.sum())
        if count == 0:
            raise ValueError("work area polygon covers no pixels")
        if self._prev is None or self._prev.shape != gray.shape:
            frame_diff = 0.0
        else:
            frame_diff = float(np.abs(gray - self._prev)[mask].mean()) / 255.0
        if self._background is None or self._background.shape != gray.shape:
            background_diff = 0.0
            self._background = gray.copy()
        else:
            background_diff = float(np.abs(gray - self._background)[mask].mean()) / 255.0
            self._background = (1.0 - self.background_alpha) * self._background + self.background_alpha * gray
        if self._prev is None or self._prev.shape != gray.shape:
            motion_ratio = 0.0
        else:
            motion_ratio = float((np.abs(gray - self._prev)[mask] > self.diff_threshold).mean())
        self._prev = gray
        return {
            "frame_diff": frame_diff,
            "background_diff": background_diff,
            "motion_ratio": motion_ratio,
            "motion_score": frame_diff,
        }
