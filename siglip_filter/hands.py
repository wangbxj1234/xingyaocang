"""工位上的手。MediaPipe 只检测手，不检测整个人。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi, yi = float(point[0]), float(point[1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            edge = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < edge:
                inside = not inside
        j = i
    return inside


def work_mask(width: int, height: int, polygon: list) -> np.ndarray:
    canvas = Image.new("1", (width, height), 0)
    points = [(float(point[0]) * width, float(point[1]) * height) for point in polygon]
    ImageDraw.Draw(canvas).polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


def bbox_overlap(bbox: list, polygon: list, width: int, height: int, mask: np.ndarray | None = None) -> float:
    """手框落在工位多边形里的面积比例。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    ix0 = max(0, min(width - 1, int(np.floor(x0))))
    iy0 = max(0, min(height - 1, int(np.floor(y0))))
    ix1 = max(ix0 + 1, min(width, int(np.ceil(x1))))
    iy1 = max(iy0 + 1, min(height, int(np.ceil(y1))))
    full = work_mask(width, height, polygon) if mask is None else mask
    patch = full[iy0:iy1, ix0:ix1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def hand_observation(landmarks: list, score: float, polygon: list, width: int, height: int, min_score: float, min_overlap: float) -> dict:
    xs = [float(point[0]) for point in landmarks]
    ys = [float(point[1]) for point in landmarks]
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    overlap = bbox_overlap(bbox, polygon, width, height)
    present = bool(score >= min_score and overlap >= min_overlap)
    return {
        "hand_detected": True,
        "hand_confidence": float(score),
        "hand_bbox": [round(v, 1) for v in bbox],
        "bbox_overlap_with_work_area": overlap,
        "hand_present_in_work_area": present,
    }


def empty_hand() -> dict:
    return {
        "hand_detected": False,
        "hand_confidence": 0.0,
        "hand_bbox": None,
        "bbox_overlap_with_work_area": 0.0,
        "hand_present_in_work_area": False,
    }


def select_hand(observations: list) -> dict:
    if not observations:
        return empty_hand()
    present = [row for row in observations if row["hand_present_in_work_area"]]
    pool = present or observations
    return max(pool, key=lambda row: row["hand_confidence"])


class HandDetector:
    """MediaPipe HandLandmarker。权重文件由调用方准备。"""

    def __init__(self, model_path: str, min_confidence: float, max_hands: int):
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=int(max_hands),
            min_hand_detection_confidence=float(min_confidence),
            min_hand_presence_confidence=float(min_confidence),
        )
        self._detector = HandLandmarker.create_from_options(options)
        self.min_confidence = float(min_confidence)

    def detect(self, image: Image.Image, polygon: list, min_overlap: float) -> dict:
        rgb = np.ascontiguousarray(image.convert("RGB"))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        height, width = rgb.shape[0], rgb.shape[1]
        observations = []
        landmarks = result.hand_landmarks or []
        handed = result.handedness or []
        for index, hand in enumerate(landmarks):
            score = 0.0
            if index < len(handed) and handed[index]:
                score = float(handed[index][0].score)
            points = [(point.x * width, point.y * height) for point in hand]
            observations.append(
                hand_observation(points, score, polygon, width, height, self.min_confidence, min_overlap)
            )
        chosen = select_hand(observations)
        chosen["n_hands"] = len(observations)
        return chosen

    def close(self) -> None:
        close = getattr(self._detector, "close", None)
        if close:
            close()


def model_path(cfg: dict) -> Path:
    path = cfg.get("model_path")
    if not path:
        raise ValueError("presence.model_path is required for mediapipe_hands")
    return Path(path)
