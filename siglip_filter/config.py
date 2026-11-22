"""YAML 配置。判断用的数都在配置里。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from siglip_filter.prompts import PROMPTS_VERSION


@dataclass
class FilterConfig:
    model_name: str
    model_dtype: str
    batch_size: int
    device: str
    interval_s: float
    image_size: int
    roi: dict
    quality: dict
    temporal: dict
    classify: dict
    pilot_max_hours: float
    workers: int
    annotation_windows: int
    raw: dict

    @classmethod
    def from_dict(cls, raw: dict) -> "FilterConfig":
        model = raw["model"]
        sample = raw["sample"]
        temporal = dict(raw["temporal"])
        boundary = float(temporal["boundary_context_s"])
        if not 2.0 <= boundary <= 5.0:
            raise ValueError(
                f"boundary_context_s must be between 2 and 5, got {boundary}"
            )
        interval = float(sample["interval_s"])
        window = float(temporal["window_s"])
        if interval <= 0 or window < interval:
            raise ValueError(
                f"window_s ({window}) must be >= interval_s ({interval}) > 0"
            )
        view = temporal.get("decision_view", "full")
        if view not in ("full", "roi"):
            raise ValueError(f"decision_view must be full or roi, got {view}")
        roi = dict(raw["roi"])
        if view == "roi" and not roi.get("enabled", True):
            raise ValueError("decision_view roi requires roi.enabled")
        mode = raw["classify"].get("mode", "zero_shot")
        if mode not in ("zero_shot", "logistic"):
            raise ValueError(f"classify.mode must be zero_shot or logistic, got {mode}")
        workers = int(raw.get("workers", 1))
        if workers < 1:
            raise ValueError("workers must be >= 1")
        return cls(
            model_name=str(model["name"]),
            model_dtype=str(model.get("dtype", "auto")),
            batch_size=int(model.get("batch_size", 32)),
            device=str(model.get("device", "auto")),
            interval_s=interval,
            image_size=int(sample["image_size"]),
            roi=roi,
            quality=dict(raw["quality"]),
            temporal=temporal,
            classify=dict(raw["classify"]),
            pilot_max_hours=float(raw["pilot"]["max_hours"]),
            workers=workers,
            annotation_windows=int(raw.get("annotation", {}).get("target_windows", 500)),
            raw=raw,
        )

    @classmethod
    def load(cls, path: Path) -> "FilterConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def sample_payload(self) -> dict:
        return {
            "interval_s": self.interval_s,
            "image_size": self.image_size,
            "roi": self.roi,
        }

    def decision_payload(self) -> dict:
        return {
            "model": self.model_name,
            "sample": self.sample_payload(),
            "quality": self.quality,
            "temporal": self.temporal,
            "classify": self.classify,
            "prompts": PROMPTS_VERSION,
        }

    def sample_hash(self) -> str:
        return _hash(self.sample_payload())

    def decision_hash(self) -> str:
        return _hash(self.decision_payload())


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
