"""每条 episode 一份清单。时间是原片上的秒。"""

from __future__ import annotations

import json
from pathlib import Path

from siglip_filter import __version__

SEGMENT_KEYS = ("start", "end", "label", "confidence", "reason")


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def public_segment(segment: dict) -> dict:
    out = {key: segment[key] for key in SEGMENT_KEYS}
    out["start"] = float(out["start"])
    out["end"] = float(out["end"])
    out["confidence"] = float(out["confidence"])
    for key in (
        "motion",
        "mean_positive",
        "median_positive",
        "mean_negative",
        "median_negative",
        "mean_uncertain",
        "relevant_ratio",
        "irrelevant_ratio",
    ):
        if key in segment:
            out[key] = float(segment[key])
    return out


def build_manifest(episode: dict, views: dict, meta: dict) -> dict:
    public_views = {
        name: {"segments": [public_segment(segment) for segment in rows]}
        for name, rows in views.items()
    }
    decision = meta["decision_view"]
    return {
        "episode_id": episode["episode_id"],
        "video_path": str(episode["video_path"]),
        "task": episode["task"],
        "duration": float(episode["duration"]),
        "model": meta["model"],
        "pipeline_version": __version__,
        "confidence_kind": meta["confidence_kind"],
        "decision_view": decision,
        "low_quality_confidence": "fraction of bad frames in the window",
        "thresholds": meta["thresholds"],
        "segments": public_views[decision]["segments"],
        "views": public_views,
    }


def assert_manifest(payload: dict) -> None:
    for key in ("episode_id", "video_path", "task", "duration", "segments"):
        if key not in payload:
            raise ValueError(f"manifest missing {key}")
    previous = 0.0
    for segment in payload["segments"]:
        for key in SEGMENT_KEYS:
            if key not in segment:
                raise ValueError(f"segment missing {key}")
        if segment["label"] not in ("relevant", "irrelevant", "uncertain", "low_quality"):
            raise ValueError(f"unknown label {segment['label']}")
        start = float(segment["start"])
        end = float(segment["end"])
        if start < previous - 1e-3:
            raise ValueError("segments overlap or go backwards")
        if abs(start - previous) > 1e-3:
            raise ValueError("segments leave a gap")
        if end < start:
            raise ValueError("segment ends before it starts")
        previous = end
    if abs(previous - float(payload["duration"])) > 1e-3:
        raise ValueError("segments do not cover the episode duration")


def checkpoint_path(out_dir: Path, episode_id: str) -> Path:
    return Path(out_dir) / "checkpoints" / f"{episode_id}.json"


def read_checkpoint(out_dir: Path, episode_id: str) -> dict | None:
    path = checkpoint_path(out_dir, episode_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
