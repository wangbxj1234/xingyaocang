"""无效片段只看手和画面动不动。不读 SigLIP。"""

from __future__ import annotations


FUSION_LABELS = ("kept", "idle_no_person", "present_but_unrelated", "uncertain")


def _covers(stamp: float, row: dict) -> bool:
    if float(row["start"]) - 1e-9 <= stamp < float(row["end"]) - 1e-9:
        return True
    return abs(stamp - float(row["end"])) <= 1e-6


def _in_idle(stamp: float, intervals: list[dict]) -> dict | None:
    hits = [row for row in intervals if _covers(stamp, row)]
    for row in hits:
        if row.get("label") == "idle_no_person":
            return row
    return hits[0] if hits else None


def sample_decision(sample: dict, intervals: list[dict], temporal: dict) -> tuple[str, str, float]:
    stamp = float(sample["timestamp_s"])
    idle = _in_idle(stamp, intervals)
    if idle is not None:
        return (
            str(idle.get("label") or "idle_no_person"),
            str(idle.get("reason") or "no_hand"),
            float(idle.get("confidence") or 1.0),
        )
    present = bool(sample.get("hand_present_in_work_area"))
    motion = float(sample.get("motion_score") or 0.0)
    motion_low = float(sample.get("motion_low", temporal.get("motion_low", 0.02)))
    if not present and motion > motion_low:
        return "uncertain", "absent_high_motion", motion
    if not present:
        return "uncertain", "no_hand_short", motion
    return "kept", "hand_moving", motion


def fuse_timeline(samples: list[dict], intervals: list[dict], duration: float, temporal: dict) -> list[dict]:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not samples:
        return [
            {
                "start": 0.0,
                "end": float(duration),
                "label": "uncertain",
                "reason": "no_sample",
                "confidence": 0.0,
            }
        ]
    ordered = sorted(samples, key=lambda row: float(row["timestamp_s"]))
    pieces = []
    for index, sample in enumerate(ordered):
        start = float(sample["timestamp_s"])
        if index + 1 < len(ordered):
            end = float(ordered[index + 1]["timestamp_s"])
        else:
            end = float(duration)
        if end <= start:
            continue
        label, reason, confidence = sample_decision(sample, intervals, temporal)
        pieces.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "reason": reason,
                "confidence": float(confidence),
                "motion": float(sample.get("motion_score") or 0.0),
            }
        )
    if pieces and pieces[0]["start"] > 1e-6:
        label, reason, confidence = sample_decision(ordered[0], intervals, temporal)
        pieces.insert(
            0,
            {
                "start": 0.0,
                "end": pieces[0]["start"],
                "label": label,
                "reason": reason,
                "confidence": float(confidence),
                "motion": float(ordered[0].get("motion_score") or 0.0),
            },
        )
    merged = []
    for piece in pieces:
        if merged and merged[-1]["label"] == piece["label"] and merged[-1]["reason"] == piece["reason"]:
            merged[-1]["end"] = piece["end"]
            continue
        merged.append(dict(piece))
    if merged:
        merged[-1]["end"] = float(duration)
    return merged
