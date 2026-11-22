"""30 秒窗口、平滑、连续门槛、边界上下文。不确定的留下。"""

from __future__ import annotations

from siglip_filter.classify import frame_label, window_label


def window_bounds(duration: float, window_s: float) -> list[tuple[float, float]]:
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    bounds = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(duration, start + window_s)
        bounds.append((start, end))
        start = end
    return bounds


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def in_window(timestamp: float, start: float, end: float, last: bool) -> bool:
    if last:
        return start - 1e-9 <= timestamp <= end + 1e-9
    return start - 1e-9 <= timestamp < end - 1e-9


def _frames_in(frames: list[dict], start: float, end: float, last: bool) -> list[dict]:
    return [frame for frame in frames if in_window(float(frame["timestamp_s"]), start, end, last)]


def _group_means(rows: list[dict]) -> dict:
    keys = []
    seen = set()
    for row in rows:
        for key in row.get("groups") or {}:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return {
        key: _mean([float(row["groups"][key]) for row in rows if key in (row.get("groups") or {})])
        for key in keys
    }


def summarize_window(start: float, end: float, frames: list[dict], view: str, temporal: dict, quality: dict) -> dict:
    labels = [frame_label(frame, view, temporal) for frame in frames]
    embedded = []
    for frame in frames:
        scores = (frame.get("scores") or {}).get(view)
        if frame.get("embedded") and scores and not frame.get("black"):
            embedded.append(scores)
    n = len(frames)
    relevant = sum(1 for label, _reason, _conf in labels if label == "relevant")
    irrelevant = sum(1 for label, _reason, _conf in labels if label == "irrelevant")
    summary = {
        "start": start,
        "end": end,
        "n_frames": n,
        "mean_positive": _mean([float(row["positive"]) for row in embedded]),
        "median_positive": _median([float(row["positive"]) for row in embedded]),
        "mean_negative": _mean([float(row["negative"]) for row in embedded]),
        "median_negative": _median([float(row["negative"]) for row in embedded]),
        "mean_uncertain": _mean([float(row["uncertain"]) for row in embedded]),
        "median_uncertain": _median([float(row["uncertain"]) for row in embedded]),
        "relevant_ratio": (relevant / n) if n else 0.0,
        "irrelevant_ratio": (irrelevant / n) if n else 0.0,
        "motion": _mean([float(frame.get("motion") or 0.0) for frame in frames]),
        "black_ratio": (sum(1 for frame in frames if frame.get("black")) / n) if n else 0.0,
        "decode_fail_ratio": (sum(1 for frame in frames if not frame.get("decode_ok", False)) / n) if n else 1.0,
        "blurry_ratio": (sum(1 for frame in frames if frame.get("blurry")) / n) if n else 0.0,
        "blur_window_ratio": float(quality["blur_window_ratio"]),
        "groups": _group_means(embedded),
        "view": view,
    }
    label, reason, confidence = window_label(summary, temporal)
    summary["label"] = label
    summary["reason"] = reason
    summary["confidence"] = confidence
    return summary


def build_windows(duration: float, frames: list[dict], view: str, temporal: dict, quality: dict) -> list[dict]:
    bounds = window_bounds(duration, float(temporal["window_s"]))
    windows = []
    for index, (start, end) in enumerate(bounds):
        chosen = _frames_in(frames, start, end, last=index == len(bounds) - 1)
        windows.append(summarize_window(start, end, chosen, view, temporal, quality))
    return windows


def _smooth_key(windows: list[dict], key: str, width: int) -> list[float]:
    if width <= 1:
        return [float(row[key]) for row in windows]
    half = width // 2
    values = [float(row[key]) for row in windows]
    smoothed = []
    for index in range(len(values)):
        chunk = values[max(0, index - half) : index + half + 1]
        smoothed.append(_median(chunk))
    return smoothed


def smooth_windows(windows: list[dict], temporal: dict, quality: dict) -> list[dict]:
    width = int(temporal.get("smooth", 1))
    if width <= 1 or len(windows) == 0:
        return windows
    smoothed = {key: _smooth_key(windows, key, width) for key in ("mean_positive", "mean_negative", "mean_uncertain")}
    out = []
    for index, row in enumerate(windows):
        copied = dict(row)
        for key in smoothed:
            copied[key] = smoothed[key][index]
        label, reason, confidence = window_label(copied, temporal)
        copied["label"] = label
        copied["reason"] = reason
        copied["confidence"] = confidence
        copied["blur_window_ratio"] = float(quality["blur_window_ratio"])
        out.append(copied)
    return out


def _removal_streak_member(row: dict, low_confidence_margin: float) -> bool:
    if row["label"] == "irrelevant":
        return True
    if row["label"] != "uncertain":
        return False
    if float(row["confidence"]) >= low_confidence_margin:
        return False
    return float(row["mean_negative"]) >= float(row["mean_positive"])


def apply_hysteresis(windows: list[dict], temporal: dict) -> list[dict]:
    """短于连续门槛的无关窗口改回不确定。低置信且偏向负类只用来把相邻无关连起来，本身不删。"""
    need = int(temporal["min_consecutive_irrelevant"])
    low = float(temporal["low_confidence_margin"])
    out = [dict(row) for row in windows]
    index = 0
    n = len(out)
    while index < n:
        if not _removal_streak_member(out[index], low):
            index += 1
            continue
        end = index
        while end < n and _removal_streak_member(out[end], low):
            end += 1
        if end - index < need:
            for cursor in range(index, end):
                if out[cursor]["label"] == "irrelevant":
                    out[cursor]["label"] = "uncertain"
                    out[cursor]["reason"] = "isolated_irrelevant"
        index = end
    return out


def gate_short_blur(windows: list[dict], temporal: dict) -> list[dict]:
    need = int(temporal["min_consecutive_irrelevant"])
    out = [dict(row) for row in windows]
    index = 0
    n = len(out)
    while index < n:
        if not (out[index]["label"] == "low_quality" and out[index]["reason"] == "blur"):
            index += 1
            continue
        end = index
        while end < n and out[end]["label"] == "low_quality" and out[end]["reason"] == "blur":
            end += 1
        if end - index < need:
            for cursor in range(index, end):
                out[cursor]["label"] = "uncertain"
                out[cursor]["reason"] = "short_blur"
        index = end
    return out


def _majority_reason(rows: list[dict]) -> str:
    weights: dict[str, float] = {}
    for row in rows:
        span = float(row["end"]) - float(row["start"])
        weights[row["reason"]] = weights.get(row["reason"], 0.0) + span
    return max(weights, key=weights.get)


def _segment(rows: list[dict], idle_motion_max: float) -> dict:
    reason = _majority_reason(rows)
    label = rows[0]["label"]
    motion = _mean([float(row["motion"]) for row in rows])
    if label == "irrelevant" and reason == "empty_workstation" and motion <= idle_motion_max:
        reason = "long_idle"
    confidence = _mean([float(row["confidence"]) for row in rows])
    return {
        "start": float(rows[0]["start"]),
        "end": float(rows[-1]["end"]),
        "label": label,
        "reason": reason,
        "confidence": confidence,
        "motion": motion,
        "mean_positive": _mean([float(row["mean_positive"]) for row in rows]),
        "median_positive": _median([float(row["median_positive"]) for row in rows]),
        "mean_negative": _mean([float(row["mean_negative"]) for row in rows]),
        "median_negative": _median([float(row["median_negative"]) for row in rows]),
        "mean_uncertain": _mean([float(row["mean_uncertain"]) for row in rows]),
        "relevant_ratio": _mean([float(row["relevant_ratio"]) for row in rows]),
        "irrelevant_ratio": _mean([float(row["irrelevant_ratio"]) for row in rows]),
    }


def merge_windows(windows: list[dict], idle_motion_max: float) -> list[dict]:
    if not windows:
        return []
    groups = [[windows[0]]]
    for row in windows[1:]:
        if row["label"] == groups[-1][-1]["label"]:
            groups[-1].append(row)
        else:
            groups.append([row])
    return [_segment(group, idle_motion_max) for group in groups]


def peel_context(segments: list[dict], boundary_s: float, idle_motion_max: float) -> list[dict]:
    peeled = []
    for seg in segments:
        if seg["label"] not in ("irrelevant", "low_quality"):
            peeled.append(seg)
            continue
        length = float(seg["end"]) - float(seg["start"])
        if length <= 2.0 * boundary_s:
            kept = dict(seg)
            kept["label"] = "uncertain"
            kept["reason"] = "boundary_context"
            peeled.append(kept)
            continue
        left = dict(seg)
        left["end"] = float(seg["start"]) + boundary_s
        left["label"] = "uncertain"
        left["reason"] = "boundary_context"
        mid = dict(seg)
        mid["start"] = float(seg["start"]) + boundary_s
        mid["end"] = float(seg["end"]) - boundary_s
        right = dict(seg)
        right["start"] = float(seg["end"]) - boundary_s
        right["label"] = "uncertain"
        right["reason"] = "boundary_context"
        peeled.extend([left, mid, right])
    return merge_windows(peeled, idle_motion_max)


def _empty_timeline(duration: float, temporal: dict) -> tuple[list[dict], list[dict]]:
    segment = {
        "start": 0.0,
        "end": float(duration),
        "label": "low_quality",
        "reason": "decode_error",
        "confidence": 1.0,
        "motion": 0.0,
        "mean_positive": 0.0,
        "median_positive": 0.0,
        "mean_negative": 0.0,
        "median_negative": 0.0,
        "mean_uncertain": 0.0,
        "relevant_ratio": 0.0,
        "irrelevant_ratio": 0.0,
    }
    segments = peel_context([segment], float(temporal["boundary_context_s"]), float(temporal["idle_motion_max"]))
    return [], segments


def label_timeline(duration: float, frames: list[dict], view: str, temporal: dict, quality: dict, adjust=None) -> tuple[list[dict], list[dict]]:
    if not frames:
        return _empty_timeline(duration, temporal)
    windows = build_windows(duration, frames, view, temporal, quality)
    windows = smooth_windows(windows, temporal, quality)
    if adjust is not None:
        windows = adjust(windows)
    windows = apply_hysteresis(windows, temporal)
    windows = gate_short_blur(windows, temporal)
    segments = merge_windows(windows, float(temporal["idle_motion_max"]))
    segments = peel_context(segments, float(temporal["boundary_context_s"]), float(temporal["idle_motion_max"]))
    return windows, segments
