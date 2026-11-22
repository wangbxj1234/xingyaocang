"""连续没有手，并且工位几乎不动，才记为空档。"""

from __future__ import annotations


def bridge_gaps(samples: list[dict], bridge_s: float) -> list[dict]:
    """单独一次误检出的手不把空档切开。"""
    if bridge_s <= 0 or len(samples) < 3:
        return [dict(row) for row in samples]
    rows = [dict(row) for row in samples]
    index = 0
    while index < len(rows):
        if rows[index].get("hand_present_in_work_area"):
            start = index
            while index < len(rows) and rows[index].get("hand_present_in_work_area"):
                index += 1
            end = index
            span = float(rows[end - 1]["timestamp_s"]) - float(rows[start]["timestamp_s"])
            left = start > 0 and not rows[start - 1].get("hand_present_in_work_area")
            right = end < len(rows) and not rows[end].get("hand_present_in_work_area")
            if left and right and span < bridge_s:
                for row in rows[start:end]:
                    row["hand_present_in_work_area"] = False
                    row["bridged"] = True
            continue
        index += 1
    return rows


def idle_intervals(samples: list[dict], duration: float, cfg: dict) -> list[dict]:
    """返回削过边界的空档。raw_start 是第一帧确认没有手的时间。"""
    if not samples:
        return []
    interval = float(cfg.get("sample_interval_s", 5))
    confirm = float(cfg["absence_confirm_s"])
    motion_low = float(cfg["motion_low"])
    context = float(cfg["boundary_context_s"])
    bridged = bridge_gaps(samples, float(cfg.get("bridge_s", 10)))
    found = []
    index = 0
    while index < len(bridged):
        row = bridged[index]
        moving = float(row.get("motion_score") or 0.0) > motion_low
        if row.get("hand_present_in_work_area") or moving:
            index += 1
            continue
        start = index
        while index < len(bridged):
            current = bridged[index]
            if current.get("hand_present_in_work_area"):
                break
            if float(current.get("motion_score") or 0.0) > motion_low:
                break
            index += 1
        quiet = bridged[start:index]
        if len(quiet) < 2:
            continue
        raw_start = float(quiet[0]["timestamp_s"])
        raw_end = min(duration, float(quiet[-1]["timestamp_s"]) + interval)
        if raw_end - raw_start < confirm - 1e-6:
            continue
        peeled_start = raw_start + context
        peeled_end = raw_end - context
        if peeled_end - peeled_start <= 0:
            continue
        found.append(
            {
                "raw_start": raw_start,
                "raw_end": raw_end,
                "start": peeled_start,
                "end": peeled_end,
                "label": "idle_no_person",
                "reason": "no_hand_low_motion",
                "confidence": 1.0,
                "motion": sum(float(row.get("motion_score") or 0.0) for row in quiet) / len(quiet),
            }
        )
    return found


def transition_to_absent(times: list[float], present: list[bool]) -> float | None:
    """1 fps 上，手从有到无的第一帧。"""
    if len(times) != len(present):
        raise ValueError("times and present must align")
    previous = None
    for time, flag in zip(times, present):
        if previous is True and not flag:
            return float(time)
        previous = bool(flag)
    return None


def transition_to_present(times: list[float], present: list[bool]) -> float | None:
    """1 fps 上，手从无到有的第一帧。空档收到这一帧为止。"""
    if len(times) != len(present):
        raise ValueError("times and present must align")
    previous = None
    for time, flag in zip(times, present):
        if previous is False and flag:
            return float(time)
        previous = bool(flag)
    return None
