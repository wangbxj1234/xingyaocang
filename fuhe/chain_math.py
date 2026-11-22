"""收链。和时间有关的常数与 360 秒那次相同，片长由调用方传入。"""

from __future__ import annotations

import json

TICK_US = 50_000
CORE_US = 12_000_000
BLOCK_US = 120_000_000
STRIDE_US = 96_000_000
MERGE_US = 500_000
FPS = 15


def seconds(tick: int) -> float:
    return tick / 1_000_000


def duration_us(duration_s: float) -> int:
    return int(round(duration_s * 1_000_000))


def blocks(duration: int) -> list[tuple[int, int]]:
    rows = []
    start = 0
    while start < duration:
        end = min(start + BLOCK_US, duration)
        rows.append((start, end))
        if end == duration:
            break
        start += STRIDE_US
    return rows


def managed(index: int, count: int, start: int, duration: int) -> tuple[int, int]:
    lo = 0 if index == 0 else start + CORE_US
    hi = duration if index == count - 1 else start + 9 * CORE_US
    return lo, hi


def seam_distance(tick: int) -> int:
    seam = round(tick / CORE_US) * CORE_US
    return abs(tick - seam)


def parse_windows(calls, block_start: int, lo: int, hi: int):
    cuts = []
    cover = []
    kept_windows = []
    for call in calls:
        window = call["window"]
        scope = window["scope_id"]
        if "fine-window" not in scope or "fine-retry" in scope:
            continue
        origin = block_start + int(window["requested_view_start_us"])
        core_lo = origin + int(window["core_start_tick"]) * TICK_US
        core_hi = origin + int(window["core_end_tick"]) * TICK_US
        payload = json.loads(call["output"])
        rows = []
        for row in payload["segments"]:
            global_s = origin + int(row["s"]) * TICK_US
            global_e = origin + int(row["e"]) * TICK_US
            cover.append((global_s, global_e))
            rows.append({"s": global_s, "e": global_e, "lo": row["lo"], "ro": row["ro"]})
            if not row["lo"] and core_lo <= global_s < core_hi and lo <= global_s < hi:
                cuts.append(global_s)
            if not row["ro"] and core_lo <= global_e < core_hi and lo <= global_e < hi:
                cuts.append(global_e)
        kept_windows.append({
            "scope_id": scope,
            "origin_us": origin,
            "core_us": [core_lo, core_hi],
            "managed_us": [lo, hi],
            "rows": rows,
        })
    return cuts, cover, kept_windows


def chain(cuts, duration: int) -> list[int]:
    points = sorted(set(cuts) | {0, duration})
    clusters = []
    for tick in points:
        if not clusters or tick - clusters[-1][-1] >= MERGE_US:
            clusters.append([tick])
        else:
            clusters[-1].append(tick)
    kept = []
    for group in clusters:
        anchors = [tick for tick in group if tick in (0, duration)]
        if anchors:
            kept.extend(anchors)
            continue
        kept.append(max(group, key=lambda tick: (seam_distance(tick), -tick)))
    kept = sorted(set(kept))
    if kept[0] != 0 or kept[-1] != duration:
        raise RuntimeError(f"chain does not span the clip: {kept[0]} {kept[-1]} duration {duration}")
    return kept


def covered(start: int, end: int, cover) -> bool:
    return any(row_s < end and row_e > start for row_s, row_e in cover)


def choose_block(start: int, end: int, block_rows, duration: int):
    need_lo = max(0, start - 2_000_000)
    need_hi = min(duration, end + 2_000_000)
    for index, (block_start, block_end) in enumerate(block_rows):
        if block_start <= need_lo and need_hi <= block_end:
            return index
    return None


def frame_window(block_start: int, block_end: int) -> tuple[int, int]:
    start_frame = int(round(seconds(block_start) * FPS))
    count = int(round(seconds(block_end - block_start) * FPS))
    return start_frame, count
