"""筛掉与复核包装发货主要工作无关的片段。判定留白。"""

from __future__ import annotations


def keep_spans(duration_s: float) -> list[tuple[float, float]]:
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    return [(0.0, float(duration_s))]
