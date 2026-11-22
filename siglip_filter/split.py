"""按整条 episode 或采集日划分。同一条录像的窗口不会拆到两边。"""

from __future__ import annotations

import numpy as np


def split_units(units: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    ordered = sorted(set(units))
    if len(ordered) < 2:
        return set(ordered), set()
    rng = np.random.RandomState(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * float(val_fraction)))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def split_rows(rows: list[dict], mode: str, val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if mode not in ("episode", "date"):
        raise ValueError(f"split mode must be episode or date, got {mode}")
    episode_unit: dict[str, str] = {}
    for row in rows:
        episode_id = row["episode_id"]
        if mode == "date":
            unit = str(row.get("date") or episode_id)
        else:
            unit = episode_id
        episode_unit.setdefault(episode_id, unit)
    train_units, val_units = split_units(list(episode_unit.values()), val_fraction, seed)
    train, val = [], []
    for row in rows:
        unit = episode_unit[row["episode_id"]]
        if unit in val_units:
            val.append(row)
        else:
            train.append(row)
    return train, val
