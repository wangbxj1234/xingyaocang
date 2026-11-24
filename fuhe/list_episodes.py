"""列出复核包装发货的完整 episode，给 siglip_filter.cut_keep 用。"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OSS = ["/dev/shm/bin/ossutil", "-c", "/dev/shm/ossutilconfig"]
TASK_NAMES = {"复核包装发货", "复核包装发货1", "复核包装发货2", "复核包装发货3", "复合包装发货"}
ROOT = "oss://quic-collect-data/prod/raw/v2/sources/"


def ls(prefix: str) -> list[str]:
    out = subprocess.check_output(OSS + ["ls", prefix, "-d"], text=True, errors="replace")
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip().startswith("oss://") and line.strip().endswith("/")
    ]


def load_meta(prefix: str) -> dict | None:
    raw = subprocess.check_output(OSS + ["cat", prefix + "metadata.json"], text=True, errors="replace")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0:
        return None
    data = json.loads(raw[start : end + 1])
    data["_prefix"] = prefix
    return data


def topic_of(data: dict, mount: str | None) -> str:
    cameras = ((data.get("sensors") or {}).get("cameras")) or []
    if cameras and cameras[0].get("topic"):
        return cameras[0]["topic"]
    return "/camera/%s/rgb" % (mount or "head")


def row_from_meta(data: dict) -> dict | None:
    if data.get("recording_status") != "complete":
        return None
    name = ((data.get("source_group") or {}).get("name")) or ""
    if name not in TASK_NAMES:
        return None
    timing = data.get("timing") or {}
    cameras = ((data.get("sensors") or {}).get("cameras")) or [{}]
    camera = cameras[0] or {}
    devices = data.get("devices") or [{}]
    mount = (devices[0] or {}).get("mount") if devices else None
    return {
        "episode_id": data.get("episode_id"),
        "task": name,
        "oss_prefix": data.get("_prefix"),
        "duration": float(timing.get("duration_s") or 0),
        "epoch_ns": int(timing.get("start_timestamp_ns") or 0),
        "topic": topic_of(data, mount),
        "fps": float(camera.get("fps") or 0),
        "width": int(camera.get("width") or 0),
        "height": int(camera.get("height") or 0),
        "mount": mount,
    }


def collect() -> tuple[list[dict], int]:
    dates = [prefix for prefix in ls(ROOT) if "date=2026-" in prefix]
    print("dates", len(dates), flush=True)
    hours: list[str] = []
    for date in dates:
        hours.extend(ls(date))
    print("hours", len(hours), flush=True)
    episodes: list[str] = []
    for hour in hours:
        episodes.extend(ls(hour))
    print("dirs", len(episodes), flush=True)
    rows: list[dict] = []
    bad = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for data in pool.map(load_meta, episodes):
            if not data:
                bad += 1
                continue
            row = row_from_meta(data)
            if row is not None:
                rows.append(row)
    return rows, bad


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="fuhe.list_episodes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, bad = collect()
    print(
        "kept",
        len(rows),
        "bad",
        bad,
        "hours",
        round(sum(row["duration"] for row in rows) / 3600, 1),
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
