"""把 Flora 链的时间映回原 episode，再按规则删段。不改留下的字幕，不补 gap。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# 第一条规则：电脑、鼠标、手机。Flora 把电脑写成 keyboard / computer / laptop。
DEVICE_RULES = (
    ("computer", re.compile(r"computer|laptop|keyboard|电脑", re.IGNORECASE)),
    ("mouse", re.compile(r"(?<![A-Za-z])mouse(?![A-Za-z])|鼠标", re.IGNORECASE)),
    ("phone", re.compile(
        r"(?<![A-Za-z])phones?(?![A-Za-z])|cellphone|smartphone|mobile device|手机",
        re.IGNORECASE,
    )),
)


def map_segments(official: dict, span_start_s: float) -> list[dict]:
    rows = []
    for seg in official["segments"]:
        rows.append({
            "start_s": seg["start_s"] + span_start_s,
            "end_s": seg["end_s"] + span_start_s,
            "caption": seg.get("caption", ""),
            "status": seg.get("status", ""),
        })
    return rows


def device_hits(caption: str) -> list[str]:
    hits = []
    for name, pattern in DEVICE_RULES:
        if pattern.search(caption or ""):
            hits.append(name)
    return hits


def drop_device_segments(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    kept = []
    dropped = []
    for row in rows:
        hits = device_hits(row.get("caption") or "")
        if hits and row.get("status") != "gap":
            dropped.append({**row, "drop": hits})
        else:
            kept.append(row)
    return kept, dropped


def assert_covers(rows: list[dict], span_start_s: float, span_end_s: float) -> None:
    if not rows:
        raise SystemExit("empty chain")
    if abs(rows[0]["start_s"] - span_start_s) > 1e-3:
        raise SystemExit(f"chain starts at {rows[0]['start_s']}, span starts at {span_start_s}")
    if abs(rows[-1]["end_s"] - span_end_s) > 1e-3:
        raise SystemExit(f"chain ends at {rows[-1]['end_s']}, span ends at {span_end_s}")
    for prev, row in zip(rows, rows[1:]):
        if abs(prev["end_s"] - row["start_s"]) > 1e-3:
            raise SystemExit(f"chain breaks at {prev['end_s']} / {row['start_s']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir
    official = json.loads((work / "official.json").read_text(encoding="utf-8"))
    spans = json.loads((work / "spans.json").read_text(encoding="utf-8"))
    if len(spans) != 1:
        raise SystemExit("postprocess expects the single span preprocess writes today")
    span = spans[0]
    rows = map_segments(official, float(span["start_s"]))
    assert_covers(rows, float(span["start_s"]), float(span["end_s"]))
    kept, dropped = drop_device_segments(rows)
    episode_id = official["source"]
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{episode_id}.json"
    dest.write_text(
        json.dumps({
            "episode_id": episode_id,
            "rotated_180": official["rotated_180"],
            "segments": kept,
            "dropped": dropped,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("POST_DONE", dest, len(kept), "dropped", len(dropped), flush=True)


if __name__ == "__main__":
    main()
