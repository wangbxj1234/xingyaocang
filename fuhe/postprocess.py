"""把 Flora 链的时间映回原 episode。不改字幕，不补 gap。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    episode_id = official["source"]
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{episode_id}.json"
    dest.write_text(
        json.dumps({
            "episode_id": episode_id,
            "rotated_180": official["rotated_180"],
            "segments": rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("POST_DONE", dest, len(rows), flush=True)


if __name__ == "__main__":
    main()
