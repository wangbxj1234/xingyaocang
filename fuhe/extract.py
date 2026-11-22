"""抽一路相机的 jpeg，按最近邻重采样到 15 fps。不转画面。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuhe.mcap_rgb import iter_rgb
from fuhe.preprocess import keep_spans

FPS = 15


def load_metadata(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])


def topic_prefix(meta: dict) -> str:
    devices = meta.get("devices") or []
    mount = (devices[0].get("mount") if devices else None) or "head"
    if mount not in ("head", "stand"):
        raise SystemExit(f"unsupported mount {mount}")
    return mount


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--rot180", action="store_true")
    args = parser.parse_args()

    meta = load_metadata(args.metadata)
    duration_s = float(meta["timing"]["duration_s"])
    epoch_ns = int(meta["timing"]["start_timestamp_ns"])
    spans = keep_spans(duration_s)
    mount = topic_prefix(meta)
    topic = f"/camera/{mount}/rgb"
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    (work / "spans.json").write_text(
        json.dumps(
            [{"start_s": start, "end_s": end} for start, end in spans],
            indent=2,
        ),
        encoding="utf-8",
    )
    if len(spans) != 1:
        raise SystemExit("preprocess still returns one span; extract does not split yet")
    start_s, end_s = spans[0]
    n_frames = int(round((end_s - start_s) * FPS))
    frames = work / "frames"
    frames.mkdir(exist_ok=True)
    print("ROT180", int(args.rot180), "mount", mount, "frames", n_frames, flush=True)

    prev = None
    slot = 0
    seen = 0

    def take(image):
        image.save(frames / f"{slot:06d}.jpg", quality=95)

    for t_abs, image in iter_rgb(
        args.mcap, topic, epoch_ns=epoch_ns, start_s=start_s, end_s=end_s,
    ):
        t = t_abs - start_s
        seen += 1
        if prev is not None:
            while slot < n_frames:
                target = slot / FPS
                if abs(target - prev[0]) <= abs(target - t):
                    take(prev[1])
                    slot += 1
                else:
                    break
        prev = (t, image)
        if seen % 300 == 0:
            print(f"src {seen} slot {slot} t {t:.2f}", flush=True)
    if prev is None:
        raise SystemExit("no frames in the kept span")
    while slot < n_frames:
        take(prev[1])
        slot += 1
    out = {
        "rotated_180": bool(args.rot180),
        "source_frames": seen,
        "slots": slot,
        "fps": FPS,
        "duration_s": end_s - start_s,
        "span_s": [start_s, end_s],
        "epoch_ns": epoch_ns,
        "source": meta["episode_id"],
        "mount": mount,
        "topic": topic,
    }
    (work / "extract_meta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("EXTRACT_DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
