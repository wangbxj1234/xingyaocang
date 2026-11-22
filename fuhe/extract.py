"""抽头顶相机，重采样到 15 fps。

iter_frames 在 episode_rot180 为真时已经把画面转了 180°。这里保存的就是转完的帧，
后面编码不再转，避免和 360 秒那次脚本一样转两遍。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    args = parser.parse_args()

    wbx1 = os.environ.get("WBX1", "/mnt/cpfs/wbx/code/wbx1")
    sys.path.insert(0, wbx1)
    import numpy as np
    from PIL import Image
    from extract.mount_orient import episode_rot180
    from extract.qrdf_io import iter_frames
    from extract.qrdf_topics import TopicMap

    meta = load_metadata(args.metadata)
    duration_s = float(meta["timing"]["duration_s"])
    epoch_ns = int(meta["timing"]["start_timestamp_ns"])
    spans = keep_spans(duration_s)
    mount = topic_prefix(meta)
    topics = TopicMap(
        rgb=f"/camera/{mount}/rgb",
        depth=f"/camera/{mount}/depth",
        rgb_info=f"/camera/{mount}/camera_info",
        depth_info=f"/camera/{mount}/depth_camera_info",
        camera_pose=f"/observation/pose/{mount}_camera",
    )
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
    rotated = bool(episode_rot180(args.mcap))
    print("ROT180", int(rotated), "mount", mount, "frames", n_frames, flush=True)

    prev = None
    slot = 0
    seen = 0

    def take(rgb):
        image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        image.save(frames / f"{slot:06d}.jpg", quality=95)

    for frame in iter_frames(
        args.mcap,
        topics=topics,
        start_s=start_s,
        end_s=end_s,
        epoch_ns=epoch_ns,
    ):
        t = float(frame.timestamp_s) - start_s
        if t < 0:
            continue
        if t > end_s - start_s:
            break
        seen += 1
        if prev is not None:
            while slot < n_frames:
                target = slot / FPS
                if abs(target - prev[0]) <= abs(target - t):
                    take(prev[1])
                    slot += 1
                else:
                    break
        prev = (t, frame.rgb)
        if seen % 300 == 0:
            print(f"src {seen} slot {slot} t {t:.2f}", flush=True)
    if prev is None:
        raise SystemExit("no frames in the kept span")
    while slot < n_frames:
        take(prev[1])
        slot += 1
    out = {
        "rotated_180": rotated,
        "source_frames": seen,
        "slots": slot,
        "fps": FPS,
        "duration_s": end_s - start_s,
        "span_s": [start_s, end_s],
        "epoch_ns": epoch_ns,
        "source": meta["episode_id"],
        "mount": mount,
    }
    (work / "extract_meta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("EXTRACT_DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
