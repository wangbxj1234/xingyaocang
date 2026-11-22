"""从抽出的帧按帧号编码重叠块。只有抽帧时带了 --rot180 才 hflip,vflip。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from fuhe.chain_math import blocks, duration_us, frame_window, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir
    meta = json.loads((work / "extract_meta.json").read_text(encoding="utf-8"))
    span = duration_us(float(meta["duration_s"]))
    rows = blocks(span)
    written = []
    for index, (start, end) in enumerate(rows):
        start_frame, count = frame_window(start, end)
        if count <= 0:
            raise SystemExit(f"block {index} has no frames")
        dest = work / f"block_{index:02d}.mp4"
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-framerate", "15",
            "-start_number", str(start_frame),
            "-i", str(work / "frames" / "%06d.jpg"),
            "-frames:v", str(count),
        ]
        if meta.get("rotated_180"):
            cmd += ["-vf", "hflip,vflip"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest)]
        subprocess.check_call(cmd)
        written.append({"start_s": seconds(start), "end_s": seconds(end), "path": dest.name, "frames": count})
        print(f"BLOCK_MP4 {index} frames {start_frame}+{count}", flush=True)
    (work / "blocks.json").write_text(json.dumps(written, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
