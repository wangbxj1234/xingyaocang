"""按 cycles.jsonl 把一单从 seg mp4 切出来。中间空档超过 1 秒的不切。"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OSS = ["/dev/shm/bin/ossutil", "-c", "/dev/shm/ossutilconfig-research"]


def probe(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def cut(src: Path, dest: Path, start: float, end: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
            str(dest),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="fuhe.cut_pack_cycles")
    parser.add_argument("--work", type=Path, default=Path("/dev/shm/fuhe_hf_cycles"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    out = args.work
    src_dir = out / "src"
    rows = [json.loads(line) for line in (out / "cycles.jsonl").read_text().splitlines()]
    keep = [row for row in rows if row["gap_s"] <= 1]
    keep.sort(key=lambda row: (row["episode_id"], int(row["seg"]), row["mp4_start_s"]))
    groups = defaultdict(list)
    for index, row in enumerate(keep):
        row["id"] = f"{index:05d}"
        row["video"] = f"videos/{index:05d}.mp4"
        groups[(row["folder"], row["file"])].append(row)
    src_dir.mkdir(parents=True, exist_ok=True)
    done = 0

    def one(job):
        (folder, name), items = job
        local = src_dir / f"{items[0]['episode_id']}_{name}"
        subprocess.run(OSS + ["cp", "-f", folder + "/" + name, str(local)], check=True, capture_output=True)
        try:
            for row in items:
                dest = out / row["video"]
                if not (dest.exists() and dest.stat().st_size > 0):
                    cut(local, dest, row["mp4_start_s"], row["mp4_end_s"])
                got = probe(dest)
                if abs(got - row["duration_s"]) > 0.75:
                    raise RuntimeError(f"{row['id']} duration {got:.2f} != {row['duration_s']:.2f}")
        finally:
            local.unlink(missing_ok=True)
        return name, len(items)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, job) for job in groups.items()]
        for future in as_completed(futures):
            name, count = future.result()
            done += count
            print("cut", done, "/", len(keep), name, count, flush=True)
    with (out / "cycles.jsonl").open("w") as handle:
        for row in keep:
            handle.write(json.dumps({
                "id": row["id"],
                "video": row["video"],
                "episode_id": row["episode_id"],
                "seg": row["seg"],
                "duration_s": row["duration_s"],
                "captions": row["captions"],
            }, ensure_ascii=False) + "\n")
    card = Path(__file__).with_name("pack_cycles_card.md")
    if card.is_file():
        (out / "README.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
    print("CUT_DONE", len(keep), flush=True)


if __name__ == "__main__":
    main()
