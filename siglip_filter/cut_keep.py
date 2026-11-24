"""Cut invalid spans and write each kept piece as its own segment.

Originals stay where they are. Kept pieces are not concatenated.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from siglip_filter.idle import idle_intervals

OSS = "/dev/shm/bin/ossutil"
READ_CFG = "/dev/shm/ossutilconfig"
WRITE_CFG = "/dev/shm/ossutilconfig-research"
DEST_ROOT = "oss://quic-research/wbx/data/fuhe_kept"


ORIENT_WINDOW_S = 180.0
ORIENT_EDGE = 0.15
ORIENT_MIN_TOP = 5


def needs_rot180(samples: list[dict]) -> bool:
    """开头 3 分钟里，手框贴上沿的帧多于贴下沿，并且上沿至少 5 帧，整段转 180°。"""
    top = 0
    bottom = 0
    for row in samples:
        if float(row.get("timestamp_s") or 0.0) >= ORIENT_WINDOW_S:
            continue
        if not row.get("hand_present_in_work_area"):
            continue
        bbox = row.get("hand_bbox")
        height = float(row.get("image_height") or 0.0)
        if not bbox or height <= 0:
            continue
        y0 = float(bbox[1]) / height
        y1 = float(bbox[3]) / height
        at_top = y0 <= ORIENT_EDGE
        at_bottom = y1 >= 1.0 - ORIENT_EDGE
        if at_top and not at_bottom:
            top += 1
        elif at_bottom and not at_top:
            bottom += 1
    return top >= ORIENT_MIN_TOP and top > bottom


def kept_ranges(intervals: list[dict], duration: float) -> list[tuple[float, float]]:
    """Time still in the episode after the invalid spans are removed."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    cursor = 0.0
    kept = []
    for row in sorted(intervals, key=lambda item: float(item["start"])):
        start = max(0.0, float(row["start"]))
        end = min(duration, float(row["end"]))
        if end <= start:
            continue
        if start > cursor:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        kept.append((cursor, duration))
    return kept


def _in_kept(stamp: float, ranges: list[tuple[float, float]], index: int) -> tuple[bool, int]:
    while index < len(ranges) and stamp >= ranges[index][1]:
        index += 1
    if index < len(ranges) and ranges[index][0] <= stamp < ranges[index][1]:
        return True, index
    return False, index


def detect_intervals(cfg, episode: dict) -> tuple[list[dict], list[dict]]:
    presence = dict(cfg.presence)
    if not presence.get("enabled"):
        raise ValueError("presence.enabled is false")
    polygon = presence["work_area"]
    interval = float(presence.get("sample_interval_s", 1))
    topic = episode.get("topic") or f"/camera/{episode.get('mount') or 'head'}/rgb"
    epoch = int(episode["epoch_ns"])
    duration = float(episode["duration"])
    from fuhe.mcap_rgb import iter_rgb
    from siglip_filter.hands import HandDetector
    from siglip_filter.motion_roi import MotionRoi

    detector = HandDetector(
        str(presence["model_path"]),
        float(presence.get("min_confidence", 0.5)),
        int(presence.get("max_hands", 2)),
    )
    motion = MotionRoi(polygon)
    samples = []
    next_stamp = 0.0
    try:
        for stamp, image in iter_rgb(
            Path(episode["video_path"]),
            topic,
            epoch_ns=epoch,
            start_s=0.0,
            end_s=duration,
        ):
            if stamp + 1e-6 < next_stamp:
                continue
            if stamp > next_stamp + interval * 0.45:
                next_stamp = stamp
            observed = detector.detect(image, polygon, float(presence.get("min_overlap", 0.2)))
            observed.update(motion.measure(image))
            observed["timestamp_s"] = float(stamp)
            observed["image_height"] = int(image.size[1])
            samples.append(observed)
            if len(samples) % 60 == 0:
                print("CUT sample %.0fs %s" % (stamp, episode["episode_id"]), flush=True)
            next_stamp += interval
    finally:
        detector.close()
    idle_cfg = {
        "sample_interval_s": interval,
        "absence_confirm_s": float(presence["absence_confirm_s"]),
        "motion_low": float(presence["motion_low"]),
        "boundary_context_s": float(presence["boundary_context_s"]),
        "bridge_s": float(presence.get("bridge_s", 1)),
    }
    intervals = idle_intervals(samples, duration, idle_cfg)
    return samples, intervals


def _infer_fps(times: list[float]) -> float:
    deltas = sorted(b - a for a, b in zip(times, times[1:]) if b - a > 1e-4)
    if not deltas:
        return 15.0
    guess = 1.0 / deltas[len(deltas) // 2]
    for candidate in (10.0, 15.0, 30.0):
        if abs(guess - candidate) < 1.5:
            return candidate
    return float(max(1, round(guess)))


def rotate_jpeg_180(blob: bytes) -> bytes:
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(blob))
    turned = image.transpose(Image.ROTATE_180)
    out = io.BytesIO()
    turned.save(out, format="JPEG", quality=95)
    return out.getvalue()


def _ffmpeg(fps: float, out_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            str(fps),
            "-vcodec",
            "mjpeg",
            "-i",
            "-",
            "-c:v",
            "copy",
            "-an",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )


def segment_record(index: int, start: float, end: float, frames: int, fps: float) -> dict:
    """One kept piece. There is no timeline that joins it to the next piece."""
    return {
        "seg": index,
        "file": "seg_%03d.mp4" % index,
        "src_start": start,
        "src_end": end,
        "frames": frames,
        "duration_s": frames / fps if fps else 0.0,
    }


def _start_ffmpeg(fps: float, path: Path, pending: list[bytes]) -> tuple[subprocess.Popen, int]:
    proc = _ffmpeg(fps, path)
    assert proc.stdin is not None
    for blob in pending:
        proc.stdin.write(blob)
    return proc, len(pending)


def _finish_segment(writer: dict, fps: float) -> tuple[dict | None, float]:
    proc = writer["proc"]
    if proc is None:
        if not writer["pending"]:
            return None, fps
        if fps <= 0:
            fps = _infer_fps(writer["times"])
        proc, writer["frames"] = _start_ffmpeg(fps, writer["path"], writer["pending"])
        writer["proc"] = proc
        writer["pending"] = []
    assert proc.stdin is not None
    proc.stdin.close()
    code = proc.wait()
    if writer["frames"] == 0:
        writer["path"].unlink(missing_ok=True)
        return None, fps
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code} for {writer['path'].name}")
    return segment_record(writer["seg"], writer["src_start"], writer["src_end"], writer["frames"], fps), fps


def write_segments(
    episode: dict,
    ranges: list[tuple[float, float]],
    out_dir: Path,
    rot180: bool = False,
) -> tuple[list[dict], float]:
    """Write one mp4 per kept range. Ranges are not concatenated."""
    from fuhe.mcap_rgb import iter_jpeg

    if not ranges:
        return [], float(episode.get("fps") or 0)
    fps = float(episode.get("fps") or 0)
    topic = episode.get("topic") or f"/camera/{episode.get('mount') or 'head'}/rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    segments: list[dict] = []
    writer: dict | None = None
    index = 0

    def open_writer(range_index: int) -> dict:
        start, end = ranges[range_index]
        seg = len(segments)
        path = out_dir / ("seg_%03d.mp4" % seg)
        proc = None
        if fps > 0:
            proc = _ffmpeg(fps, path)
        return {
            "seg": seg,
            "range_index": range_index,
            "path": path,
            "proc": proc,
            "pending": [],
            "times": [],
            "frames": 0,
            "src_start": start,
            "src_end": end,
        }

    def push(blob: bytes, stamp: float) -> None:
        nonlocal fps, writer
        assert writer is not None
        if rot180:
            blob = rotate_jpeg_180(blob)
        if fps <= 0:
            writer["pending"].append(blob)
            writer["times"].append(stamp)
            if len(writer["times"]) >= 8:
                fps = _infer_fps(writer["times"])
                writer["proc"], writer["frames"] = _start_ffmpeg(fps, writer["path"], writer["pending"])
                writer["pending"] = []
            return
        if writer["proc"] is None:
            writer["proc"], _ = _start_ffmpeg(fps, writer["path"], [])
        assert writer["proc"].stdin is not None
        writer["proc"].stdin.write(blob)
        writer["frames"] += 1

    def close_writer() -> None:
        nonlocal fps, writer
        if writer is None:
            return
        current = writer
        writer = None
        row, fps = _finish_segment(current, fps)
        if row is not None:
            segments.append(row)

    try:
        for stamp, blob in iter_jpeg(
            Path(episode["video_path"]),
            topic,
            epoch_ns=int(episode["epoch_ns"]),
            start_s=0.0,
            end_s=float(episode["duration"]),
        ):
            inside, index = _in_kept(stamp, ranges, index)
            if not inside:
                close_writer()
                continue
            if writer is None or writer["range_index"] != index:
                close_writer()
                writer = open_writer(index)
            push(blob, float(stamp))
    finally:
        close_writer()
    return segments, fps


def _oss(cfg: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([OSS, "-c", cfg, *args], check=check, capture_output=True, text=True)


def object_size(url: str) -> int | None:
    result = _oss(WRITE_CFG, ["stat", url], check=False)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "Content-Length" in line:
            return int(line.split(":")[-1].strip())
    return None


def upload(local: Path, url: str) -> None:
    _oss(WRITE_CFG, ["cp", "-f", str(local), url])
    remote = object_size(url)
    if remote != local.stat().st_size:
        raise RuntimeError(f"upload size mismatch {url} local {local.stat().st_size} remote {remote}")


def already_done(episode_id: str) -> bool:
    url = f"{DEST_ROOT}/{episode_id}/done.json"
    if object_size(url) is None:
        return False
    result = _oss(WRITE_CFG, ["cat", url], check=False)
    return result.returncode == 0 and '"layout": "segments"' in result.stdout


def dest_for(episode_id: str) -> str:
    return f"{DEST_ROOT}/{episode_id}"


def process_episode(cfg, episode: dict, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    local_mcap = work / "data.mcap"
    prefix = episode["oss_prefix"]
    if not prefix.endswith("/"):
        prefix += "/"
    started = time.time()
    _oss(READ_CFG, ["cp", "-f", prefix + "data.mcap", str(local_mcap)])
    episode = dict(episode)
    episode["video_path"] = str(local_mcap)
    try:
        _samples, intervals = detect_intervals(cfg, episode)
        rot180 = needs_rot180(_samples)
        ranges = kept_ranges(intervals, float(episode["duration"]))
        segments, fps = write_segments(episode, ranges, work, rot180=rot180)
        kept_s = sum(float(row["duration_s"]) for row in segments)
        frames = sum(int(row["frames"]) for row in segments)
        payload = {
            "episode_id": episode["episode_id"],
            "task": episode.get("task"),
            "source": prefix + "data.mcap",
            "topic": episode.get("topic"),
            "fps": fps,
            "src_duration_s": float(episode["duration"]),
            "kept_duration_s": kept_s,
            "frames": frames,
            "rotated_180": rot180,
            "segments": segments,
            "invalid": [
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "label": row.get("label"),
                    "reason": row.get("reason"),
                }
                for row in intervals
            ],
            "wall_s": round(time.time() - started, 1),
        }
        spans = work / "spans.json"
        spans.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        folder = dest_for(episode["episode_id"])
        upload(spans, folder + "/spans.json")
        for row in segments:
            upload(work / row["file"], folder + "/" + row["file"])
        done = work / "done.json"
        done.write_text(
            json.dumps(
                {
                    "episode_id": episode["episode_id"],
                    "layout": "segments",
                    "segments": len(segments),
                    "frames": frames,
                    "rotated_180": rot180,
                }
            ),
            encoding="utf-8",
        )
        upload(done, folder + "/done.json")
        print(
            "CUT done %s segs %d kept %.0fs invalid %d rot180 %d wall %.0fs"
            % (episode["episode_id"], len(segments), kept_s, len(intervals), int(rot180), payload["wall_s"]),
            flush=True,
        )
        return payload
    finally:
        local_mcap.unlink(missing_ok=True)
        for path in work.glob("seg_*.mp4"):
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from siglip_filter.config import FilterConfig

    parser = argparse.ArgumentParser(prog="siglip_filter.cut_keep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    episodes = json.loads(args.episodes.read_text(encoding="utf-8"))
    run_shard(FilterConfig.load(args.config), episodes, args.shard, args.shards, args.work)


def run_shard(cfg, episodes: list[dict], shard: int, shards: int, work: Path) -> None:
    mine = [row for index, row in enumerate(episodes) if index % shards == shard]
    print(f"CUT shard {shard}/{shards} episodes {len(mine)}", flush=True)
    for episode in mine:
        if already_done(episode["episode_id"]):
            print(f"CUT skip {episode['episode_id']}", flush=True)
            continue
        try:
            process_episode(cfg, episode, work)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"CUT fail {episode['episode_id']} {message}", flush=True)
            fail = work / "error.json"
            fail.write_text(
                json.dumps(
                    {"episode_id": episode["episode_id"], "source": episode.get("oss_prefix"), "error": message},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                upload(fail, dest_for(episode["episode_id"]) + "/error.json")
            except Exception as upload_error:
                print(f"CUT fail-upload {episode['episode_id']} {upload_error}", flush=True)
                raise


if __name__ == "__main__":
    main()
