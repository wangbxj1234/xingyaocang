"""Cut invalid spans from one episode and splice the kept camera frames.

Originals stay where they are. The spliced file is a new mp4.
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


def write_kept_mp4(episode: dict, ranges: list[tuple[float, float]], out_path: Path) -> dict:
    from fuhe.mcap_rgb import iter_jpeg

    fps = float(episode.get("fps") or 0)
    topic = episode.get("topic") or f"/camera/{episode.get('mount') or 'head'}/rgb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    pending: list[bytes] = []
    times: list[float] = []
    frames = 0
    index = 0
    code = 0
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
                continue
            if proc is None and fps <= 0:
                pending.append(blob)
                times.append(float(stamp))
                if len(times) >= 8:
                    fps = _infer_fps(times)
                    proc = _ffmpeg(fps, out_path)
                    assert proc.stdin is not None
                    for item in pending:
                        proc.stdin.write(item)
                    frames += len(pending)
                    pending = []
                continue
            if proc is None:
                proc = _ffmpeg(fps, out_path)
            assert proc.stdin is not None
            proc.stdin.write(blob)
            frames += 1
    finally:
        if proc is None and pending:
            fps = _infer_fps(times)
            proc = _ffmpeg(fps, out_path)
            assert proc.stdin is not None
            for item in pending:
                proc.stdin.write(item)
            frames += len(pending)
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            code = proc.wait()
    if frames == 0:
        out_path.unlink(missing_ok=True)
        return {"frames": 0, "fps": fps, "duration_s": 0.0}
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code} for {episode['episode_id']}")
    return {"frames": frames, "fps": fps, "duration_s": frames / fps}


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
    return object_size(f"{DEST_ROOT}/{episode_id}/done.json") is not None


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
        ranges = kept_ranges(intervals, float(episode["duration"]))
        mapped = []
        cursor = 0.0
        for start, end in ranges:
            span = end - start
            mapped.append(
                {
                    "src_start": start,
                    "src_end": end,
                    "dst_start": cursor,
                    "dst_end": cursor + span,
                }
            )
            cursor += span
        mp4 = work / "kept.mp4"
        video = {"frames": 0, "fps": float(episode.get("fps") or 0), "duration_s": 0.0}
        if ranges:
            video = write_kept_mp4(episode, ranges, mp4)
        payload = {
            "episode_id": episode["episode_id"],
            "task": episode.get("task"),
            "source": prefix + "data.mcap",
            "topic": episode.get("topic"),
            "fps": video["fps"],
            "src_duration_s": float(episode["duration"]),
            "kept_duration_s": video["duration_s"],
            "frames": video["frames"],
            "kept": mapped,
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
        if video["frames"]:
            upload(mp4, folder + "/kept.mp4")
        done = work / "done.json"
        done.write_text(json.dumps({"episode_id": episode["episode_id"], "frames": video["frames"]}), encoding="utf-8")
        upload(done, folder + "/done.json")
        print(
            "CUT done %s kept %.0fs invalid %d wall %.0fs"
            % (episode["episode_id"], video["duration_s"], len(intervals), payload["wall_s"]),
            flush=True,
        )
        return payload
    finally:
        local_mcap.unlink(missing_ok=True)
        (work / "kept.mp4").unlink(missing_ok=True)


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
