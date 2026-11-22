"""在已有的 SigLIP 分数上补人手和工位运动。不改原片，也不重训。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from siglip_filter.fusion import fuse_timeline
from siglip_filter.hands import HandDetector
from siglip_filter.idle import idle_intervals, transition_to_absent, transition_to_present
from siglip_filter.manifest import write_json, write_jsonl
from siglip_filter.motion_roi import MotionRoi


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _nearest(rows: list[dict], stamp: float, slack: float = 2.5) -> dict | None:
    best = None
    best_distance = slack
    for row in rows:
        distance = abs(float(row["timestamp_s"]) - stamp)
        if distance <= best_distance:
            best = row
            best_distance = distance
    return best


def _clock(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def _stamps(duration: float, interval: float) -> list[float]:
    stamps = []
    cursor = 0.0
    while cursor < duration - 1e-6:
        stamps.append(cursor)
        cursor += interval
    return stamps


def _grab(episode: dict, targets: list[float], slack: float = 0.45) -> dict:
    from fuhe.mcap_rgb import iter_rgb

    pending = sorted(targets)
    found = {}
    next_index = 0
    epoch = int(episode.get("epoch_ns", episode.get("start_timestamp_ns")))
    topic = episode.get("topic") or f"/camera/{episode.get('mount') or 'head'}/rgb"
    for stamp, image in iter_rgb(
        Path(episode["video_path"]),
        topic,
        epoch_ns=epoch,
        start_s=0.0,
        end_s=float(episode["duration"]),
    ):
        while next_index < len(pending) and stamp > pending[next_index] + slack:
            next_index += 1
        if next_index >= len(pending):
            break
        target = pending[next_index]
        if abs(stamp - target) <= slack and target not in found:
            found[target] = image.copy()
            next_index += 1
    return found


def _observe(image, detector: HandDetector, motion: MotionRoi, polygon: list, min_overlap: float) -> dict:
    hand = detector.detect(image, polygon, min_overlap)
    moved = motion.measure(image)
    hand.update(moved)
    return hand


def _refine(episode, coarse: list[dict], detector, polygon, min_overlap, radius: float, fps: float) -> list[dict]:
    if fps <= 0 or not coarse:
        return coarse
    step = 1.0 / fps
    wanted = []
    for row in coarse:
        for edge in (row["raw_start"], row["raw_end"]):
            cursor = max(0.0, edge - radius)
            stop = min(float(episode["duration"]), edge + radius)
            while cursor <= stop + 1e-9:
                wanted.append(round(cursor, 3))
                cursor += step
    images = _grab(episode, sorted(set(wanted)), slack=step * 0.6)
    refined = []
    for row in coarse:
        start_times = sorted(stamp for stamp in images if abs(stamp - row["raw_start"]) <= radius + 1e-6)
        end_times = sorted(stamp for stamp in images if abs(stamp - row["raw_end"]) <= radius + 1e-6)
        start_flags = []
        end_flags = []
        for stamp in start_times:
            seen = detector.detect(images[stamp], polygon, min_overlap)
            start_flags.append(bool(seen["hand_present_in_work_area"]))
        for stamp in end_times:
            seen = detector.detect(images[stamp], polygon, min_overlap)
            end_flags.append(bool(seen["hand_present_in_work_area"]))
        raw_start = transition_to_absent(start_times, start_flags) if start_times else None
        raw_end = transition_to_present(end_times, end_flags) if end_times else None
        updated = dict(row)
        if raw_start is not None:
            updated["raw_start"] = raw_start
        if raw_end is not None:
            updated["raw_end"] = raw_end
        refined.append(updated)
    return refined, images


def reapply_peel(intervals: list[dict], context_s: float, duration: float) -> list[dict]:
    peeled = []
    for row in intervals:
        start = float(row["raw_start"]) + context_s
        end = float(row["raw_end"]) - context_s
        if end - start <= 0 or end > duration + 1e-6:
            continue
        item = dict(row)
        item["start"] = start
        item["end"] = min(duration, end)
        peeled.append(item)
    return peeled


def attach_siglip(samples: list[dict], siglip_rows: list[dict], view: str, motion_low: float) -> list[dict]:
    attached = []
    for sample in samples:
        source = _nearest(siglip_rows, float(sample["timestamp_s"]))
        row = dict(sample)
        row["siglip_view"] = view
        row["motion_low"] = motion_low
        if source is None:
            row["decode_ok"] = True
            row["embedded"] = False
            row["scores"] = {}
        else:
            row["decode_ok"] = bool(source.get("decode_ok", True))
            row["embedded"] = bool(source.get("embedded", False))
            row["black"] = bool(source.get("black", False))
            row["scores"] = source.get("scores") or {}
            row["siglip_motion"] = source.get("motion")
        attached.append(row)
    return attached


def run_hands(cfg, episode: dict, siglip_dirs: dict, out_dir: Path) -> dict:
    presence = dict(cfg.presence)
    if not presence.get("enabled"):
        raise ValueError("presence.enabled is false")
    if presence.get("detector") != "mediapipe_hands":
        raise ValueError(f"unsupported presence detector {presence.get('detector')}")
    polygon = presence["work_area"]
    interval = float(presence.get("sample_interval_s", cfg.interval_s))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = HandDetector(
        str(presence["model_path"]),
        float(presence.get("min_confidence", 0.5)),
        int(presence.get("max_hands", 2)),
    )
    motion = MotionRoi(polygon)
    stamps = _stamps(float(episode["duration"]), interval)
    images = _grab(episode, stamps, slack=interval * 0.45)
    samples = []
    try:
        for stamp in stamps:
            image = images.get(stamp)
            if image is None:
                continue
            observed = _observe(
                image, detector, motion, polygon, float(presence.get("min_overlap", 0.2))
            )
            observed["timestamp_s"] = stamp
            samples.append(observed)
            if len(samples) % 30 == 0:
                print("HANDS sampled %.0fs" % stamp, flush=True)
        idle_cfg = {
            "sample_interval_s": interval,
            "absence_confirm_s": float(presence["absence_confirm_s"]),
            "motion_low": float(presence["motion_low"]),
            "boundary_context_s": float(presence["boundary_context_s"]),
            "bridge_s": float(presence.get("bridge_s", 10)),
        }
        coarse = idle_intervals(samples, float(episode["duration"]), idle_cfg)
        radius = float(presence.get("refine_radius_s", 8))
        fps = float(presence.get("refine_fps", 1))
        refined, edge_images = _refine(
            episode,
            coarse,
            detector,
            polygon,
            float(presence.get("min_overlap", 0.2)),
            radius,
            fps,
        )
        intervals = reapply_peel(refined, float(presence["boundary_context_s"]), float(episode["duration"]))
    finally:
        detector.close()
    thumbs = _write_thumbs(out_dir, intervals, images, edge_images)
    write_jsonl(out_dir / "hands.jsonl", samples)
    write_json(out_dir / "idle.json", {"intervals": intervals, "coarse": coarse})
    fused = {}
    for name, folder in siglip_dirs.items():
        rows = _read_jsonl(Path(folder) / "frames" / f"{episode['episode_id']}.jsonl")
        attached = attach_siglip(samples, rows, "full", float(presence["motion_low"]))
        segments = fuse_timeline(attached, intervals, float(episode["duration"]), cfg.temporal)
        fused[name] = segments
        write_jsonl(out_dir / f"fusion_{name}.jsonl", segments)
    payload = {
        "episode_id": episode["episode_id"],
        "duration": float(episode["duration"]),
        "detector": "mediapipe_hands",
        "intervals": intervals,
        "fusion": fused,
        "thumbs": thumbs,
        "n_hands": sum(1 for row in samples if row.get("hand_present_in_work_area")),
        "n_frames": len(samples),
    }
    write_json(out_dir / "hand_summary.json", payload)
    return payload


def _write_thumbs(out_dir: Path, intervals: list[dict], images: dict, edge_images: dict) -> list[dict]:
    folder = out_dir / "thumbs"
    folder.mkdir(parents=True, exist_ok=True)
    pool = dict(images)
    pool.update(edge_images)
    written = []
    for index, row in enumerate(intervals):
        for key in ("raw_start", "start", "end", "raw_end"):
            stamp = float(row[key])
            nearest = min(pool, key=lambda item: abs(item - stamp), default=None)
            if nearest is None or abs(nearest - stamp) > 2.5:
                continue
            image = pool[nearest].copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle([8, 8, 120, 36], fill=(0, 0, 0))
            draw.text((14, 12), _clock(nearest), fill=(255, 255, 255))
            name = f"idle{index}_{key}_{_clock(nearest).replace(':', '')}.jpg"
            image.save(folder / name, quality=85)
            written.append({"interval": index, "edge": key, "timestamp_s": nearest, "file": f"thumbs/{name}"})
    return written


def write_comparison(path: Path, episode: dict, summary: dict, focus: list[tuple[float, float]]) -> None:
    lines = [
        "# 人手检测后的空档",
        "",
        "工位相机主要拍到的是手和前臂，不用人体框。检测器是 MediaPipe Hands。",
        "连续 15 秒工位上看不到手，并且运动低于门槛，才标 `idle_no_person`。",
        "两端各留 5 秒不标成空档。SigLIP 的 0.02 门槛没有改，模型也没有换成更大的。",
        "",
        f"- `{episode['episode_id']}`",
        f"- 时长 {float(episode['duration']):.1f} 秒",
        f"- 采样 {summary['n_frames']} 帧，其中 {summary['n_hands']} 帧工位上有手",
        "",
        "## 空档",
        "",
        "| 原始起 | 原始止 | 削边后起 | 削边后止 | 运动 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    if not summary["intervals"]:
        lines.append("| 无 | 无 | 无 | 无 |  |")
    for row in summary["intervals"]:
        lines.append(
            "| %s | %s | %s | %s | %.4f |"
            % (
                _clock(row["raw_start"]),
                _clock(row["raw_end"]),
                _clock(row["start"]),
                _clock(row["end"]),
                float(row.get("motion") or 0.0),
            )
        )
    lines.extend(["", "## 人工指出的两段", ""])
    for start, end in focus:
        hit = [
            row
            for row in summary["intervals"]
            if float(row["raw_end"]) > start and float(row["raw_start"]) < end
        ]
        if hit:
            text = "，".join(
                f"{_clock(row['raw_start'])}–{_clock(row['raw_end'])}（削边 {_clock(row['start'])}–{_clock(row['end'])}）"
                for row in hit
            )
            lines.append(f"- {_clock(start)}–{_clock(end)}：对上 {text}")
        else:
            lines.append(f"- {_clock(start)}–{_clock(end)}：这段没有被标成空档")
    lines.extend(["", "## 和原来的 SigLIP 对照", ""])
    lines.append("原来两个模型整段都是 `uncertain`。下面是融合后的标签。")
    lines.append("")
    for name, segments in summary["fusion"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| 起 | 止 | 标签 | 原因 |")
        lines.append("| --- | --- | --- | --- |")
        for segment in segments:
            if segment["label"] == "uncertain" and segment["reason"] not in ("absent_high_motion",):
                if float(segment["end"]) - float(segment["start"]) < 20:
                    continue
            lines.append(
                "| %s | %s | `%s` | %s |"
                % (_clock(segment["start"]), _clock(segment["end"]), segment["label"], segment["reason"])
            )
        lines.append("")
    if summary.get("thumbs"):
        lines.extend(["## 切换附近的帧", ""])
        for thumb in summary["thumbs"]:
            lines.append(f"- `{thumb['file']}` {_clock(thumb['timestamp_s'])} {thumb['edge']}")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
