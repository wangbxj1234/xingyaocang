"""顺序抽帧。原片只读。抽过的结果留在缓存里，同一段不重复解。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from siglip_filter.config import FilterConfig
from siglip_filter.quality import failed_measure, measure


def video_stat(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        names = sorted(
            child.name
            for child in path.iterdir()
            if child.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        return {"kind": "dir", "count": len(names), "names": names}
    st = path.stat()
    return {"kind": "file", "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def source_kind(episode: dict) -> str:
    kind = episode.get("source") or "auto"
    path = Path(episode["video_path"])
    if kind == "auto":
        if path.is_dir():
            return "frames"
        if path.suffix.lower() == ".mcap":
            return "mcap"
        return "video"
    if kind not in ("frames", "mcap", "video"):
        raise ValueError(f"unsupported source {kind}")
    return kind


def crop_roi(image: Image.Image, roi: dict) -> Image.Image:
    if not roi.get("enabled", True):
        return image.convert("RGB")
    width, height = image.size
    x = int(round(float(roi["x"]) * width))
    y = int(round(float(roi["y"]) * height))
    box_w = int(round(float(roi["w"]) * width))
    box_h = int(round(float(roi["h"]) * height))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    x2 = max(x + 1, min(width, x + max(box_w, 1)))
    y2 = max(y + 1, min(height, y + max(box_h, 1)))
    return image.crop((x, y, x2, y2)).convert("RGB")


def fit_square(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.BILINEAR)


def _iter_frames_dir(path: Path, interval_s: float, duration: float):
    files = sorted(
        child for child in Path(path).iterdir() if child.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    sidecar = Path(path) / "timestamps.json"
    stamps = None
    if sidecar.exists():
        stamps = json.loads(sidecar.read_text(encoding="utf-8"))
        if len(stamps) != len(files):
            raise ValueError(f"{sidecar} has {len(stamps)} stamps for {len(files)} frames")
    for index, file in enumerate(files):
        timestamp = float(stamps[index]) if stamps is not None else index * interval_s
        if timestamp < -1e-6 or timestamp > duration + 1e-6:
            continue
        try:
            with Image.open(file) as image:
                yield {"timestamp_s": timestamp, "image": image.convert("RGB"), "error": None}
        except Exception as exc:
            yield {"timestamp_s": timestamp, "image": None, "error": str(exc)}


def _iter_video(path: Path, interval_s: float, duration: float):
    errors = []
    try:
        import av
    except ImportError:
        av = None
        errors.append("av")
    if av is not None:
        yield from _iter_av(path, interval_s, duration)
        return
    try:
        import cv2
    except ImportError:
        cv2 = None
        errors.append("opencv-python-headless")
    if cv2 is not None:
        yield from _iter_cv2(cv2, path, interval_s, duration)
        return
    raise RuntimeError(
        "video decode needs one of: " + ", ".join(errors)
    )


def _iter_av(path: Path, interval_s: float, duration: float):
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    next_grid = 0.0
    try:
        for frame in container.decode(stream):
            if frame.time is None:
                timestamp = float(frame.pts * stream.time_base)
            else:
                timestamp = float(frame.time)
            if timestamp > duration + 1e-6:
                break
            if timestamp + 1e-3 < next_grid:
                continue
            try:
                image = frame.to_image().convert("RGB")
                yield {"timestamp_s": timestamp, "image": image, "error": None}
            except Exception as exc:
                yield {"timestamp_s": timestamp, "image": None, "error": str(exc)}
            next_grid += interval_s
    finally:
        container.close()


def _iter_cv2(cv2, path: Path, interval_s: float, duration: float):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 15.0
    step = max(1, int(round(interval_s * fps)))
    index = 0
    next_keep = 0
    try:
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            timestamp = index / fps
            if timestamp > duration + 1e-6:
                break
            if index == next_keep:
                ok, bgr = capture.retrieve()
                pos_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pos_ms > 0:
                    timestamp = pos_ms / 1000.0
                if not ok or bgr is None:
                    yield {"timestamp_s": timestamp, "image": None, "error": "retrieve failed"}
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    yield {
                        "timestamp_s": timestamp,
                        "image": Image.fromarray(rgb),
                        "error": None,
                    }
                next_keep += step
            index += 1
    finally:
        capture.release()


def _iter_mcap(episode: dict, interval_s: float, duration: float):
    from mcap.reader import make_reader

    from fuhe.mcap_rgb import _camera_frame_cls
    import io

    path = Path(episode["video_path"])
    topic = episode.get("topic")
    if not topic:
        mount = episode.get("mount") or "head"
        topic = f"/camera/{mount}/rgb"
    epoch_ns = episode.get("epoch_ns", episode.get("start_timestamp_ns"))
    if epoch_ns is None:
        raise ValueError(f"{episode['episode_id']} is mcap and needs epoch_ns")
    epoch_ns = int(epoch_ns)
    camera_cls = _camera_frame_cls(path)
    next_grid = 0.0
    with path.open("rb") as handle:
        for _schema, _channel, message in make_reader(handle).iter_messages(
            topics=[topic],
            start_time=epoch_ns,
            end_time=int(epoch_ns + duration * 1e9),
        ):
            log_t = (int(message.log_time) - epoch_ns) / 1e9
            if log_t > duration + 1.0:
                break
            if log_t + 0.25 < next_grid:
                continue
            try:
                camera = camera_cls()
                camera.ParseFromString(message.data)
                timestamp = (int(camera.header.timestamp_ns) - epoch_ns) / 1e9
                if timestamp + 1e-3 < next_grid or timestamp > duration + 1e-6:
                    continue
                encoding = camera.encoding.lower()
                if encoding not in ("jpeg", "jpg"):
                    raise ValueError(f"unsupported RGB encoding: {camera.encoding}")
                image = Image.open(io.BytesIO(camera.data)).convert("RGB")
                yield {"timestamp_s": float(timestamp), "image": image, "error": None}
                next_grid = float(timestamp) + interval_s
            except Exception as exc:
                timestamp = max(next_grid, log_t)
                yield {"timestamp_s": float(timestamp), "image": None, "error": str(exc)}
                next_grid = float(timestamp) + interval_s


def iter_samples(episode: dict, cfg: FilterConfig):
    kind = source_kind(episode)
    path = Path(episode["video_path"])
    duration = float(episode["duration"])
    if kind == "frames":
        yield from _iter_frames_dir(path, cfg.interval_s, duration)
    elif kind == "mcap":
        yield from _iter_mcap(episode, cfg.interval_s, duration)
    else:
        yield from _iter_video(path, cfg.interval_s, duration)


def cache_dir(out_dir: Path, episode_id: str) -> Path:
    return Path(out_dir) / "cache" / episode_id


def load_index(out_dir: Path, episode: dict, cfg: FilterConfig) -> dict | None:
    path = cache_dir(out_dir, episode["episode_id"]) / "frames_index.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sample_hash") != cfg.sample_hash():
        return None
    if payload.get("video_stat") != video_stat(Path(episode["video_path"])):
        return None
    return payload


def cache_samples(episode: dict, samples, cfg: FilterConfig, out_dir: Path) -> dict:
    folder = cache_dir(out_dir, episode["episode_id"])
    thumbs = folder / "thumbs"
    rois = folder / "roi"
    thumbs.mkdir(parents=True, exist_ok=True)
    rois.mkdir(parents=True, exist_ok=True)
    prev = None
    frames = []
    for index, sample in enumerate(samples):
        row = {
            "index": index,
            "timestamp_s": float(sample["timestamp_s"]),
            "thumb": f"thumbs/{index:06d}.jpg",
            "roi": f"roi/{index:06d}.jpg",
        }
        image = sample.get("image")
        if image is None:
            row.update(failed_measure(sample.get("error") or "decode failed"))
            frames.append(row)
            continue
        metrics, prev = measure(image, prev, cfg.quality)
        row.update(metrics)
        full = fit_square(image, cfg.image_size)
        roi = fit_square(crop_roi(image, cfg.roi), cfg.image_size)
        full.save(thumbs / f"{index:06d}.jpg", quality=90)
        roi.save(rois / f"{index:06d}.jpg", quality=90)
        frames.append(row)
    payload = {
        "episode_id": episode["episode_id"],
        "video_path": str(episode["video_path"]),
        "sample_hash": cfg.sample_hash(),
        "video_stat": video_stat(Path(episode["video_path"])),
        "image_size": cfg.image_size,
        "interval_s": cfg.interval_s,
        "frames": frames,
    }
    _write_json(folder / "frames_index.json", payload)
    return payload


def decode_episode(episode: dict, cfg: FilterConfig, out_dir: Path) -> dict:
    cached = load_index(out_dir, episode, cfg)
    if cached is not None:
        return cached
    path = Path(episode["video_path"])
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    return cache_samples(episode, iter_samples(episode, cfg), cfg, out_dir)


def _decode_worker(job: dict) -> str:
    cfg = FilterConfig.from_dict(job["config"])
    payload = decode_episode(job["episode"], cfg, Path(job["out"]))
    return payload["episode_id"]


def decode_many(episodes: list[dict], cfg: FilterConfig, out_dir: Path) -> list[dict]:
    pending = []
    ready = []
    for episode in episodes:
        cached = load_index(out_dir, episode, cfg)
        if cached is not None:
            ready.append(cached)
        else:
            pending.append(episode)
    if not pending:
        return ready
    if cfg.workers == 1 or len(pending) == 1:
        for episode in pending:
            ready.append(decode_episode(episode, cfg, out_dir))
        return ready
    from concurrent.futures import ProcessPoolExecutor

    jobs = [{"episode": episode, "config": cfg.raw, "out": str(out_dir)} for episode in pending]
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        list(pool.map(_decode_worker, jobs))
    for episode in pending:
        ready.append(load_index(out_dir, episode, cfg))
    missing = [episode["episode_id"] for episode, row in zip(pending, ready[-len(pending):]) if row is None]
    if missing:
        raise RuntimeError(f"decode produced no cache for {missing}")
    return ready


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_thumbs(out_dir: Path, episode_id: str, frames: list[dict], kind: str) -> list[Image.Image | None]:
    folder = cache_dir(out_dir, episode_id)
    images = []
    for frame in frames:
        if not frame.get("decode_ok"):
            images.append(None)
            continue
        images.append(Image.open(folder / frame[kind]).convert("RGB"))
    return images
