"""把抽帧、编码、打分、时间轴串起来。原片只读。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from siglip_filter.annotate import build_queue, load_window_rows
from siglip_filter.config import FilterConfig, _hash
from siglip_filter.decode import cache_dir, decode_episode, decode_many, video_stat
from siglip_filter.encode import cached_text
from siglip_filter.logistic import LogisticPredictor, apply_logistic
from siglip_filter.manifest import (
    assert_manifest,
    build_manifest,
    checkpoint_path,
    read_checkpoint,
    write_json,
    write_jsonl,
)
from siglip_filter.prompts import PROMPTS_VERSION, canonical_task, prompts_for
from siglip_filter.report import write_index, write_report
from siglip_filter.similarity import aggregate, l2_normalize
from siglip_filter.temporal import label_timeline


class PilotLimit(RuntimeError):
    pass


def load_episodes(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["episodes"]
    for row in data:
        for key in ("episode_id", "video_path", "task", "duration"):
            if key not in row:
                raise ValueError(f"episode missing {key}")
        if float(row["duration"]) <= 0:
            raise ValueError(f"{row['episode_id']} duration must be positive")
    return data


def assert_pilot(episodes: list[dict], cfg: FilterConfig, allow_full: bool, validation: Path | None) -> float:
    hours = sum(float(row["duration"]) for row in episodes) / 3600.0
    if hours <= float(cfg.pilot_max_hours) + 1e-9:
        return hours
    if not allow_full:
        raise PilotLimit(
            f"{hours:.1f}h exceeds the pilot cap of {cfg.pilot_max_hours}h; "
            "finish validation before a full run"
        )
    if validation is None or not Path(validation).exists():
        raise PilotLimit("a full run needs --validation with relevant_segment_recall")
    report = json.loads(Path(validation).read_text(encoding="utf-8"))
    recall = report.get("relevant_segment_recall")
    if recall is None and isinstance(report.get("full"), dict):
        recall = report["full"].get("relevant_segment_recall")
    if recall is None:
        raise PilotLimit("validation report has no relevant_segment_recall")
    return hours


def _done(out_dir: Path, episode: dict, cfg: FilterConfig) -> dict | None:
    checkpoint = read_checkpoint(out_dir, episode["episode_id"])
    manifest_path = Path(out_dir) / "manifests" / f"{episode['episode_id']}.json"
    if not checkpoint or checkpoint.get("status") != "complete":
        return None
    if checkpoint.get("decision_hash") != cfg.decision_hash():
        return None
    if checkpoint.get("video_stat") != video_stat(Path(episode["video_path"])):
        return None
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float16))
    tmp.replace(path)


def _embeddings(episode, frames, cfg: FilterConfig, out_dir: Path, get_encoder):
    folder = cache_dir(out_dir, episode["episode_id"])
    meta_path = folder / "emb_meta.json"
    full_path = folder / "emb_full.npy"
    roi_path = folder / "emb_roi.npy"
    want_roi = bool(cfg.roi.get("enabled", True))
    if meta_path.exists() and full_path.exists() and (not want_roi or roi_path.exists()):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("model") == cfg.model_name
            and meta.get("sample_hash") == cfg.sample_hash()
            and meta.get("count") == len(frames)
            and bool(meta.get("roi")) == want_roi
        ):
            full = np.load(full_path).astype(np.float32)
            roi = np.load(roi_path).astype(np.float32) if want_roi else None
            return full, roi
    from PIL import Image

    ok = [frame for frame in frames if frame.get("decode_ok")]
    if not ok:
        return np.zeros((len(frames), 1), np.float32), (
            np.zeros((len(frames), 1), np.float32) if want_roi else None
        )
    encoder = get_encoder()
    if encoder.model_name != cfg.model_name:
        raise RuntimeError(
            f"encoder {encoder.model_name} does not match config model {cfg.model_name}"
        )
    full_images = [Image.open(folder / frame["thumb"]).convert("RGB") for frame in ok]
    encoded = encoder.encode_images(full_images)
    full = np.zeros((len(frames), encoded.shape[1]), np.float32)
    full[[frame["index"] for frame in ok]] = encoded
    roi = None
    if want_roi:
        roi_images = [Image.open(folder / frame["roi"]).convert("RGB") for frame in ok]
        encoded_roi = encoder.encode_images(roi_images)
        roi = np.zeros((len(frames), encoded_roi.shape[1]), np.float32)
        roi[[frame["index"] for frame in ok]] = encoded_roi
    _write_npy(full_path, full)
    if roi is not None:
        _write_npy(roi_path, roi)
    write_json(
        meta_path,
        {
            "model": cfg.model_name,
            "sample_hash": cfg.sample_hash(),
            "count": len(frames),
            "roi": want_roi,
            "dim": int(full.shape[1]),
        },
    )
    return full, roi


def _text_vectors(task: str, cfg: FilterConfig, out_dir: Path, get_encoder):
    prompts = prompts_for(task)
    path = Path(out_dir) / "cache" / "text" / (
        _hash(
            {
                "model": cfg.model_name,
                "task": canonical_task(task),
                "prompts": PROMPTS_VERSION,
            }
        )
        + ".npy"
    )
    if path.exists():
        return prompts, np.load(path)
    return prompts, cached_text(get_encoder(), [prompt.text for prompt in prompts], path)


def _score(frames, full, roi, text, prompts):
    text_n = l2_normalize(text)
    full_sims = l2_normalize(full) @ text_n.T
    roi_sims = l2_normalize(roi) @ text_n.T if roi is not None else None
    scored = []
    for index, frame in enumerate(frames):
        row = dict(frame)
        row["embedded"] = bool(frame.get("decode_ok"))
        row["scores"] = {}
        if row["embedded"]:
            row["scores"]["full"] = aggregate(full_sims[index], prompts)
            if roi_sims is not None:
                row["scores"]["roi"] = aggregate(roi_sims[index], prompts)
        scored.append(row)
    return scored


def _timeline(duration, frames, view, cfg: FilterConfig, predictor, full, roi):
    adjust = None
    if predictor is not None:
        def adjust(windows, view=view):
            del view
            return apply_logistic(windows, frames, full, roi, predictor, duration)
    return label_timeline(duration, frames, view, cfg.temporal, cfg.quality, adjust=adjust)


def process_episode(episode: dict, cfg: FilterConfig, out_dir: Path, get_encoder, predictor=None) -> tuple[dict, dict]:
    out_dir = Path(out_dir)
    started = time.time()
    try:
        cached = _done(out_dir, episode, cfg)
        if cached is not None:
            return cached, {
                "episode_id": episode["episode_id"],
                "status": "cached",
                "wall_s": 0.0,
                "video_s": float(episode["duration"]),
            }
        index = decode_episode(episode, cfg, out_dir)
        frames = index["frames"]
        prompts = None
        full = roi = None
        if any(frame.get("decode_ok") for frame in frames):
            full, roi = _embeddings(episode, frames, cfg, out_dir, get_encoder)
            prompts, text = _text_vectors(episode["task"], cfg, out_dir, get_encoder)
            frames = _score(frames, full, roi, text, prompts)
        else:
            frames = [dict(frame, embedded=False, scores={}) for frame in frames]
        views = {}
        window_sets = {}
        view_names = ("full", "roi") if roi is not None else ("full",)
        for view in view_names:
            windows, segments = label_timeline(
                float(episode["duration"]), frames, view, cfg.temporal, cfg.quality
            )
            views[view] = segments
            window_sets[view] = windows
        decision = cfg.temporal.get("decision_view", "full")
        confidence_kind = "cosine_margin"
        if predictor is not None:
            windows, segments = _timeline(
                float(episode["duration"]), frames, decision, cfg, predictor, full, roi
            )
            views["logistic"] = segments
            window_sets["logistic"] = windows
            decision = "logistic"
            confidence_kind = "logistic_probability"
        manifest = build_manifest(
            episode,
            views,
            {
                "decision_view": decision,
                "model": cfg.model_name,
                "confidence_kind": confidence_kind,
                "thresholds": {"quality": cfg.quality, "temporal": cfg.temporal, "classify": cfg.classify},
            },
        )
        assert_manifest(manifest)
        for name, block in manifest["views"].items():
            probe = dict(manifest)
            probe["segments"] = block["segments"]
            assert_manifest(probe)
        episode_id = episode["episode_id"]
        write_json(out_dir / "manifests" / f"{episode_id}.json", manifest)
        write_jsonl(
            out_dir / "frames" / f"{episode_id}.jsonl",
            [{**frame, "episode_id": episode_id} for frame in frames],
        )
        for name, windows in window_sets.items():
            write_jsonl(
                out_dir / "windows" / f"{episode_id}.{name}.jsonl",
                [{**row, "episode_id": episode_id} for row in windows],
            )
        write_report(
            out_dir / "reports" / f"{episode_id}.html",
            manifest,
            frames,
            window_sets[decision],
        )
        encoder = get_encoder.holder.get("encoder") if hasattr(get_encoder, "holder") else None
        gpu = encoder.peak_memory_mb() if encoder is not None and hasattr(encoder, "peak_memory_mb") else None
        log = {
            "episode_id": episode_id,
            "status": "complete",
            "frames": len(frames),
            "video_s": float(episode["duration"]),
            "wall_s": time.time() - started,
            "gpu_memory_mb": gpu,
            "model": cfg.model_name,
            "decision_hash": cfg.decision_hash(),
        }
        write_json(out_dir / "logs" / f"{episode_id}.json", log)
        write_json(
            checkpoint_path(out_dir, episode_id),
            {
                "episode_id": episode_id,
                "status": "complete",
                "decision_hash": cfg.decision_hash(),
                "video_stat": video_stat(Path(episode["video_path"])),
                "model": cfg.model_name,
            },
        )
        return manifest, log
    except Exception as exc:
        write_json(
            checkpoint_path(out_dir, episode["episode_id"]),
            {
                "episode_id": episode["episode_id"],
                "status": "error",
                "error": str(exc),
                "decision_hash": cfg.decision_hash(),
            },
        )
        raise


def run(cfg, episodes, out_dir, allow_full: bool = False, validation=None, encoder=None):
    if isinstance(cfg, (str, Path)):
        cfg = FilterConfig.load(cfg)
    if isinstance(episodes, (str, Path)):
        episodes = load_episodes(episodes)
    out_dir = Path(out_dir)
    assert_pilot(episodes, cfg, allow_full, None if validation is None else Path(validation))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", cfg.raw)
    holder = {"encoder": encoder}

    def get_encoder():
        if holder["encoder"] is None:
            from siglip_filter.encode import Siglip2Encoder

            holder["encoder"] = Siglip2Encoder(
                cfg.model_name, cfg.model_dtype, cfg.device, cfg.batch_size
            )
        return holder["encoder"]

    get_encoder.holder = holder
    predictor = None
    if cfg.classify.get("mode") == "logistic":
        directory = cfg.classify.get("classifier_dir")
        if not directory:
            raise RuntimeError("classify.mode logistic needs classify.classifier_dir")
        predictor = LogisticPredictor(directory)
    todo = [episode for episode in episodes if _done(out_dir, episode, cfg) is None]
    if todo:
        decode_many(todo, cfg, out_dir)
    manifests = []
    logs = []
    for episode in episodes:
        manifest, log = process_episode(episode, cfg, out_dir, get_encoder, predictor)
        manifests.append(manifest)
        logs.append(log)
        print(
            f"FILTER {episode['episode_id']} {log['status']} "
            f"video_s {float(episode['duration']):.1f} wall_s {log['wall_s']:.2f}",
            flush=True,
        )
    write_index(out_dir / "reports" / "index.html", manifests)
    write_json(out_dir / "logs" / "run.json", {"episodes": logs})
    rows = load_window_rows(out_dir, "full")
    if rows:
        write_jsonl(
            out_dir / "annotation_queue.jsonl",
            build_queue(rows, cfg.annotation_windows),
        )
    return manifests
