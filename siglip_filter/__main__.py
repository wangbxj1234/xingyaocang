"""试点入口。超过配置里的小时数会拒绝，不会把全库跑起来。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from siglip_filter.annotate import build_queue, load_window_rows, serve
from siglip_filter.config import FilterConfig
from siglip_filter.decode import cache_dir
from siglip_filter.logistic import fit_logistic, pool_vector
from siglip_filter.manifest import write_json
from siglip_filter.metrics import evaluate
from siglip_filter.pipeline import load_episodes, run


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _cache(out_dir: Path, episode_id: str) -> dict:
    folder = cache_dir(out_dir, episode_id)
    index = json.loads((folder / "frames_index.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "manifests" / f"{episode_id}.json").read_text(encoding="utf-8"))
    roi_path = folder / "emb_roi.npy"
    return {
        "timestamps": [frame["timestamp_s"] for frame in index["frames"]],
        "embedded": [bool(frame.get("decode_ok")) for frame in index["frames"]],
        "full": np.load(folder / "emb_full.npy").astype(np.float32),
        "roi": np.load(roi_path).astype(np.float32) if roi_path.exists() else None,
        "duration": float(manifest["duration"]),
    }


def _train(args) -> None:
    cfg = FilterConfig.load(args.config)
    feature_mode = args.features or cfg.classify.get("features", "both")
    loaded = {}
    examples = []
    for row in _read_jsonl(args.labels):
        episode_id = row["episode_id"]
        if episode_id not in loaded:
            loaded[episode_id] = _cache(args.cache, episode_id)
        cache = loaded[episode_id]
        vector = pool_vector(
            cache["timestamps"],
            cache["embedded"],
            cache["full"],
            cache["roi"],
            row["start"],
            row["end"],
            cache["duration"],
            feature_mode,
        )
        if vector is None:
            raise SystemExit(f"no embedding for {episode_id} {row['start']}-{row['end']}")
        examples.append({**row, "vector": vector})
    metrics = fit_logistic(
        examples,
        feature_mode,
        args.out,
        float(args.val_fraction if args.val_fraction is not None else cfg.classify.get("val_fraction", 0.2)),
        int(args.seed if args.seed is not None else cfg.classify.get("seed", 1)),
        args.split or cfg.classify.get("split", "episode"),
    )
    print(
        f"TRAIN relevant_recall {metrics['relevant_recall']} "
        f"n_train {metrics['n_train']} n_val {metrics['n_val']}",
        flush=True,
    )


def _validate(args) -> None:
    manifests = {}
    for path in sorted(Path(args.manifests).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[payload["episode_id"]] = payload
    manual = _read_jsonl(args.labels)
    if args.view == "both":
        report = {
            "full": evaluate(manual, manifests, "full"),
            "roi": evaluate(manual, manifests, "roi"),
        }
    else:
        report = evaluate(manual, manifests, args.view)
    write_json(args.out, report)
    recall = report.get("relevant_segment_recall")
    if recall is None and "full" in report:
        recall = report["full"].get("relevant_segment_recall")
    print(f"VALIDATE relevant_segment_recall {recall}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="siglip_filter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="decode, embed, and write manifests")
    run_p.add_argument("--config", type=Path, required=True)
    run_p.add_argument("--episodes", type=Path, required=True)
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--allow-full", action="store_true")
    run_p.add_argument("--validation", type=Path)

    val_p = sub.add_parser("validate", help="score manifests against manual windows")
    val_p.add_argument("--manifests", type=Path, required=True)
    val_p.add_argument("--labels", type=Path, required=True)
    val_p.add_argument("--out", type=Path, required=True)
    val_p.add_argument("--view", choices=("full", "roi", "both"), default="full")

    train_p = sub.add_parser("train", help="logistic regression on cached embeddings")
    train_p.add_argument("--config", type=Path, required=True)
    train_p.add_argument("--cache", type=Path, required=True)
    train_p.add_argument("--labels", type=Path, required=True)
    train_p.add_argument("--out", type=Path, required=True)
    train_p.add_argument("--features", choices=("full", "roi", "both"))
    train_p.add_argument("--split", choices=("episode", "date"))
    train_p.add_argument("--val-fraction", type=float)
    train_p.add_argument("--seed", type=int)

    queue_p = sub.add_parser("queue", help="pick windows for annotation")
    queue_p.add_argument("--out", type=Path, required=True)
    queue_p.add_argument("--n", type=int, default=500)
    queue_p.add_argument("--view", default="full")

    serve_p = sub.add_parser("serve", help="open the report and save manual labels")
    serve_p.add_argument("--out", type=Path, required=True)
    serve_p.add_argument("--port", type=int, default=8765)

    hands_p = sub.add_parser("hands", help="mark idle stretches from hands, not from SigLIP")
    hands_p.add_argument("--config", type=Path, required=True)
    hands_p.add_argument("--episodes", type=Path, required=True)
    hands_p.add_argument("--siglip", action="append", required=True, help="name=output_dir")
    hands_p.add_argument("--out", type=Path, required=True)
    hands_p.add_argument("--report", type=Path)
    hands_p.add_argument("--focus", action="append", default=[], help="start:end seconds")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        episodes = load_episodes(args.episodes)
        run(args.config, episodes, args.out, allow_full=args.allow_full, validation=args.validation)
        return
    if args.cmd == "validate":
        _validate(args)
        return
    if args.cmd == "train":
        _train(args)
        return
    if args.cmd == "queue":
        from siglip_filter.manifest import write_jsonl

        rows = build_queue(load_window_rows(args.out, args.view), args.n)
        write_jsonl(args.out / "annotation_queue.jsonl", rows)
        print(f"QUEUE {len(rows)}", flush=True)
        return
    if args.cmd == "serve":
        serve(args.out, args.port)
        return
    if args.cmd == "hands":
        from siglip_filter.hand_pass import run_hands, write_comparison

        episodes = load_episodes(args.episodes)
        if len(episodes) != 1:
            raise SystemExit("hands currently runs one episode")
        siglip_dirs = {}
        for item in args.siglip:
            name, folder = item.split("=", 1)
            siglip_dirs[name] = folder
        cfg = FilterConfig.load(args.config)
        summary = run_hands(cfg, episodes[0], siglip_dirs, args.out)
        focus = []
        for item in args.focus:
            start, end = item.split(":")
            focus.append((float(start), float(end)))
        if args.report:
            write_comparison(args.report, episodes[0], summary, focus)
        print(
            f"HANDS {episodes[0]['episode_id']} idle {len(summary['intervals'])} "
            f"frames {summary['n_frames']} with_hand {summary['n_hands']}",
            flush=True,
        )
        return
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
