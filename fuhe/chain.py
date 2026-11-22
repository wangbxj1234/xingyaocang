"""切段、收链、写字幕。不改 Flora 的 prompt 和代码。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from fuhe.chain_math import (
    blocks,
    chain,
    choose_block,
    covered,
    duration_us,
    managed,
    parse_windows,
    seconds,
)

H20_BYTES = 80 * 1024 ** 3


def gpu_fits(count: int, biggest_bytes: int) -> bool:
    """One 80GB-class GPU holds both models. 24GB cards still need three."""
    if count < 1:
        return False
    if biggest_bytes >= H20_BYTES:
        return True
    return count == 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    flora = os.environ.get("FLORA_CODE", "/mnt/cpfs/zhongjiayu0757/flora/code")
    sys.path.insert(0, flora)
    import torch
    from batch_utils.media import probe
    from batch_utils.models import Session
    from batch_utils.pipeline import caption_single, segment

    visible = torch.cuda.device_count()
    biggest = max((torch.cuda.get_device_properties(i).total_memory for i in range(visible)), default=0)
    if not gpu_fits(visible, biggest):
        raise SystemExit(f"need one 80GB GPU or three smaller GPUs, saw {visible}")
    work = args.workdir
    meta = json.loads((work / "extract_meta.json").read_text(encoding="utf-8"))
    span = duration_us(float(meta["duration_s"]))
    block_rows = blocks(span)
    paths = [work / f"block_{index:02d}.mp4" for index in range(len(block_rows))]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    print("BLOCKS", [(seconds(a), seconds(b)) for a, b in block_rows], flush=True)
    session = Session()
    try:
        print("LOAD_TEMPSEG", flush=True)
        session.load("tempseg")
        all_cuts = []
        all_cover = []
        audit_windows = []
        for index, (start, end) in enumerate(block_rows):
            lo, hi = managed(index, len(block_rows), start, span)
            videos = {"main": probe(paths[index])}
            print(f"SEGMENT block {index} local_duration {videos['main'].duration_s}", flush=True)
            _segments, calls = segment(videos, "human", session, "", lambda *_a, **_k: None)
            cuts, cover, windows = parse_windows(calls, start, lo, hi)
            print(
                f"BLOCK {index} windows {len(windows)} cuts {len(cuts)} manage {seconds(lo)} {seconds(hi)}",
                flush=True,
            )
            all_cuts.extend(cuts)
            all_cover.extend(cover)
            audit_windows.extend(windows)
        kept = chain(all_cuts, span)
        pieces = []
        for start, end in zip(kept, kept[1:]):
            is_gap = not covered(start, end, all_cover)
            pieces.append({
                "start_us": start,
                "end_us": end,
                "gap": is_gap,
                "block": None if is_gap else choose_block(start, end, block_rows, span),
            })
        print("CHAIN", len(pieces), "gaps", sum(row["gap"] for row in pieces), flush=True)
        (work / "chain.json").write_text(
            json.dumps({"cuts_us": kept, "pieces": pieces, "windows": audit_windows}),
            encoding="utf-8",
        )
        missing = [row for row in pieces if not row["gap"] and row["block"] is None]
        if missing:
            raise RuntimeError(
                "a segment does not fit in one block, including 2 seconds of context: "
                + json.dumps([
                    {"start_s": seconds(row["start_us"]), "end_s": seconds(row["end_us"])}
                    for row in missing
                ])
            )
        print("LOAD_CAPTIONER", flush=True)
        session.load("captioner")
        captions = {}
        by_block = {}
        for piece_index, row in enumerate(pieces):
            if row["gap"]:
                continue
            by_block.setdefault(row["block"], []).append(piece_index)
        for block_index, piece_indexes in by_block.items():
            block_start = block_rows[block_index][0]
            local = [{
                "start_s": seconds(pieces[piece_index]["start_us"] - block_start),
                "end_s": seconds(pieces[piece_index]["end_us"] - block_start),
            } for piece_index in piece_indexes]
            videos = {"main": probe(paths[block_index])}
            print(f"CAPTION block {block_index} rows {len(local)}", flush=True)
            rows, _audit = caption_single(videos, local, "Human", session, "", lambda *_a, **_k: None)
            if len(rows) != len(piece_indexes):
                raise RuntimeError(f"caption count {len(rows)} != {len(piece_indexes)} on block {block_index}")
            for piece_index, row in zip(piece_indexes, rows):
                captions[piece_index] = row
        result_rows = []
        for piece_index, piece in enumerate(pieces):
            if piece["gap"]:
                result_rows.append({
                    "start_s": seconds(piece["start_us"]),
                    "end_s": seconds(piece["end_us"]),
                    "caption": "",
                    "status": "gap",
                })
                continue
            row = captions[piece_index]
            result_rows.append({
                "start_s": seconds(piece["start_us"]),
                "end_s": seconds(piece["end_us"]),
                "caption": row.get("caption", ""),
                "status": row.get("status", ""),
            })
        result = {
            "rotated_180": meta["rotated_180"],
            "duration_s": meta["duration_s"],
            "source": meta["source"],
            "source_span_s": meta["span_s"],
            "blocks": [{"start_s": seconds(a), "end_s": seconds(b)} for a, b in block_rows],
            "segments": result_rows,
        }
        (work / "official.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("CHAIN_DONE", len(result_rows), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("CHAIN_FAIL", flush=True)
        raise
