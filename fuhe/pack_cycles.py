"""从已经标好的 seg mp4 里，切出扫码、装外包装、贴面单都有的一单。"""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OSS = ["/dev/shm/bin/ossutil", "-c", "/dev/shm/ossutilconfig-research"]


def bits(caption: str) -> tuple[bool, bool, bool]:
    text = caption.lower()
    scan = bool(re.search(r"\bscans?\b|\bscanning\b", text))
    label = bool(re.search(r"\blabels?\b", text) and re.search(r"place|apply|stick|put|slide|press|fold", text))
    pack = bool(re.search(r"\binsert\b", text) and re.search(r"bag|bubble|box|container|wrap", text))
    pack = pack or bool(re.search(r"\btape\b", text))
    return scan, label, pack


def cycles_of(data: dict) -> list[dict]:
    src = float(data["src_start"])
    rows = []
    clock = 0.0
    for seg in data.get("segments") or []:
        dur = float(seg["end_s"]) - float(seg["start_s"])
        scan, label, pack = bits(seg.get("caption") or "")
        rows.append({
            "clock": clock,
            "dur": dur,
            "start_s": float(seg["start_s"]),
            "end_s": float(seg["end_s"]),
            "caption": (seg.get("caption") or "").replace("\n", " ").strip(),
            "scan": scan,
            "label": label,
            "pack": pack,
        })
        clock += dur
    ends = [row["clock"] for row in rows if row["label"]]
    merged = []
    for stamp in ends:
        if not merged or stamp - merged[-1] > 6:
            merged.append(stamp)
    found = []
    prev = 0.0
    for end_t in merged:
        span = end_t - prev
        window = [row for row in rows if prev <= row["clock"] <= end_t]
        if 3 <= span <= 90 and any(row["scan"] for row in window) and any(row["pack"] for row in window):
            label_row = next(row for row in window if row["label"] and row["clock"] == end_t)
            clip_src_start = window[0]["start_s"]
            clip_src_end = label_row["end_s"]
            captions = []
            for row in window:
                if row["end_s"] <= clip_src_start or row["start_s"] >= clip_src_end:
                    continue
                captions.append({
                    "start_s": round(max(row["start_s"], clip_src_start) - clip_src_start, 3),
                    "end_s": round(min(row["end_s"], clip_src_end) - clip_src_start, 3),
                    "caption": row["caption"],
                })
            found.append({
                "episode_id": data["episode_id"],
                "seg": data["seg"],
                "file": data["file"],
                "folder": data["_folder"],
                "mp4_start_s": round(clip_src_start - src, 3),
                "mp4_end_s": round(clip_src_end - src, 3),
                "duration_s": round(clip_src_end - clip_src_start, 3),
                "caption_span_s": round(span + label_row["dur"], 3),
                "gap_s": round((clip_src_end - clip_src_start) - (span + label_row["dur"]), 3),
                "captions": captions,
            })
        prev = end_t
    return found


def fetch(row: dict) -> dict:
    url = row["folder"].rstrip("/") + "/" + row["stem"] + ".json"
    raw = subprocess.check_output(OSS + ["cat", url], text=True, errors="replace")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0:
        raise RuntimeError("no json " + url)
    data = json.loads(raw[start : end + 1])
    data["_folder"] = row["folder"].rstrip("/")
    return data


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="fuhe.pack_cycles")
    parser.add_argument("--catalog", type=Path, default=Path("/dev/shm/fuhe_flora/catalog.json"))
    parser.add_argument("--out", type=Path, default=Path("/dev/shm/fuhe_hf_cycles/cycles.jsonl"))
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text())
    rows = [row for row in catalog["rows"] if row.get("annotated")]
    cycles: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, row) for row in rows]
        for future in as_completed(futures):
            done += 1
            cycles.extend(cycles_of(future.result()))
            if done % 40 == 0 or done == len(rows):
                print("json", done, "/", len(rows), "cycles", len(cycles), flush=True)
    cycles.sort(key=lambda row: (row["episode_id"], int(row["seg"]), row["mp4_start_s"]))
    for index, row in enumerate(cycles):
        row["id"] = f"{index:05d}"
        row["video"] = f"videos/{index:05d}.mp4"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in cycles:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(row["duration_s"] for row in cycles) / 3600
    gaps = sum(1 for row in cycles if row["gap_s"] > 1)
    print("cycles", len(cycles), "video_hours", round(hours, 2), "gap_over_1s", gaps, flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
