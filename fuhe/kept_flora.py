"""Send each kept segment through the existing Flora chain.

Does not change Flora prompts. One segment at a time. Results go back
beside the mp4 on OSS. Original mcaps stay where they are.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zlib
from pathlib import Path

from siglip_filter.cut_keep import DEST_ROOT, upload

OSS = "/dev/shm/bin/ossutil"
WRITE_CFG = "/dev/shm/ossutilconfig-research"
FLORA_PY = os.environ.get("FLORA_PY", "/mnt/cpfs/zhongjiayu0757/flora/venv312/bin/python")
PY = os.environ.get("PY", "python3")


def oss(cfg: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([OSS, "-c", cfg, *args], check=check, capture_output=True, text=True)


def parse_oss_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("oss output has no json object")
    return json.loads(text[start : end + 1])


def object_exists(url: str) -> bool:
    result = oss(WRITE_CFG, ["stat", url], check=False)
    return result.returncode == 0 and "Content-Length" in result.stdout


def list_episode_dirs() -> list[str]:
    marker = ""
    found: list[str] = []
    while True:
        args = ["ls", DEST_ROOT + "/", "-d", "--limited-num", "1000"]
        if marker:
            args += ["--marker", marker]
        result = oss(WRITE_CFG, args)
        next_marker = ""
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("oss://") and stripped.endswith("/"):
                found.append(stripped.rstrip("/"))
            if stripped.startswith("NextMarker:"):
                next_marker = stripped.split(":", 1)[1].strip()
        if not next_marker or next_marker == marker:
            return found
        marker = next_marker


def choose_segment(rows: list[dict], shard: int = 0, shards: int = 1) -> dict | None:
    pending = []
    for row in rows:
        if row.get("annotated") or row.get("failed") or int(row.get("frames") or 0) < 2:
            continue
        key = ("%s/%s" % (row["episode_id"], row["seg"])).encode()
        if zlib.adler32(key) % shards != shard:
            continue
        pending.append(row)
    if not pending:
        return None
    return min(pending, key=lambda row: (float(row["duration_s"]), row["episode_id"], int(row["seg"])))


def pending_segments() -> list[dict]:
    rows = []
    for folder in list_episode_dirs():
        episode_id = folder.rsplit("/", 1)[-1]
        done = oss(WRITE_CFG, ["cat", folder + "/done.json"], check=False)
        if done.returncode != 0 or '"layout": "segments"' not in done.stdout:
            continue
        spans = parse_oss_json(oss(WRITE_CFG, ["cat", folder + "/spans.json"]).stdout)
        for seg in spans.get("segments") or []:
            name = str(seg["file"])
            stem = name[:-4] if name.endswith(".mp4") else name
            rows.append({
                "episode_id": episode_id,
                "folder": folder,
                "seg": int(seg["seg"]),
                "file": name,
                "stem": stem,
                "src_start": float(seg["src_start"]),
                "src_end": float(seg["src_end"]),
                "frames": int(seg.get("frames") or 0),
                "duration_s": float(seg.get("duration_s") or 0),
                "fps": float(spans.get("fps") or 0),
                "annotated": object_exists(folder + "/" + stem + ".json"),
                "failed": object_exists(folder + "/" + stem + ".flora_error.json"),
            })
    return rows


def _ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    frames = stream.get("nb_frames")
    return {
        "fps": float(num) / float(den),
        "frames": int(frames) if frames not in (None, "N/A", "") else 0,
        "duration": float(stream.get("duration") or 0),
    }


def _dump_jpg(mp4: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(mp4), "-start_number", "0", "-q:v", "2",
        str(dest / "%06d.jpg"),
    ])
    return len(list(dest.glob("*.jpg")))


def write_frames(mp4: Path, work: Path, job: dict) -> dict:
    info = _ffprobe(mp4)
    frames = work / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    src_count = _dump_jpg(mp4, frames)
    if src_count < 2:
        raise RuntimeError(f"{job['file']} decoded {src_count} frames")
    fps = info["fps"] or float(job["fps"]) or 15.0
    duration_s = src_count / fps
    if abs(fps - 15.0) > 0.05:
        src = work / "src_frames"
        if src.exists():
            shutil.rmtree(src)
        frames.rename(src)
        frames.mkdir()
        slots = int(round(duration_s * 15.0))
        for slot in range(slots):
            index = int(round((slot / 15.0) * fps))
            index = min(max(index, 0), src_count - 1)
            shutil.copyfile(src / f"{index:06d}.jpg", frames / f"{slot:06d}.jpg")
        shutil.rmtree(src)
        src_count = slots
        duration_s = slots / 15.0
    span_start = float(job["src_start"])
    span_end = span_start + duration_s
    meta = {
        "rotated_180": False,
        "source_frames": src_count,
        "slots": src_count,
        "fps": 15,
        "duration_s": duration_s,
        "span_s": [span_start, span_end],
        "epoch_ns": 0,
        "source": job["episode_id"],
        "mount": "kept",
        "topic": job["file"],
    }
    (work / "extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (work / "spans.json").write_text(
        json.dumps([{"start_s": span_start, "end_s": span_end}], indent=2),
        encoding="utf-8",
    )
    return meta


def flora_env() -> dict:
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("FLORA_CODE", "/mnt/cpfs/zhongjiayu0757/flora/code")
    env.setdefault("FLORA_MODEL_CACHE", "/mnt/cpfs/zhongjiayu0757/flora/models")
    env.setdefault("HF_HOME", "/mnt/cpfs/zhongjiayu0757/flora/hf_home")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def assert_gpus() -> None:
    subprocess.run(
        [
            FLORA_PY, "-c",
            "import torch,sys; n=torch.cuda.device_count(); "
            "biggest=max((torch.cuda.get_device_properties(i).total_memory for i in range(n)), default=0); "
            "ok=n>=1 and (biggest>=80*1024**3 or n==3); "
            "print('GPUS', n, 'biggest_gib', round(biggest/1024**3,1), flush=True); "
            "sys.exit(0 if ok else 1)",
        ],
        check=True,
        env=flora_env(),
    )


def run_stage(cmd: list[str], env: dict) -> None:
    subprocess.run(cmd, check=True, env=env)


def annotate_one(job: dict, work: Path) -> dict:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    local_mp4 = work / job["file"]
    oss(WRITE_CFG, ["cp", "-f", job["folder"] + "/" + job["file"], str(local_mp4)])
    try:
        meta = write_frames(local_mp4, work, job)
        env = flora_env()
        run_stage([PY, "-m", "fuhe.blocks", "--workdir", str(work)], env)
        run_stage([FLORA_PY, "-m", "fuhe.chain", "--workdir", str(work)], env)
        out = work / "out"
        run_stage([PY, "-m", "fuhe.postprocess", "--workdir", str(work), "--out", str(out)], env)
        posted = json.loads((out / f"{job['episode_id']}.json").read_text(encoding="utf-8"))
        posted.update({
            "seg": job["seg"],
            "file": job["file"],
            "src_start": job["src_start"],
            "src_end": job["src_end"],
            "timeline_start_s": meta["span_s"][0],
            "timeline_end_s": meta["span_s"][1],
        })
        dest = work / (job["stem"] + ".json")
        dest.write_text(json.dumps(posted, ensure_ascii=False, indent=2), encoding="utf-8")
        upload(dest, job["folder"] + "/" + dest.name)
        return posted
    finally:
        shutil.rmtree(work, ignore_errors=True)


def record_failure(job: dict, message: str, work: Path) -> None:
    fail = work / (job["stem"] + ".flora_error.json")
    work.mkdir(parents=True, exist_ok=True)
    fail.write_text(
        json.dumps({
            "episode_id": job["episode_id"],
            "seg": job["seg"],
            "file": job["file"],
            "error": message,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    upload(fail, job["folder"] + "/" + fail.name)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="fuhe.kept_flora")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    assert_gpus()
    while True:
        job = choose_segment(pending_segments(), args.shard, args.shards)
        if job is None:
            print("FLORA_WAIT", flush=True)
            if args.once:
                return
            time.sleep(60)
            continue
        print(
            "FLORA_PICK %s %s %.1fs" % (job["episode_id"], job["file"], job["duration_s"]),
            flush=True,
        )
        try:
            posted = annotate_one(job, args.work)
            print(
                "FLORA_DONE %s %s segments %d" % (job["episode_id"], job["file"], len(posted["segments"])),
                flush=True,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print("FLORA_FAIL %s %s %s" % (job["episode_id"], job["file"], message), flush=True)
            try:
                record_failure(job, message, args.work / "error")
            except Exception as upload_error:
                print("FLORA_FAIL_UPLOAD %s %s" % (job["episode_id"], upload_error), flush=True)
                raise
        if args.once:
            return


if __name__ == "__main__":
    main()
