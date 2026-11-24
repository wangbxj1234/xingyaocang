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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from siglip_filter.cut_keep import DEST_ROOT, upload

OSS = "/dev/shm/bin/ossutil"
WRITE_CFG = "/dev/shm/ossutilconfig-research"
FLORA_PY = os.environ.get("FLORA_PY", "/mnt/cpfs/zhongjiayu0757/flora/venv312/bin/python")
PY = os.environ.get("PY", "python3")
CATALOG_MAX_AGE_S = 90


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
    return max(pending, key=lambda row: (float(row["duration_s"]), row["episode_id"], int(row["seg"])))


def names_from_ls(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        url = ""
        for token in line.split():
            if token.startswith("oss://"):
                url = token
        if not url or url.endswith("/"):
            continue
        names.add(url.rsplit("/", 1)[-1])
    return names


def list_object_names(folder: str) -> set[str]:
    marker = ""
    names: set[str] = set()
    while True:
        args = ["ls", folder + "/", "--limited-num", "1000"]
        if marker:
            args += ["--marker", marker]
        result = oss(WRITE_CFG, args)
        names.update(names_from_ls(result.stdout))
        next_marker = ""
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("NextMarker:"):
                next_marker = stripped.split(":", 1)[1].strip()
        if not next_marker or next_marker == marker:
            return names
        marker = next_marker


def rows_from_listing(episode_id: str, folder: str, spans: dict, names: set[str]) -> list[dict]:
    rows = []
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
            "rotated_180": bool(spans.get("rotated_180")),
            "annotated": (stem + ".json") in names,
            "failed": (stem + ".flora_error.json") in names,
        })
    return rows


def _episode_rows(folder: str) -> list[dict]:
    episode_id = folder.rsplit("/", 1)[-1]
    try:
        names = list_object_names(folder)
        if "done.json" not in names or "spans.json" not in names:
            return []
        done = oss(WRITE_CFG, ["cat", folder + "/done.json"], check=False)
        if done.returncode != 0 or '"layout": "segments"' not in done.stdout:
            return []
        spans = parse_oss_json(oss(WRITE_CFG, ["cat", folder + "/spans.json"]).stdout)
        return rows_from_listing(episode_id, folder, spans, names)
    except Exception as exc:
        print("FLORA_LIST_FAIL %s %s: %s" % (episode_id, type(exc).__name__, exc), flush=True)
        return []


def pending_segments() -> list[dict]:
    folders = list_episode_dirs()
    started = time.time()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_episode_rows, folder) for folder in folders]
        for future in as_completed(futures):
            rows.extend(future.result())
    annotated = sum(1 for row in rows if row["annotated"])
    print(
        "FLORA_CATALOG episodes %d segments %d annotated %d %.0fs"
        % (len(folders), len(rows), annotated, time.time() - started),
        flush=True,
    )
    return rows


def _read_catalog(path: Path) -> list[dict] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - float(payload.get("built_at") or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    rows = payload.get("rows") if age < CATALOG_MAX_AGE_S else None
    return rows if isinstance(rows, list) else None


def _write_catalog(path: Path, rows: list[dict], built_at: float | None = None) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"built_at": time.time() if built_at is None else built_at, "rows": rows}),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_catalog(path: Path, *, fresh: bool = False) -> list[dict]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = None if fresh else (_read_catalog(path) if path.is_file() else None)
        if cached is not None:
            return cached
        print("FLORA_CATALOG_BUILD", flush=True)
        rows = pending_segments()
        _write_catalog(path, rows)
        return rows


def after_idle(confirmed_empty: bool, *, drain: bool, once: bool) -> str:
    """队列空了之后：exit 结束，rebuild 丢掉缓存再看一次，wait 过一分钟再看。"""
    if drain:
        return "exit" if confirmed_empty else "rebuild"
    if once:
        return "exit"
    return "wait"


def mark_catalog(path: Path, episode_id: str, seg: int, *, failed: bool) -> None:
    import fcntl

    if not path.is_file():
        return
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for row in payload.get("rows") or []:
            if row.get("episode_id") == episode_id and int(row.get("seg")) == seg:
                if failed:
                    row["failed"] = True
                else:
                    row["annotated"] = True
        built_at = payload.get("built_at")
        _write_catalog(
            path,
            payload.get("rows") or [],
            float(built_at) if isinstance(built_at, (int, float)) else None,
        )


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
        "rotated_180": False,  # 成片已经转过。blocks 看到 true 会再翻一次。
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
            "rotated_180": bool(job.get("rotated_180")),
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
    parser.add_argument("--drain", action="store_true")
    args = parser.parse_args(argv)
    assert_gpus()
    catalog = Path(args.work).resolve().parent / "catalog.json"
    confirmed_empty = False
    while True:
        job = choose_segment(
            load_catalog(catalog, fresh=confirmed_empty), args.shard, args.shards
        )
        if job is None:
            action = after_idle(confirmed_empty, drain=args.drain, once=args.once)
            if action == "exit":
                print("FLORA_DRAIN" if args.drain else "FLORA_WAIT", flush=True)
                return
            if action == "rebuild":
                confirmed_empty = True
                continue
            print("FLORA_WAIT", flush=True)
            time.sleep(60)
            continue
        confirmed_empty = False
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
            try:
                mark_catalog(catalog, job["episode_id"], int(job["seg"]), failed=False)
            except Exception as mark_error:
                print("FLORA_CATALOG_MARK %s" % mark_error, flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print("FLORA_FAIL %s %s %s" % (job["episode_id"], job["file"], message), flush=True)
            try:
                record_failure(job, message, args.work / "error")
            except Exception as upload_error:
                print("FLORA_FAIL_UPLOAD %s %s" % (job["episode_id"], upload_error), flush=True)
                raise
            try:
                mark_catalog(catalog, job["episode_id"], int(job["seg"]), failed=True)
            except Exception as mark_error:
                print("FLORA_CATALOG_MARK %s" % mark_error, flush=True)
        if args.once:
            return


if __name__ == "__main__":
    main()
