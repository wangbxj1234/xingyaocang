"""人工标窗口。标签只有 relevant / irrelevant / uncertain。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LABELS = ("relevant", "irrelevant", "uncertain")


def append_annotation(path: Path, record: dict) -> None:
    if record.get("label") not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    for key in ("episode_id", "start", "end"):
        if key not in record:
            raise ValueError(f"annotation missing {key}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_queue(window_rows: list[dict], limit: int) -> list[dict]:
    by_episode: dict[str, list[dict]] = {}
    for row in window_rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    episodes = sorted(by_episode)
    picked = []
    cursor = 0
    while len(picked) < limit and episodes:
        progressed = False
        for episode in episodes:
            rows = by_episode[episode]
            if cursor >= len(rows):
                continue
            picked.append(rows[cursor])
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
        cursor += 1
    return picked


def load_window_rows(out_dir: Path, view: str) -> list[dict]:
    rows = []
    folder = Path(out_dir) / "windows"
    if not folder.exists():
        return rows
    for path in sorted(folder.glob(f"*.{view}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_server(out_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    out_dir = Path(out_dir).resolve()
    labels = out_dir / "annotations.jsonl"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/label":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                record = json.loads(raw.decode("utf-8"))
                append_annotation(labels, record)
            except (ValueError, json.JSONDecodeError) as exc:
                body = str(exc).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            rel = self.path.split("?", 1)[0]
            if rel in ("", "/"):
                rel = "/reports/index.html"
            target = (out_dir / rel.lstrip("/")).resolve()
            if out_dir != target and out_dir not in target.parents:
                self.send_error(403)
                return
            if not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            kind = "text/html; charset=utf-8" if target.suffix == ".html" else "application/octet-stream"
            if target.suffix == ".json":
                kind = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(out_dir: Path, port: int = 8765) -> None:
    server = make_server(out_dir, port)
    print(f"FILTER_REPORT http://127.0.0.1:{server.server_address[1]}/reports/index.html", flush=True)
    server.serve_forever()
