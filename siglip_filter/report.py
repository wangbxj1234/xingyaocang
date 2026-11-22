"""审片页。时间轴、缩略图、分数、运动，以及人工改标签。"""

from __future__ import annotations

import html
import json
from pathlib import Path


def _thumb(frames: list[dict], start: float, end: float) -> str | None:
    inside = [
        frame
        for frame in frames
        if frame.get("decode_ok") and start - 1e-6 <= float(frame["timestamp_s"]) <= end + 1e-6
    ]
    if not inside:
        return None
    mid = 0.5 * (start + end)
    best = min(inside, key=lambda frame: abs(float(frame["timestamp_s"]) - mid))
    return best.get("thumb")


def write_report(path: Path, manifest: dict, frames: list[dict], windows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episode = manifest["episode_id"]
    rows = []
    for window in windows:
        thumb = _thumb(frames, float(window["start"]), float(window["end"]))
        src = f"../cache/{episode}/{thumb}" if thumb else ""
        image = f'<img src="{html.escape(src)}" alt="">' if src else ""
        rows.append(
            "<tr>"
            f"<td>{window['start']:.1f}–{window['end']:.1f}</td>"
            f"<td class='{html.escape(window['label'])}'>{html.escape(window['label'])}</td>"
            f"<td>{html.escape(str(window['reason']))}</td>"
            f"<td>{float(window['confidence']):.3f}</td>"
            f"<td>{float(window['motion']):.3f}</td>"
            f"<td>{float(window['mean_positive']):.3f}</td>"
            f"<td>{float(window['mean_negative']):.3f}</td>"
            f"<td>{float(window['relevant_ratio']):.2f}</td>"
            f"<td>{image}</td>"
            "<td>"
            + "".join(
                (
                    f"<button type='button' onclick='mark({json.dumps(episode)},"
                    f"{float(window['start'])},{float(window['end'])},{json.dumps(label)})'>"
                    f"{label}</button>"
                )
                for label in ("relevant", "irrelevant", "uncertain")
            )
            + "</td></tr>"
        )
    bars = []
    duration = float(manifest["duration"]) or 1.0
    for segment in manifest["segments"]:
        width = 100.0 * (float(segment["end"]) - float(segment["start"])) / duration
        title = f"{segment['start']:.1f}-{segment['end']:.1f} {segment['label']} {segment['reason']}"
        bars.append(
            f"<i class='{html.escape(segment['label'])}' style='width:{width:.4f}%' "
            f"title='{html.escape(title)}'></i>"
        )
    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{html.escape(episode)}</title>
<style>
body {{ font: 14px/1.4 sans-serif; margin: 24px; }}
.timeline {{ display: flex; height: 28px; width: 100%; background: #eee; }}
.timeline i {{ display: block; height: 100%; }}
.relevant {{ background: #2e7d32; color: #fff; }}
.irrelevant {{ background: #c62828; color: #fff; }}
.uncertain {{ background: #ef6c00; color: #fff; }}
.low_quality {{ background: #546e7a; color: #fff; }}
td.relevant, td.irrelevant, td.uncertain, td.low_quality {{ padding: 2px 6px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 4px; text-align: left; vertical-align: top; }}
img {{ width: 96px; height: 96px; object-fit: cover; }}
button {{ margin-right: 4px; }}
</style>
</head>
<body>
<h1>{html.escape(episode)}</h1>
<p>{html.escape(str(manifest["task"]))} · {duration:.1f}s · {html.escape(str(manifest["model"]))}
· {html.escape(str(manifest["confidence_kind"]))}
· 余弦不是概率。低运动本身不会标成无关。</p>
<div class="timeline">{''.join(bars)}</div>
<table>
<thead><tr><th>时间</th><th>标签</th><th>原因</th><th>置信</th><th>运动</th><th>正</th><th>负</th><th>相关帧比例</th><th>画面</th><th>改正</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<script>
async function mark(episode, start, end, label) {{
  const row = {{episode_id: episode, start: start, end: end, label: label}};
  const key = "siglip-filter-labels";
  const kept = JSON.parse(localStorage.getItem(key) || "[]");
  kept.push(row);
  localStorage.setItem(key, JSON.stringify(kept));
  try {{
    const response = await fetch("/label", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(row)
    }});
    if (!response.ok) throw new Error(await response.text());
  }} catch (err) {{
    console.warn(err);
  }}
}}
</script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def write_index(path: Path, manifests: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for manifest in manifests:
        episode = html.escape(manifest["episode_id"])
        items.append(
            f"<li><a href='{episode}.html'>{episode}</a> "
            f"{html.escape(str(manifest['task']))} {float(manifest['duration']):.0f}s</li>"
        )
    path.write_text(
        "<!DOCTYPE html><meta charset='utf-8'><h1>filter reports</h1><ul>"
        + "".join(items)
        + "</ul>",
        encoding="utf-8",
    )
