"""只在人工标过的时间上算。最重要的是相关片段召回，不确定算留下。"""

from __future__ import annotations


def _overlap(left0: float, left1: float, right0: float, right1: float) -> float:
    return max(0.0, min(left1, right1) - max(left0, right0))


def _segments(manifest: dict, view: str) -> list[dict]:
    if view == manifest.get("decision_view"):
        return manifest["segments"]
    views = manifest.get("views") or {}
    if view in views:
        return views[view]["segments"]
    raise KeyError(f"manifest has no view {view}")


def evaluate(manual: list[dict], manifests: dict[str, dict], view: str = "full") -> dict:
    relevant = 0.0
    relevant_kept = 0.0
    relevant_strict = 0.0
    irrelevant = 0.0
    irrelevant_removed = 0.0
    predicted_irrelevant_annotated = 0.0
    predicted_irrelevant_correct = 0.0
    windows = 0
    windows_hit = 0
    for row in manual:
        if row["label"] not in ("relevant", "irrelevant", "uncertain"):
            raise ValueError(f"manual label must be relevant, irrelevant, or uncertain, got {row['label']}")
        manifest = manifests[row["episode_id"]]
        segments = _segments(manifest, view)
        start = float(row["start"])
        end = float(row["end"])
        span = max(0.0, end - start)
        if span == 0:
            continue
        kept = 0.0
        strict = 0.0
        removed = 0.0
        for segment in segments:
            overlap = _overlap(start, end, float(segment["start"]), float(segment["end"]))
            if overlap <= 0:
                continue
            if row["label"] == "relevant":
                if segment["label"] in ("relevant", "uncertain"):
                    kept += overlap
                if segment["label"] == "relevant":
                    strict += overlap
            if row["label"] == "irrelevant" and segment["label"] == "irrelevant":
                removed += overlap
        if row["label"] == "relevant":
            relevant += span
            relevant_kept += kept
            relevant_strict += strict
            windows += 1
            if kept >= 0.5 * span:
                windows_hit += 1
        if row["label"] == "irrelevant":
            irrelevant += span
            irrelevant_removed += removed
        for segment in segments:
            if segment["label"] != "irrelevant":
                continue
            overlap = _overlap(start, end, float(segment["start"]), float(segment["end"]))
            predicted_irrelevant_annotated += overlap
            if row["label"] == "irrelevant":
                predicted_irrelevant_correct += overlap
    return {
        "view": view,
        "relevant_segment_recall": _ratio(relevant_kept, relevant),
        "relevant_segment_recall_strict": _ratio(relevant_strict, relevant),
        "relevant_window_recall": _ratio(windows_hit, windows),
        "irrelevant_segment_precision": _ratio(
            predicted_irrelevant_correct, predicted_irrelevant_annotated
        ),
        "relevant_duration_retention": _ratio(relevant_kept, relevant),
        "irrelevant_duration_removal": _ratio(irrelevant_removed, irrelevant),
        "relevant_seconds": relevant,
        "irrelevant_seconds": irrelevant,
        "note": "relevant recall counts uncertain as kept; cosine scores are not probabilities",
    }


def _ratio(numer: float, denom: float):
    if denom <= 0:
        return None
    return float(numer / denom)
