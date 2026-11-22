"""零样本标签。余弦差不是概率。低运动不参与这一步。"""

from __future__ import annotations

REASON = {
    "product_handling": "active_product_handling",
    "barcode_scanning": "barcode_scanning",
    "inspection": "product_inspection",
    "label_checking": "label_checking",
    "packaging": "packaging",
    "empty_workstation": "empty_workstation",
    "worker_leaving": "worker_left",
    "unrelated_cleaning": "unrelated_cleaning",
    "conversation": "conversation",
    "unrelated": "unrelated_activity",
    "observing": "observing_product",
    "waiting_scan": "waiting_for_scan",
    "standing_near": "standing_near_workstation",
}

POLARITY_OF = {
    "relevant": "positive",
    "irrelevant": "negative",
    "uncertain": "uncertain",
}


def _scores(frame: dict, view: str) -> dict | None:
    block = frame.get("scores") or {}
    return block.get(view)


def _reason(polarity: str, groups: dict, fallback: str) -> str:
    prefix = polarity + "/"
    subset = {key: value for key, value in groups.items() if key.startswith(prefix)}
    if not subset:
        return fallback
    group = max(subset, key=subset.get).split("/", 1)[1]
    return REASON.get(group, fallback)


def frame_label(frame: dict, view: str, temporal: dict) -> tuple[str, str, float]:
    """单帧标签。motion / frozen / duplicate 不读。"""
    if not frame.get("decode_ok", False):
        return "low_quality", "decode_error", 1.0
    if frame.get("black"):
        return "low_quality", "black_frame", 1.0
    if not frame.get("embedded", False):
        return "uncertain", "no_embedding", 0.0
    scores = _scores(frame, view)
    if not scores:
        return "uncertain", "no_embedding", 0.0
    pos = float(scores["positive"])
    neg = float(scores["negative"])
    unc = float(scores["uncertain"])
    margin = abs(pos - neg)
    groups = scores.get("groups") or {}
    relevant_margin = float(temporal["relevant_margin"])
    irrelevant_margin = float(temporal["irrelevant_margin"])
    if pos - max(neg, unc) >= relevant_margin:
        return "relevant", _reason("positive", groups, "active_product_handling"), margin
    if neg - max(pos, unc) >= irrelevant_margin:
        return "irrelevant", _reason("negative", groups, "unrelated_activity"), margin
    polarity = "uncertain"
    if pos >= neg and pos >= unc:
        polarity = "positive"
    elif neg >= pos and neg >= unc:
        polarity = "negative"
    return "uncertain", _reason(polarity, groups, "low_margin"), margin


def window_label(summary: dict, temporal: dict) -> tuple[str, str, float]:
    if int(summary["n_frames"]) == 0:
        return "uncertain", "no_sample", 0.0
    fail = float(summary["decode_fail_ratio"])
    black = float(summary["black_ratio"])
    if fail >= 0.5 or black >= 0.5:
        if fail >= black:
            return "low_quality", "decode_error", max(fail, black)
        return "low_quality", "black_frame", max(fail, black)
    if float(summary["blurry_ratio"]) >= float(summary["blur_window_ratio"]):
        return "low_quality", "blur", float(summary["blurry_ratio"])
    pos = float(summary["mean_positive"])
    neg = float(summary["mean_negative"])
    unc = float(summary["mean_uncertain"])
    margin = abs(pos - neg)
    groups = summary.get("groups") or {}
    relevant_margin = float(temporal["relevant_margin"])
    irrelevant_margin = float(temporal["irrelevant_margin"])
    if (
        pos - max(neg, unc) >= relevant_margin
        and float(summary["relevant_ratio"]) >= 0.5
    ):
        return "relevant", _reason("positive", groups, "active_product_handling"), margin
    if (
        neg - max(pos, unc) >= irrelevant_margin
        and float(summary["irrelevant_ratio"]) >= 0.5
    ):
        return "irrelevant", _reason("negative", groups, "unrelated_activity"), margin
    polarity = "uncertain"
    if pos >= neg and pos >= unc:
        polarity = "positive"
    elif neg >= pos and neg >= unc:
        polarity = "negative"
    return "uncertain", _reason(polarity, groups, "low_margin"), margin
