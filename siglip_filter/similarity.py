"""余弦相似度。这里的数不是概率，也不做 softmax。"""

from __future__ import annotations

import numpy as np

from siglip_filter.prompts import Prompt


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def cosine(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    image = l2_normalize(image_emb)
    text = l2_normalize(text_emb)
    return image @ text.T


def aggregate(sim_row: np.ndarray, prompts: list[Prompt]) -> dict:
    """组内取提示词余弦的平均，类别再取各组的平均。各组权重相同。"""
    if len(sim_row) != len(prompts):
        raise ValueError("similarity row does not match the prompt list")
    grouped: dict[tuple[str, str], list[float]] = {}
    for score, prompt in zip(sim_row, prompts):
        grouped.setdefault((prompt.polarity, prompt.group), []).append(float(score))
    group_scores = {key: float(np.mean(values)) for key, values in grouped.items()}
    out = {"groups": {}}
    for polarity in ("positive", "negative", "uncertain"):
        scores = [value for (name, _group), value in group_scores.items() if name == polarity]
        out[polarity] = float(np.mean(scores)) if scores else 0.0
    for (polarity, group), value in group_scores.items():
        out["groups"][f"{polarity}/{group}"] = value
    return out
