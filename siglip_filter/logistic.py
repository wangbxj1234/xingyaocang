"""缓存向量上的逻辑回归。按整条 episode 或采集日切分。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from siglip_filter.manifest import write_json
from siglip_filter.split import split_rows
from siglip_filter.temporal import in_window

LABELS = ("relevant", "irrelevant", "uncertain")


def pool_vector(timestamps, embedded, full, roi, start, end, duration, mode: str) -> np.ndarray | None:
    if mode not in ("full", "roi", "both"):
        raise ValueError(f"features must be full, roi, or both, got {mode}")
    if mode in ("roi", "both") and roi is None:
        raise RuntimeError("logistic features need roi embeddings")
    if mode in ("full", "both") and full is None:
        raise RuntimeError("logistic features need full-frame embeddings")
    is_last = abs(float(end) - float(duration)) <= 1e-6
    indexes = [
        index
        for index, (timestamp, ok) in enumerate(zip(timestamps, embedded))
        if ok and in_window(float(timestamp), float(start), float(end), is_last)
    ]
    if not indexes:
        return None
    chunks = []
    if mode in ("full", "both"):
        chunks.append(np.asarray(full, dtype=np.float32)[indexes].mean(axis=0))
    if mode in ("roi", "both"):
        chunks.append(np.asarray(roi, dtype=np.float32)[indexes].mean(axis=0))
    return np.concatenate(chunks, axis=0).astype(np.float32)


class LogisticPredictor:
    def __init__(self, directory: Path):
        import joblib

        directory = Path(directory)
        self.clf = joblib.load(directory / "classifier.joblib")
        self.scaler = joblib.load(directory / "scaler.joblib")
        config = json.loads((directory / "train_config.json").read_text(encoding="utf-8"))
        self.feature_mode = config["feature_mode"]

    def __call__(self, vector: np.ndarray) -> tuple[str, float]:
        scores = self.scaler.transform(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        proba = self.clf.predict_proba(scores)[0]
        classes = [str(name) for name in self.clf.classes_]
        best = int(np.argmax(proba))
        return classes[best], float(proba[best])


def apply_logistic(windows: list[dict], frames: list[dict], full, roi, predictor: LogisticPredictor, duration: float) -> list[dict]:
    timestamps = [frame["timestamp_s"] for frame in frames]
    embedded = [bool(frame.get("embedded")) for frame in frames]
    updated = []
    for window in windows:
        row = dict(window)
        if row["label"] == "low_quality":
            updated.append(row)
            continue
        vector = pool_vector(
            timestamps,
            embedded,
            full,
            roi,
            row["start"],
            row["end"],
            duration,
            predictor.feature_mode,
        )
        if vector is None:
            updated.append(row)
            continue
        label, confidence = predictor(vector)
        if label == "irrelevant" and row["reason"] == "empty_workstation":
            reason = "empty_workstation"
        else:
            reason = "logistic_" + label
        row["label"] = label
        row["reason"] = reason
        row["confidence"] = confidence
        updated.append(row)
    return updated


def _recall(truth: list[str], pred: list[str], label: str):
    total = sum(1 for item in truth if item == label)
    if total == 0:
        return None
    hit = sum(1 for left, right in zip(truth, pred) if left == label and right == label)
    return hit / total


def fit_logistic(examples: list[dict], feature_mode: str, out_dir: Path, val_fraction: float, seed: int, split_mode: str) -> dict:
    if len(examples) < 2:
        raise ValueError("logistic regression needs at least two labeled windows")
    labels = {row["label"] for row in examples}
    if not labels <= set(LABELS):
        raise ValueError(f"labels must be {LABELS}, got {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError("logistic regression needs at least two classes")
    train_rows, val_rows = split_rows(examples, split_mode, val_fraction, seed)
    if not train_rows:
        raise ValueError("training split is empty")
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("train needs scikit-learn") from exc

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.stack([row["vector"] for row in train_rows]))
    y_train = [row["label"] for row in train_rows]
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)
    y_hat = [str(item) for item in clf.predict(x_train)]
    metrics = {
        "split": split_mode,
        "feature_mode": feature_mode,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_episodes": sorted({row["episode_id"] for row in train_rows}),
        "val_episodes": sorted({row["episode_id"] for row in val_rows}),
        "train_relevant_recall": _recall(y_train, y_hat, "relevant"),
    }
    if val_rows:
        x_val = scaler.transform(np.stack([row["vector"] for row in val_rows]))
        y_val = [row["label"] for row in val_rows]
        y_val_hat = [str(item) for item in clf.predict(x_val)]
        metrics["relevant_recall"] = _recall(y_val, y_val_hat, "relevant")
        metrics["val_predicted"] = y_val_hat
        metrics["val_truth"] = y_val
    else:
        metrics["relevant_recall"] = None
        metrics["warning"] = "no held-out episode; adjacent frames were not split"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "classifier.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    write_json(out_dir / "label_map.json", {"classes": [str(name) for name in clf.classes_]})
    write_json(
        out_dir / "normalization.json",
        {"mean": [float(v) for v in scaler.mean_], "scale": [float(v) for v in scaler.scale_]},
    )
    write_json(
        out_dir / "train_config.json",
        {
            "feature_mode": feature_mode,
            "val_fraction": val_fraction,
            "seed": seed,
            "split": split_mode,
            "train_episodes": metrics["train_episodes"],
            "val_episodes": metrics["val_episodes"],
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    return metrics
