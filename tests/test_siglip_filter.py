import hashlib
import json
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from siglip_filter.annotate import append_annotation, build_queue, make_server
from siglip_filter.classify import frame_label
from siglip_filter.config import FilterConfig
from siglip_filter.decode import decode_many
from siglip_filter.encode import feature_tensor
from siglip_filter.logistic import fit_logistic
from siglip_filter.metrics import evaluate
from siglip_filter.pipeline import PilotLimit, run
from siglip_filter.prompts import prompts_for
from siglip_filter.quality import measure
from siglip_filter.similarity import aggregate, l2_normalize
from siglip_filter.prompts import Prompt
from siglip_filter.split import split_rows
from siglip_filter.temporal import apply_hysteresis, gate_short_blur, label_timeline


def sample_raw():
    return {
        "model": {"name": "stub-siglip", "dtype": "fp32", "batch_size": 4, "device": "cpu"},
        "sample": {"interval_s": 5, "image_size": 32},
        "roi": {"enabled": True, "x": 0.12, "y": 0.08, "w": 0.76, "h": 0.78},
        "quality": {
            "black_mean_max": 12,
            "sharp_var_min": -1,
            "duplicate_mad_max": 1.5,
            "frozen_ssim_min": 0.985,
            "blur_window_ratio": 0.8,
        },
        "temporal": {
            "window_s": 30,
            "relevant_margin": 0.02,
            "irrelevant_margin": 0.02,
            "low_confidence_margin": 0.02,
            "min_consecutive_irrelevant": 2,
            "boundary_context_s": 3,
            "smooth": 3,
            "idle_motion_max": 0.02,
            "decision_view": "full",
        },
        "classify": {
            "mode": "zero_shot",
            "features": "both",
            "split": "episode",
            "val_fraction": 0.2,
            "seed": 1,
        },
        "pilot": {"max_hours": 20},
        "workers": 1,
        "annotation": {"target_windows": 4},
    }


def sample_config():
    return FilterConfig.from_dict(sample_raw())


class StubEncoder:
    model_name = "stub-siglip"

    def __init__(self):
        self.image_calls = 0
        self.text_calls = 0
        self._polarity = {}
        for task in ("复核包装发货", "组套", "销退拆包", "衣服质检", "扫码", "分拣", "销退质检", "通用"):
            for prompt in prompts_for(task):
                self._polarity[prompt.text] = prompt.polarity

    def encode_texts(self, texts):
        self.text_calls += 1
        table = {
            "positive": [1.0, 0.0, 0.0],
            "negative": [0.0, 0.0, 1.0],
            "uncertain": [0.0, 1.0, 0.0],
        }
        rows = [table[self._polarity[text]] for text in texts]
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def encode_images(self, images):
        self.image_calls += 1
        rows = [np.asarray(image).astype(np.float32).mean(axis=(0, 1)) for image in images]
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def peak_memory_mb(self):
        return None


def solid(color):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr)


def split_color():
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    cut = int(40 * 0.8)
    arr[:, :cut] = (0, 0, 255)
    arr[:, cut:] = (255, 0, 0)
    return Image.fromarray(arr)


def write_frames(root, name, image, count=12):
    folder = Path(root) / name
    folder.mkdir()
    stamps = []
    for index in range(count):
        image.save(folder / f"{index:06d}.jpg")
        stamps.append(index * 5)
    (folder / "timestamps.json").write_text(json.dumps(stamps), encoding="utf-8")
    return {
        "episode_id": name,
        "video_path": str(folder),
        "task": "复核包装发货",
        "duration": count * 5,
        "source": "frames",
    }


def digest(folder):
    digest_ = hashlib.sha256()
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() == ".jpg":
            digest_.update(path.read_bytes())
    return digest_.hexdigest()


def scored(timestamp, positive, negative, uncertain, groups, **flags):
    frame = {
        "timestamp_s": timestamp,
        "decode_ok": True,
        "black": False,
        "blurry": False,
        "frozen": False,
        "duplicate": False,
        "embedded": True,
        "motion": 0.0,
        "scores": {
            "full": {
                "positive": positive,
                "negative": negative,
                "uncertain": uncertain,
                "groups": groups,
            }
        },
    }
    frame.update(flags)
    return frame


class Quality(unittest.TestCase):
    def test_black_blur_duplicate_and_frozen(self):
        cfg = sample_config().quality
        sharp = dict(cfg)
        sharp["sharp_var_min"] = 5
        black, _ = measure(solid((0, 0, 0)), None, sharp)
        self.assertTrue(black["black"])
        rng = np.random.RandomState(0)
        noisy = Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        noise, gray = measure(noisy, None, sharp)
        self.assertFalse(noise["blurry"])
        flat, flat_gray = measure(solid((180, 180, 180)), None, sharp)
        self.assertTrue(flat["blurry"])
        self.assertFalse(flat["black"])
        again, _ = measure(solid((180, 180, 180)), flat_gray, sharp)
        self.assertTrue(again["duplicate"])
        self.assertTrue(again["frozen"])
        self.assertEqual(again["motion"], 0.0)
        self.assertGreater(again["ssim"], 0.99)

    def test_motion_and_frozen_do_not_decide_the_label(self):
        temporal = sample_config().temporal
        groups = {"positive/product_handling": 0.6, "negative/empty_workstation": 0.0}
        quiet = scored(0, 0.6, 0.0, 0.0, groups, motion=0.0, frozen=True, duplicate=True)
        moving = scored(5, 0.6, 0.0, 0.0, groups, motion=0.9, frozen=False, duplicate=False)
        self.assertEqual(frame_label(quiet, "full", temporal), frame_label(moving, "full", temporal))
        self.assertEqual(frame_label(quiet, "full", temporal)[0], "relevant")


class Similarity(unittest.TestCase):
    def test_category_score_is_the_mean_of_groups(self):
        prompts = [
            Prompt("t", "positive", "a", "a1"),
            Prompt("t", "positive", "a", "a2"),
            Prompt("t", "positive", "b", "b1"),
        ]
        scores = aggregate(np.array([0.8, 0.4, 0.0]), prompts)
        self.assertAlmostEqual(scores["positive"], 0.3)
        self.assertNotAlmostEqual(scores["positive"], 0.4)

    def test_vectors_are_l2_normalized(self):
        out = l2_normalize(np.array([[3.0, 0.0, 0.0]]))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0)
        self.assertAlmostEqual(out[0, 0], 1.0)


class Timeline(unittest.TestCase):
    def _cfg(self):
        cfg = sample_config()
        return cfg.temporal, cfg.quality

    def test_two_irrelevant_windows_peel_context_and_can_be_long_idle(self):
        temporal, quality = self._cfg()
        groups = {"negative/empty_workstation": 0.5, "negative/conversation": 0.1}
        frames = [
            scored(t, 0.0, 0.5, 0.0, groups, motion=0.0)
            for t in range(0, 60, 5)
        ]
        _windows, segments = label_timeline(60, frames, "full", temporal, quality)
        self.assertEqual(segments[0]["label"], "uncertain")
        self.assertEqual(segments[0]["reason"], "boundary_context")
        self.assertEqual(segments[1]["label"], "irrelevant")
        self.assertEqual(segments[1]["reason"], "long_idle")
        self.assertEqual(segments[1]["start"], 3)
        self.assertEqual(segments[1]["end"], 57)
        self.assertEqual(segments[-1]["label"], "uncertain")

    def test_low_motion_relevant_stays(self):
        temporal, quality = self._cfg()
        groups = {"positive/product_handling": 0.6}
        frames = [scored(t, 0.6, 0.0, 0.0, groups, motion=0.0) for t in range(0, 60, 5)]
        _windows, segments = label_timeline(60, frames, "full", temporal, quality)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["label"], "relevant")
        self.assertEqual(segments[0]["reason"], "active_product_handling")
        self.assertLess(segments[0]["motion"], 0.02)

    def test_isolated_irrelevant_is_not_removed(self):
        temporal, _quality = self._cfg()
        windows = [
            {"label": "relevant", "start": 0, "end": 30, "confidence": 0.4, "mean_negative": 0.0, "mean_positive": 0.4},
            {"label": "irrelevant", "start": 30, "end": 60, "confidence": 0.4, "mean_negative": 0.4, "mean_positive": 0.0},
            {"label": "relevant", "start": 60, "end": 90, "confidence": 0.4, "mean_negative": 0.0, "mean_positive": 0.4},
        ]
        out = apply_hysteresis(windows, temporal)
        self.assertEqual([row["label"] for row in out], ["relevant", "uncertain", "relevant"])
        self.assertEqual(out[1]["reason"], "isolated_irrelevant")

    def test_low_confidence_gap_does_not_break_a_long_irrelevant_run(self):
        temporal, _quality = self._cfg()
        windows = [
            {"label": "irrelevant", "start": 0, "end": 30, "confidence": 0.4, "mean_negative": 0.4, "mean_positive": 0.0},
            {"label": "uncertain", "start": 30, "end": 60, "confidence": 0.001, "mean_negative": 0.2, "mean_positive": 0.1},
            {"label": "irrelevant", "start": 60, "end": 90, "confidence": 0.4, "mean_negative": 0.4, "mean_positive": 0.0},
        ]
        out = apply_hysteresis(windows, temporal)
        self.assertEqual([row["label"] for row in out], ["irrelevant", "uncertain", "irrelevant"])

    def test_short_blur_is_kept(self):
        temporal, _quality = self._cfg()
        one = gate_short_blur(
            [{"label": "low_quality", "reason": "blur", "start": 0, "end": 30}],
            temporal,
        )
        self.assertEqual(one[0]["label"], "uncertain")
        self.assertEqual(one[0]["reason"], "short_blur")
        two = gate_short_blur(
            [
                {"label": "low_quality", "reason": "blur", "start": 0, "end": 30},
                {"label": "low_quality", "reason": "blur", "start": 30, "end": 60},
            ],
            temporal,
        )
        self.assertTrue(all(row["label"] == "low_quality" for row in two))

    def test_black_outranks_a_positive_score(self):
        temporal, quality = self._cfg()
        groups = {"positive/product_handling": 0.8}
        frames = [
            scored(t, 0.8, 0.0, 0.0, groups, black=True)
            for t in range(0, 60, 5)
        ]
        _windows, segments = label_timeline(60, frames, "full", temporal, quality)
        self.assertIn("low_quality", [row["label"] for row in segments])
        self.assertTrue(any(row["reason"] == "black_frame" for row in segments))


class SplitAndMetrics(unittest.TestCase):
    def test_episode_split_keeps_a_video_together(self):
        rows = []
        for episode, date in (("a", "2026-11-01"), ("b", "2026-11-02"), ("c", "2026-11-02")):
            rows.append({"episode_id": episode, "date": date, "start": 0})
            rows.append({"episode_id": episode, "date": date, "start": 30})
        train, val = split_rows(rows, "episode", 0.34, 1)
        train_ids = {row["episode_id"] for row in train}
        val_ids = {row["episode_id"] for row in val}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(len(train_ids) + len(val_ids), 3)
        for episode in train_ids | val_ids:
            sides = {row["episode_id"] for row in train if row["episode_id"] == episode}
            sides |= {row["episode_id"] for row in val if row["episode_id"] == episode}
            self.assertEqual(sides, {episode})

    def test_recall_counts_uncertain_as_kept(self):
        manual = [
            {"episode_id": "e", "start": 0, "end": 30, "label": "relevant"},
            {"episode_id": "e", "start": 30, "end": 60, "label": "irrelevant"},
        ]
        manifest = {
            "decision_view": "full",
            "segments": [
                {"start": 0, "end": 30, "label": "uncertain"},
                {"start": 30, "end": 60, "label": "irrelevant"},
            ],
        }
        report = evaluate(manual, {"e": manifest}, "full")
        self.assertEqual(report["relevant_segment_recall"], 1.0)
        self.assertEqual(report["relevant_segment_recall_strict"], 0.0)
        self.assertEqual(report["irrelevant_segment_precision"], 1.0)
        self.assertEqual(report["relevant_duration_retention"], 1.0)
        self.assertEqual(report["irrelevant_duration_removal"], 1.0)


class Pipeline(unittest.TestCase):
    def test_color_pipeline_keeps_low_motion_work_and_separates_roi(self):
        raw = sample_raw()
        raw["roi"] = {"enabled": True, "x": 0.8, "y": 0.0, "w": 0.2, "h": 1.0}
        cfg = FilterConfig.from_dict(raw)
        encoder = StubEncoder()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            source.mkdir()
            blue = write_frames(source, "blue", split_color())
            red = write_frames(source, "red", solid((255, 0, 0)))
            before = {name: digest(source / name) for name in ("blue", "red")}
            out = root / "out"
            manifests = run(cfg, [blue, red], out, encoder=encoder)
            after = {name: digest(source / name) for name in ("blue", "red")}
            self.assertEqual(before, after)
            by_id = {row["episode_id"]: row for row in manifests}
            blue_labels = [row["label"] for row in by_id["blue"]["segments"]]
            self.assertIn("irrelevant", blue_labels)
            self.assertTrue(
                any(row["reason"] == "long_idle" for row in by_id["blue"]["segments"])
            )
            self.assertTrue(
                all(row["label"] == "relevant" for row in by_id["blue"]["views"]["roi"]["segments"])
            )
            self.assertEqual(by_id["red"]["segments"][0]["label"], "relevant")
            self.assertEqual(by_id["red"]["segments"][0]["reason"], "active_product_handling")
            self.assertLess(by_id["red"]["segments"][0]["motion"], 0.02)
            self.assertEqual(by_id["red"]["confidence_kind"], "cosine_margin")
            frames = [
                json.loads(line)
                for line in (out / "frames" / "red.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(row["frozen"] for row in frames))
            html = (out / "reports" / "red.html").read_text(encoding="utf-8")
            self.assertIn("red", html)
            self.assertIn("relevant", html)
            image_calls = encoder.image_calls
            text_calls = encoder.text_calls
            checkpoint = out / "checkpoints" / "red.json"
            checkpoint.unlink()
            run(cfg, [red], out, encoder=encoder)
            self.assertEqual(encoder.image_calls, image_calls)
            self.assertEqual(encoder.text_calls, text_calls)

    def test_pilot_refuses_the_full_corpus(self):
        cfg = sample_config()
        episode = {
            "episode_id": "all",
            "video_path": "/no/such/video.mcap",
            "task": "组套",
            "duration": 3000 * 3600,
        }
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(PilotLimit):
                run(cfg, [episode], out, encoder=StubEncoder())
            with self.assertRaises(PilotLimit):
                run(cfg, [episode], out, allow_full=True, encoder=StubEncoder())
            report = out / "validation.json"
            report.write_text(json.dumps({"relevant_segment_recall": 0.99}), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                run(cfg, [episode], out, allow_full=True, validation=report, encoder=StubEncoder())

    def test_two_workers_decode_without_the_model(self):
        raw = sample_raw()
        raw["workers"] = 2
        cfg = FilterConfig.from_dict(raw)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            episodes = [
                write_frames(root / "src", "a", solid((255, 0, 0))),
                write_frames(root / "src", "b", solid((0, 0, 255))),
            ]
            decode_many(episodes, cfg, root / "out")
            for episode in episodes:
                index = root / "out" / "cache" / episode["episode_id"] / "frames_index.json"
                self.assertTrue(index.exists())
                payload = json.loads(index.read_text(encoding="utf-8"))
                self.assertEqual(len(payload["frames"]), 12)


class Logistic(unittest.TestCase):
    def test_fit_holds_out_whole_episodes(self):
        examples = []
        for index, (episode, label, vector) in enumerate(
            (
                ("a", "relevant", [3.0, 0.0, 0.0, 0.0]),
                ("b", "relevant", [2.5, 0.2, 0.0, 0.0]),
                ("c", "irrelevant", [0.0, 3.0, 0.0, 0.0]),
                ("d", "irrelevant", [0.1, 2.5, 0.0, 0.0]),
                ("e", "uncertain", [0.0, 0.0, 3.0, 0.0]),
                ("f", "uncertain", [0.0, 0.2, 2.5, 0.0]),
            )
        ):
            examples.append(
                {
                    "episode_id": episode,
                    "start": 0,
                    "end": 30,
                    "label": label,
                    "vector": np.asarray(vector, dtype=np.float32),
                }
            )
            del index
        with TemporaryDirectory() as tmp:
            metrics = fit_logistic(examples, "full", Path(tmp), 0.2, 1, "episode")
            self.assertFalse(set(metrics["train_episodes"]) & set(metrics["val_episodes"]))
            self.assertTrue((Path(tmp) / "classifier.joblib").exists())
            self.assertTrue((Path(tmp) / "normalization.json").exists())
            self.assertIn("classes", json.loads((Path(tmp) / "label_map.json").read_text(encoding="utf-8")))
            norm = json.loads((Path(tmp) / "normalization.json").read_text(encoding="utf-8"))
            self.assertEqual(len(norm["mean"]), 4)


class Annotation(unittest.TestCase):
    def test_queue_round_robin_and_post(self):
        rows = []
        for episode in ("a", "b"):
            for start in (0, 30, 60):
                rows.append({"episode_id": episode, "start": start, "end": start + 30, "label": "relevant"})
        picked = build_queue(rows, 4)
        self.assertEqual([row["episode_id"] for row in picked], ["a", "b", "a", "b"])
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "reports").mkdir()
            (out / "reports" / "index.html").write_text("<p>ok</p>", encoding="utf-8")
            server = make_server(out, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                body = json.dumps(
                    {"episode_id": "a", "start": 0, "end": 30, "label": "uncertain"}
                ).encode("utf-8")
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/label",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 200)
                saved = [
                    json.loads(line)
                    for line in (out / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(saved[0]["label"], "uncertain")
                with self.assertRaises(ValueError):
                    append_annotation(out / "annotations.jsonl", {"episode_id": "a", "start": 0, "end": 1, "label": "delete"})
            finally:
                server.shutdown()
                server.server_close()


class ConfigAndModel(unittest.TestCase):
    def test_boundary_must_stay_between_two_and_five_seconds(self):
        raw = sample_raw()
        raw["temporal"]["boundary_context_s"] = 1
        with self.assertRaises(ValueError):
            FilterConfig.from_dict(raw)

    def test_pilot_yaml(self):
        cfg = FilterConfig.load(Path(__file__).resolve().parents[1] / "configs" / "filter_pilot.yaml")
        self.assertEqual(cfg.model_name, "google/siglip2-base-patch16-224")
        self.assertEqual(cfg.interval_s, 5)
        self.assertEqual(cfg.temporal["window_s"], 30)
        self.assertEqual(cfg.pilot_max_hours, 20)
        self.assertEqual(cfg.temporal["min_consecutive_irrelevant"], 2)

    def test_feature_tensor_prefers_pooler_output(self):
        class Out:
            pooler_output = "pooled"

        self.assertEqual(feature_tensor(Out()), "pooled")
        self.assertEqual(feature_tensor(("image",)), "image")

    def test_aliases_share_prompts(self):
        self.assertEqual(
            [prompt.text for prompt in prompts_for("复核包装发货1")],
            [prompt.text for prompt in prompts_for("复核包装发货")],
        )
        polarities = {prompt.polarity for prompt in prompts_for("没写任务名")}
        self.assertEqual(polarities, {"positive", "negative", "uncertain"})


if __name__ == "__main__":
    unittest.main()
