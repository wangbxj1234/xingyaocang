import unittest

from siglip_filter.fusion import fuse_timeline
from siglip_filter.hands import bbox_overlap
from siglip_filter.idle import bridge_gaps, idle_intervals, transition_to_absent, transition_to_present


FULL = [[0, 0], [1, 0], [1, 1], [0, 1]]
LEFT = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]


def sample(stamp, present, motion=0.001, positive=0.08, negative=0.01, uncertain=0.01):
    return {
        "timestamp_s": stamp,
        "hand_present_in_work_area": present,
        "motion_score": motion,
        "motion_low": 0.02,
        "decode_ok": True,
        "embedded": True,
        "black": False,
        "siglip_view": "full",
        "scores": {
            "full": {
                "positive": positive,
                "negative": negative,
                "uncertain": uncertain,
                "groups": {},
            }
        },
    }


class HandIdleTest(unittest.TestCase):
    def test_hand_inside_work_area(self):
        overlap = bbox_overlap([10, 10, 40, 40], FULL, 100, 100)
        self.assertAlmostEqual(overlap, 1.0)
        outside = bbox_overlap([60, 10, 90, 40], LEFT, 100, 100)
        self.assertEqual(outside, 0.0)

    def test_one_sample_is_not_idle(self):
        rows = [sample(0, False)]
        self.assertEqual(idle_intervals(rows, 30, _cfg()), [])

    def test_one_second_grid_marks_five_seconds(self):
        cfg = dict(_cfg())
        cfg["sample_interval_s"] = 1
        cfg["bridge_s"] = 1
        short = [sample(float(t), False, motion=0.2) for t in range(4)]
        self.assertEqual(idle_intervals(short, 30, cfg), [])
        long = [sample(float(t), False, motion=0.2) for t in range(5)]
        found = idle_intervals(long, 30, cfg)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["start"], 0.0)
        self.assertAlmostEqual(found[0]["end"], 5.0)
        rows = [sample(float(t), False, motion=0.2) for t in range(16)]
        rows[6]["hand_present_in_work_area"] = True
        rows[7]["hand_present_in_work_area"] = True
        found = idle_intervals(rows, 30, cfg)
        self.assertEqual([round(row["start"]) for row in found], [0, 8])

    def test_five_seconds_without_a_hand_is_idle(self):
        rows = [sample(t, False, motion=0.2) for t in (0, 5)]
        found = idle_intervals(rows, 30, _cfg())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["label"], "idle_no_person")
        self.assertEqual(found[0]["reason"], "no_hand")
        self.assertAlmostEqual(found[0]["start"], 0.0)
        self.assertAlmostEqual(found[0]["end"], 10.0)

    def test_still_hands_are_invalid(self):
        rows = [sample(t, True, motion=0.001, positive=0.2) for t in (0, 5, 10)]
        found = idle_intervals(rows, 30, _cfg())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["label"], "present_but_unrelated")
        self.assertEqual(found[0]["reason"], "low_motion")
        segments = fuse_timeline(rows, found, 20, _temporal())
        self.assertTrue(all(row["label"] == "present_but_unrelated" for row in segments))

    def test_one_false_hand_does_not_split(self):
        rows = [sample(t, False) for t in range(0, 45, 5)]
        rows[3]["hand_present_in_work_area"] = True
        bridged = bridge_gaps(rows, 10)
        self.assertFalse(bridged[3]["hand_present_in_work_area"])
        found = idle_intervals(rows, 50, _cfg())
        self.assertEqual(len(found), 1)

    def test_short_moving_gap_stays_uncertain(self):
        rows = [sample(0, False, motion=0.2)]
        self.assertEqual(idle_intervals(rows, 20, _cfg()), [])
        segments = fuse_timeline(rows, [], 10, _temporal())
        self.assertEqual(segments[0]["reason"], "absent_high_motion")

    def test_idle_overrides_siglip(self):
        rows = [sample(t, False, positive=0.2) for t in range(0, 40, 5)]
        intervals = idle_intervals(rows, 50, _cfg())
        segments = fuse_timeline(rows, intervals, 40, _temporal())
        labels = {row["label"] for row in segments}
        self.assertIn("idle_no_person", labels)
        self.assertNotIn("kept", labels)

    def test_moving_hands_are_kept_without_siglip(self):
        rows = [
            sample(0, True, motion=0.1, positive=0.08, negative=0.01, uncertain=0.01),
            sample(10, True, motion=0.1, positive=0.01, negative=0.08, uncertain=0.01),
        ]
        segments = fuse_timeline(rows, [], 20, _temporal())
        self.assertTrue(all(row["label"] == "kept" for row in segments))
        self.assertTrue(all(row["reason"] == "hand_moving" for row in segments))

    def test_one_fps_edges(self):
        times = [13.0, 14.0, 15.0, 16.0]
        present = [True, True, False, False]
        self.assertEqual(transition_to_absent(times, present), 15.0)
        back = [False, False, True, True]
        self.assertEqual(transition_to_present(times, back), 15.0)
        self.assertIsNone(transition_to_absent(times, [False, False, False, False]))


def _cfg():
    return {
        "sample_interval_s": 5,
        "absence_confirm_s": 5,
        "motion_low": 0.02,
        "boundary_context_s": 0,
        "bridge_s": 5,
    }


def _temporal():
    return {
        "relevant_margin": 0.02,
        "irrelevant_margin": 0.02,
        "low_confidence_margin": 0.02,
    }


if __name__ == "__main__":
    unittest.main()
