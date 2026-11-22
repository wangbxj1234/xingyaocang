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

    def test_short_absence_is_not_idle(self):
        rows = [sample(t, False) for t in (0, 5)]
        found = idle_intervals(rows, 30, _cfg())
        self.assertEqual(found, [])

    def test_fifteen_seconds_without_a_hand_is_idle(self):
        rows = [sample(t, False) for t in (0, 5, 10, 15, 20)]
        found = idle_intervals(rows, 40, _cfg())
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["label"], "idle_no_person")
        self.assertAlmostEqual(found[0]["raw_start"], 0.0)
        self.assertAlmostEqual(found[0]["start"], 5.0)
        self.assertAlmostEqual(found[0]["end"], 20.0)

    def test_one_false_hand_does_not_split(self):
        rows = [sample(t, False) for t in range(0, 45, 5)]
        rows[3]["hand_present_in_work_area"] = True
        bridged = bridge_gaps(rows, 10)
        self.assertFalse(bridged[3]["hand_present_in_work_area"])
        found = idle_intervals(rows, 50, _cfg())
        self.assertEqual(len(found), 1)

    def test_moving_empty_station_stays_uncertain(self):
        rows = [sample(t, False, motion=0.2) for t in range(0, 40, 5)]
        self.assertEqual(idle_intervals(rows, 50, _cfg()), [])
        segments = fuse_timeline(rows, [], 40, _temporal())
        self.assertTrue(all(row["label"] == "uncertain" for row in segments))
        self.assertTrue(any(row["reason"] == "absent_high_motion" for row in segments))

    def test_idle_overrides_siglip(self):
        rows = [sample(t, False, positive=0.2) for t in range(0, 40, 5)]
        intervals = idle_intervals(rows, 50, _cfg())
        segments = fuse_timeline(rows, intervals, 40, _temporal())
        labels = {row["label"] for row in segments}
        self.assertIn("idle_no_person", labels)
        self.assertNotIn("active_relevant", labels)

    def test_present_relevant_and_unrelated(self):
        rows = [
            sample(0, True, positive=0.08, negative=0.01, uncertain=0.01),
            sample(10, True, positive=0.01, negative=0.08, uncertain=0.01),
        ]
        segments = fuse_timeline(rows, [], 20, _temporal())
        self.assertEqual(segments[0]["label"], "active_relevant")
        self.assertEqual(segments[1]["label"], "present_but_unrelated")

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
        "absence_confirm_s": 15,
        "motion_low": 0.02,
        "boundary_context_s": 5,
        "bridge_s": 10,
    }


def _temporal():
    return {
        "relevant_margin": 0.02,
        "irrelevant_margin": 0.02,
        "low_confidence_margin": 0.02,
    }


if __name__ == "__main__":
    unittest.main()
