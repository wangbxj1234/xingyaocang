import unittest

from fuhe.pack_cycles import cycles_of


def episode():
    return {
        "episode_id": "ep",
        "seg": 0,
        "file": "seg_000.mp4",
        "src_start": 100.0,
        "_folder": "oss://bucket/ep",
        "segments": [
            {"start_s": 100.0, "end_s": 104.0, "caption": "hold bag with left hand, scan bag with scanner in right hand"},
            {"start_s": 104.0, "end_s": 112.0, "caption": "hold bag with left hand, insert plastic bag into blue bag with right hand"},
            {"start_s": 112.0, "end_s": 116.0, "caption": "place label on bag with right hand"},
            {"start_s": 116.0, "end_s": 118.0, "caption": "pick up bag with left hand"},
        ],
    }


class PackCyclesTest(unittest.TestCase):
    def test_one_order_runs_through_the_label(self):
        found = cycles_of(episode())
        self.assertEqual(len(found), 1)
        row = found[0]
        self.assertEqual(row["mp4_start_s"], 0.0)
        self.assertEqual(row["mp4_end_s"], 16.0)
        self.assertLessEqual(row["gap_s"], 1)
        self.assertIn("place label", row["captions"][-1]["caption"])

    def test_a_hole_inside_the_order_is_marked(self):
        data = episode()
        data["segments"][1]["start_s"] = 110.0
        data["segments"][1]["end_s"] = 118.0
        data["segments"][2]["start_s"] = 118.0
        data["segments"][2]["end_s"] = 122.0
        row = cycles_of(data)[0]
        self.assertGreater(row["gap_s"], 1)


if __name__ == "__main__":
    unittest.main()
