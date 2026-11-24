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

    def test_previous_order_sentences_stay_out(self):
        data = episode()
        data["segments"] = [
            {"start_s": 100.0, "end_s": 102.0, "caption": "place label on blue bag with right hand"},
            {"start_s": 102.0, "end_s": 104.0, "caption": "pick up blue bag with left hand, place blue bag on stack with right hand"},
            {"start_s": 104.0, "end_s": 106.0, "caption": "pick up the clear plastic bag with beige shirt with the left hand"},
            {"start_s": 106.0, "end_s": 108.0, "caption": "scan the plastic bag with the scanner"},
            {"start_s": 108.0, "end_s": 114.0, "caption": "insert plastic bag into blue bag with right hand"},
            {"start_s": 114.0, "end_s": 116.0, "caption": "place label on blue bag with right hand"},
        ]
        found = cycles_of(data)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["mp4_start_s"], 4.0)
        self.assertEqual(found[0]["mp4_end_s"], 16.0)
        self.assertTrue(found[0]["captions"][0]["caption"].startswith("pick up the clear"))

    def test_a_glued_sentence_stays_whole(self):
        data = episode()
        data["segments"] = [
            {"start_s": 100.0, "end_s": 102.0, "caption": "place label on blue bag with right hand"},
            {
                "start_s": 102.0,
                "end_s": 104.0,
                "caption": "place the blue bag on the stack with the left hand, pick up the clear plastic with white shirt with the left hand",
            },
            {"start_s": 104.0, "end_s": 106.0, "caption": "scan the plastic bag"},
            {"start_s": 106.0, "end_s": 112.0, "caption": "insert plastic bag into blue bag"},
            {"start_s": 112.0, "end_s": 114.0, "caption": "place label on blue bag with right hand"},
        ]
        row = cycles_of(data)[0]
        self.assertEqual(row["mp4_start_s"], 2.0)
        self.assertTrue(row["captions"][0]["caption"].startswith("place the blue bag on the stack"))

    def test_the_next_order_does_not_repeat_the_label(self):
        data = episode()
        data["segments"] = [
            {"start_s": 100.0, "end_s": 102.0, "caption": "pick up the clear plastic bag with beige shirt"},
            {"start_s": 102.0, "end_s": 104.0, "caption": "scan the plastic bag"},
            {"start_s": 104.0, "end_s": 110.0, "caption": "insert plastic bag into blue bag"},
            {"start_s": 110.0, "end_s": 112.0, "caption": "place label on blue bag with right hand"},
            {"start_s": 112.0, "end_s": 113.0, "caption": "place blue bag on stack with left hand"},
            {"start_s": 113.0, "end_s": 115.0, "caption": "pick up the clear plastic bag with white shirt"},
            {"start_s": 115.0, "end_s": 117.0, "caption": "scan the plastic bag"},
            {"start_s": 117.0, "end_s": 123.0, "caption": "insert plastic bag into blue bag"},
            {"start_s": 123.0, "end_s": 125.0, "caption": "place label on blue bag with right hand"},
        ]
        found = cycles_of(data)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]["mp4_end_s"], 12.0)
        self.assertEqual(found[1]["mp4_start_s"], 13.0)
        self.assertTrue(found[1]["captions"][0]["caption"].startswith("pick up the clear"))
        self.assertNotIn("place label", found[1]["captions"][0]["caption"])

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
