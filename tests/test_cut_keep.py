import unittest

from siglip_filter.cut_keep import kept_ranges, needs_rot180, segment_record


class CutKeepTest(unittest.TestCase):
    def test_invalid_middle_leaves_two_pieces_in_order(self):
        intervals = [{"start": 5.0, "end": 8.0}]
        self.assertEqual(kept_ranges(intervals, 10.0), [(0.0, 5.0), (8.0, 10.0)])

    def test_touching_invalid_spans_join(self):
        intervals = [{"start": 2.0, "end": 4.0}, {"start": 4.0, "end": 6.0}]
        self.assertEqual(kept_ranges(intervals, 10.0), [(0.0, 2.0), (6.0, 10.0)])

    def test_everything_invalid(self):
        self.assertEqual(kept_ranges([{"start": 0.0, "end": 10.0}], 10.0), [])

    def test_kept_pieces_stay_separate_segments(self):
        ranges = kept_ranges([{"start": 5.0, "end": 8.0}], 10.0)
        rows = [segment_record(i, start, end, 10, 15.0) for i, (start, end) in enumerate(ranges)]
        self.assertEqual([row["file"] for row in rows], ["seg_000.mp4", "seg_001.mp4"])
        self.assertEqual(rows[0]["src_end"], 5.0)
        self.assertEqual(rows[1]["src_start"], 8.0)
        self.assertNotIn("dst_start", rows[0])
        self.assertNotIn("dst_end", rows[1])

    def _hand(self, t, y0, y1, present=True, height=720):
        return {
            "timestamp_s": t,
            "hand_present_in_work_area": present,
            "hand_bbox": [0, y0, 10, y1],
            "image_height": height,
        }

    def test_top_edge_turns_the_episode(self):
        samples = [self._hand(i, 10, 200) for i in range(5)]
        self.assertTrue(needs_rot180(samples))

    def test_bottom_edge_stays(self):
        samples = [self._hand(i, 400, 700) for i in range(8)]
        self.assertFalse(needs_rot180(samples))

    def test_top_must_outnumber_bottom_and_reach_five(self):
        mixed = [self._hand(i, 10, 200) for i in range(5)]
        mixed += [self._hand(i, 400, 700) for i in range(5, 10)]
        self.assertFalse(needs_rot180(mixed))
        self.assertFalse(needs_rot180([self._hand(i, 10, 200) for i in range(4)]))

    def test_later_than_three_minutes_does_not_vote(self):
        late = [self._hand(180 + i, 10, 200) for i in range(6)]
        self.assertFalse(needs_rot180(late))

    def test_box_touching_both_edges_does_not_vote(self):
        samples = [self._hand(i, 10, 700) for i in range(6)]
        self.assertFalse(needs_rot180(samples))


if __name__ == "__main__":
    unittest.main()
