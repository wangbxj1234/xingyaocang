import unittest

from siglip_filter.cut_keep import kept_ranges


class CutKeepTest(unittest.TestCase):
    def test_invalid_middle_leaves_two_pieces_in_order(self):
        intervals = [{"start": 5.0, "end": 8.0}]
        self.assertEqual(kept_ranges(intervals, 10.0), [(0.0, 5.0), (8.0, 10.0)])

    def test_touching_invalid_spans_join(self):
        intervals = [{"start": 2.0, "end": 4.0}, {"start": 4.0, "end": 6.0}]
        self.assertEqual(kept_ranges(intervals, 10.0), [(0.0, 2.0), (6.0, 10.0)])

    def test_everything_invalid(self):
        self.assertEqual(kept_ranges([{"start": 0.0, "end": 10.0}], 10.0), [])


if __name__ == "__main__":
    unittest.main()
