import unittest

from fuhe.chain_math import blocks, chain, duration_us, frame_window, managed
from fuhe.postprocess import assert_covers, map_segments
from fuhe.preprocess import keep_spans


class ChainMath(unittest.TestCase):
    def test_360_blocks_match_the_run(self):
        rows = blocks(duration_us(360))
        self.assertEqual(rows, [
            (0, 120_000_000),
            (96_000_000, 216_000_000),
            (192_000_000, 312_000_000),
            (288_000_000, 360_000_000),
        ])
        self.assertEqual(frame_window(0, 120_000_000), (0, 1800))
        self.assertEqual(frame_window(288_000_000, 360_000_000), (4320, 1080))
        self.assertEqual(managed(0, 4, 0, 360_000_000), (0, 108_000_000))
        self.assertEqual(managed(3, 4, 288_000_000, 360_000_000), (300_000_000, 360_000_000))

    def test_merge_cuts_closer_than_half_a_second(self):
        duration = duration_us(360)
        kept = chain([10_000_000, 10_200_000], duration)
        self.assertEqual(kept, [0, 10_000_000, duration])

    def test_preprocess_keeps_the_whole_episode(self):
        self.assertEqual(keep_spans(12.5), [(0.0, 12.5)])

    def test_postprocess_shifts_onto_the_episode(self):
        official = {"segments": [
            {"start_s": 0.0, "end_s": 1.5, "caption": "a", "status": "complete"},
            {"start_s": 1.5, "end_s": 4.0, "caption": "", "status": "gap"},
        ]}
        rows = map_segments(official, 10.0)
        assert_covers(rows, 10.0, 14.0)
        self.assertEqual(rows[1]["status"], "gap")
        self.assertEqual(rows[1]["caption"], "")


if __name__ == "__main__":
    unittest.main()
