import unittest

from fuhe.chain import gpu_fits
from fuhe.kept_flora import choose_segment, parse_oss_json


class KeptFloraTest(unittest.TestCase):
    def test_parse_ignores_ossutil_trailer(self):
        text = '{"layout": "segments"}\n0.08(s) elapsed\n'
        self.assertEqual(parse_oss_json(text)["layout"], "segments")

    def test_shortest_unannotated_segment_goes_first(self):
        rows = [
            {"episode_id": "b", "seg": 0, "duration_s": 30, "frames": 100, "annotated": False, "failed": False},
            {"episode_id": "a", "seg": 1, "duration_s": 2, "frames": 24, "annotated": False, "failed": False},
            {"episode_id": "a", "seg": 0, "duration_s": 2, "frames": 1, "annotated": False, "failed": False},
            {"episode_id": "c", "seg": 0, "duration_s": 1, "frames": 10, "annotated": True, "failed": False},
        ]
        picked = choose_segment(rows)
        self.assertEqual((picked["episode_id"], picked["seg"]), ("a", 1))

    def test_one_h20_is_enough_and_three_small_cards_still_are(self):
        self.assertTrue(gpu_fits(1, 96 * 1024 ** 3))
        self.assertTrue(gpu_fits(3, 24 * 1024 ** 3))
        self.assertFalse(gpu_fits(1, 24 * 1024 ** 3))
        self.assertFalse(gpu_fits(2, 24 * 1024 ** 3))


if __name__ == "__main__":
    unittest.main()
